#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

template <int topk, int num_experts>
__global__ void token_unzip_kernel(
    const phi::bfloat16 *__restrict__ X,
    const int *__restrict__ routemap_topk,
    const phi::bfloat16 *__restrict__ probs_topk_in,
    phi::bfloat16 *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    phi::bfloat16 *__restrict__ probs_unzipped_out,
    int *__restrict__ expert_idx_unzipped,
    int *__restrict__ atomic_extended_offset_counter,
    int *__restrict__ row_valid,
    const int total_zipped_tokens_num,
    const int total_unzipped_tokens_num,
    const int token_length) {
  const __nv_bfloat16 *probs_topk =
      reinterpret_cast<const __nv_bfloat16 *>(probs_topk_in);
  __nv_bfloat16 *probs_unzipped =
      reinterpret_cast<__nv_bfloat16 *>(probs_unzipped_out);
  // 每个线程处理一行数据
  const int row_idx = blockIdx.x;
  // 仅在线程组2中被更新，不初始化
  int extended_row_offset;

  if (row_idx < total_unzipped_tokens_num) [[likely]] {
    // 线程组0，
    // 主要处理topk和增广部分的行索引、处理专家广播后的行表、一对一搬移
    if (row_idx < total_zipped_tokens_num) [[likely]] {
      if (threadIdx.x == 0) [[unlikely]] {
        // 寄存器加载、存储，消耗2xtopk 个reg
        // 每行只有一次非广播的机会
        bool isFirst = true;
        int local_expert_rowmap[num_experts];
// 填入非法值，避免误用（0为合法rowidx）
#pragma unroll
        for (int i = 0; i < num_experts; i++) {
          local_expert_rowmap[i] = -1;
        }
        for (int i = 0; i < topk; i++) {
          int this_expert_idx = routemap_topk[row_idx * topk + i];
          __nv_bfloat16 this_expert_prob = probs_topk[row_idx * topk + i];
          if (this_expert_idx < num_experts && this_expert_idx >= 0)
              [[unlikely]] {
            if (isFirst) [[likely]] {
              isFirst = false;
              probs_unzipped[row_idx] = this_expert_prob;
              expert_idx_unzipped[row_idx] = this_expert_idx;
              local_expert_rowmap[this_expert_idx] = row_idx;
            } else {
              // 增广部分， 原子更新行偏置
              extended_row_offset =
                  atomicAdd(&atomic_extended_offset_counter[0], 1);
              int extended_row_idx =
                  total_zipped_tokens_num + extended_row_offset;
              // 立即唤起相关的线程组1，减少忙等,
              // 也强保证zipped_expertwise_rowmap的变动对组1可见
              atomicExch(&zipped_expertwise_rowmap[extended_row_idx], row_idx);
              atomicExch(&row_valid[extended_row_offset], 1);
              probs_unzipped[extended_row_idx] = this_expert_prob;
              expert_idx_unzipped[extended_row_idx] = this_expert_idx;
              // 处理专家广播后的行表，用于zip进行收集
              local_expert_rowmap[this_expert_idx] = extended_row_idx;
            }
          }
        }
// 将合法值和未被触碰的非法值返回给zipped_expertwise_rowmap
#pragma unroll
        for (int i = 0; i < num_experts; i++) {
          zipped_expertwise_rowmap[row_idx * num_experts + i] =
              local_expert_rowmap[i];
        }
      }
      //这个syncthread可能并不必要，但尽可能为了不让线程间差太多，还是这样吧。
      __syncthreads();
      // 搬第一次出现的数据
      for (int i = threadIdx.x; i < token_length; i += blockDim.x) {
        X_unzipped[row_idx * token_length + i] = X[row_idx * token_length + i];
      }
    } else {  // 线程组1， 忙等、并发处理数据搬移
      if (threadIdx.x == 0) {
        int extended_row_offset = row_idx - total_zipped_tokens_num;
        // 忙等该行的 row_valid变为1
        while (!atomicExch(&row_valid[extended_row_offset], 0)) {
        }
      }
      __syncthreads();  // 所有该组线程都等0完成等待
      // 搬
      for (int i = threadIdx.x; i < token_length; i += blockDim.x) {
        int origin_row = zipped_expertwise_rowmap[row_idx];
        X_unzipped[row_idx * token_length + i] =
            X[origin_row * token_length + i];
      }
    }
  }
}

void dispatch_tokens_unzip(const paddle::Tensor &X,
                           const paddle::Tensor &expert_routemap_topk,
                           const paddle::Tensor &expert_prob_topk,
                           paddle::Tensor &X_unzipped,
                           paddle::Tensor &zipped_expertwise_rowmap,
                           paddle::Tensor &token_prob_unzipped,
                           paddle::Tensor &expert_idx_unzipped,
                           paddle::Tensor &atomic_extended_offset_counter,
                           paddle::Tensor &row_valid,
                           const int total_zipped_tokens_num,
                           const int total_unzipped_tokens_num,
                           const int token_length,
                           const int topk,
                           const int num_experts) {
  dim3 grid, block;
  grid.x = total_unzipped_tokens_num;
  block.x = 256;
  if (topk == 8 && num_experts == 4) {
    token_unzip_kernel<8, 4><<<grid, block, 0, X.stream()>>>(
        X.data<phi::bfloat16>(),
        expert_routemap_topk.data<int>(),
        expert_prob_topk.data<phi::bfloat16>(),
        X_unzipped.data<phi::bfloat16>(),
        zipped_expertwise_rowmap.data<int>(),
        token_prob_unzipped.data<phi::bfloat16>(),
        expert_idx_unzipped.data<int>(),
        atomic_extended_offset_counter.data<int>(),
        row_valid.data<int>(),
        total_zipped_tokens_num,
        total_unzipped_tokens_num,
        token_length);
  }
}

std::vector<paddle::Tensor> tokens_unzip(
    const paddle::Tensor &X,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &expert_prob_topk,
    const int &total_unzipped_tokens_num,
    const int &topk,
    const int &num_experts) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);
  int rows = X.shape()[0];  // seqlen
  int cols = X.shape()[1];  //一般为7168
  int original_token_num = rows;

  //------------------------ 输出四张量 ------------------------
  auto X_unzipped =
      paddle::empty({total_unzipped_tokens_num, cols}, X.dtype(), X.place());
  // seqlen x num_experts, 每个token的每个专家(如果被发到)对应的行索引, 未初始化
  auto zipped_expertwise_rowmap = paddle::empty(
      {original_token_num, num_experts}, paddle::DataType::INT32, X.place());
  auto token_prob_unzipped = paddle::empty(
      {total_unzipped_tokens_num}, paddle::DataType::BFLOAT16, X.place());
  auto expert_idx_unzipped = paddle::empty(
      {total_unzipped_tokens_num}, paddle::DataType::INT32, X.place());

  //------------------------ 辅助二张量 ------------------------
  //用于原子记录当前以增广的行数，其上限应为 total_unzipped_tokens_num - rows
  auto atomic_extended_offset_counter =
      paddle::zeros({1}, paddle::DataType::INT32, X.place());
  // 增广行数的合法性向量，用于线程组1唤起
  auto row_valid = paddle::zeros({total_unzipped_tokens_num - rows + 1},
                                 paddle::DataType::INT32,
                                 X.place());

  dispatch_tokens_unzip(X,
                        expert_routemap_topk,
                        expert_prob_topk,
                        X_unzipped,
                        zipped_expertwise_rowmap,
                        token_prob_unzipped,
                        expert_idx_unzipped,
                        atomic_extended_offset_counter,
                        row_valid,
                        rows,
                        total_unzipped_tokens_num,
                        cols,
                        topk,
                        num_experts);
  return {X_unzipped,
          zipped_expertwise_rowmap,
          token_prob_unzipped,
          expert_idx_unzipped};
}

// ---------------------------------- zip -------------------------------

template <int num_experts>
__global__ void tokens_weighted_zip_kernel(
    const phi::bfloat16 *__restrict__ unzipped_tokens_in,
    const phi::bfloat16 *__restrict__ unzipped_token_probs_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    phi::bfloat16 *__restrict__ weighted_zipped_tokens_out,
    const int total_zipped_tokens_num,
    const int token_length) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;

  const __nv_bfloat16 *unzipped_tokens =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_tokens_in);
  const __nv_bfloat16 *probs_unzipped =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_token_probs_in);
  __nv_bfloat16 *weighted_zipped_tokens =
      reinterpret_cast<__nv_bfloat16 *>(weighted_zipped_tokens_out);

  int local_row_fetchlist[num_experts];
  __nv_bfloat16 local_expert_problist[num_experts];
// 填充该行token被广播到的rows和对应的概率
#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    local_row_fetchlist[expert] =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
    if (local_row_fetchlist[expert] >= 0)
      local_expert_problist[expert] =
          probs_unzipped[local_row_fetchlist[expert]];
  }

  for (int i = threadIdx.x; i < token_length; i += blockDim.x) {
// tensor内部元素加权和
#pragma unroll
    for (int expert = 0; expert < num_experts; ++expert) {
      const bool is_expert_taken = (local_row_fetchlist[expert] >= 0);
      const int fetch_row = local_row_fetchlist[expert];
      if (is_expert_taken) {
      }
      weighted_zipped_tokens[this_row * token_length + i] +=
          is_expert_taken ? local_expert_problist[expert] *
                                unzipped_tokens[fetch_row * token_length + i]
                          : (__nv_bfloat16)0;
    }
  }
}

void dispatch_tokens_weighted_zip(
    const paddle::Tensor &unzipped_tokens,
    const paddle::Tensor &unzipped_token_probs,
    const paddle::Tensor &zipped_expertwise_rowmap,
    paddle::Tensor &weighted_zipped_tokens,
    const int total_zipped_tokens_num,
    const int num_experts,
    const int token_length) {
  dim3 grid, block;
  grid.x = total_zipped_tokens_num;
  block.x = 256;
  if (num_experts == 4) {
    tokens_weighted_zip_kernel<4><<<grid, block, 0, unzipped_tokens.stream()>>>(
        unzipped_tokens.data<phi::bfloat16>(),
        unzipped_token_probs.data<phi::bfloat16>(),
        zipped_expertwise_rowmap.data<int>(),
        weighted_zipped_tokens.data<phi::bfloat16>(),
        total_zipped_tokens_num,
        token_length);
  }
  cudaDeviceSynchronize();
}

std::vector<paddle::Tensor> tokens_weighted_zip(
    const paddle::Tensor &unzipped_tokens,
    const paddle::Tensor &unzipped_token_probs,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const int &total_zipped_tokens_num,
    const int &num_experts) {
  PD_CHECK(unzipped_tokens.dtype() == paddle::DataType::BFLOAT16);
  int rows = unzipped_tokens.shape()[0];  // seqlen
  int cols = unzipped_tokens.shape()[1];  //一般为7168

  //------------------------ 输出1张量 ------------------------
  auto weighted_zipped_tokens = paddle::empty({total_zipped_tokens_num, cols},
                                              unzipped_tokens.dtype(),
                                              unzipped_tokens.place());

  dispatch_tokens_weighted_zip(unzipped_tokens,
                               unzipped_token_probs,
                               zipped_expertwise_rowmap,
                               weighted_zipped_tokens,
                               total_zipped_tokens_num,
                               num_experts,
                               cols);
  return {weighted_zipped_tokens};
}


PD_BUILD_OP(tokens_unzip)
    .Inputs({"X", "expert_routemap_topk", "expert_prob_topk"})
    .Outputs({"X_unzipped",
              "zipped_expertwise_rowmap",
              "token_prob_unzipped",
              "expert_idx_unzipped"})
    .Attrs({"total_unzipped_tokens_num: int", "topk: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_unzip));

PD_BUILD_OP(tokens_weighted_zip)
    .Inputs({"unzipped_tokens",
             "unzipped_token_probs",
             "zipped_expertwise_rowmap"})
    .Outputs({"weighted_zipped_tokens"})
    .Attrs({"total_zipped_tokens: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_weighted_zip));