#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

template <int topk>
__global__ void token_unzip_kernel(
    const phi::bfloat16 *__restrict__ X,
    const int *__restrict__ routemap_topk,
    const phi::bfloat16*__restrict__ probs_topk_in,
    phi::bfloat16 *__restrict__ X_unzipped,
    int *__restrict__ rowmap_unzipped,
    phi::bfloat16*__restrict__ probs_unzipped_out,
    int* __restrict__ expert_idx,
    int *__restrict__ atomic_extended_offset_counter,
    int *__restrict__ row_valid,
    const int total_zipped_tokens_num,
    const int total_unzipped_tokens_num,
    const int token_length,
    const int num_experts) {
  const __nv_bfloat16* probs_topk = reinterpret_cast<const __nv_bfloat16*>(probs_topk_in);
  __nv_bfloat16* probs_unzipped= reinterpret_cast<__nv_bfloat16*>(probs_unzipped_out);
  // 每个线程处理一行数据
  const int row_idx = blockIdx.x;
  // 仅在线程组2中被更新，不初始化
  int extended_row_offset;

  if (row_idx < total_unzipped_tokens_num) [[likely]] {
    // 线程组0， 主要处理topk和增广部分的行索引,以及一对一搬移
    if (row_idx < total_zipped_tokens_num) [[likely]] {
      if (threadIdx.x == 0) [[unlikely]] {
        // 寄存器加载、存储，消耗2xtopk 个reg
        // 每行只有一次非广播的机会
        bool isFirst = true;
        for (int i = 0; i < topk; i++) {
          int local_routemap_topk = routemap_topk[row_idx * topk + i];
          __nv_bfloat16 local_probs_topk = probs_topk[row_idx * topk + i];
          if (local_routemap_topk < num_experts && 
              local_routemap_topk >= 0) [[unlikely]] {
            if (isFirst) [[likely]] {
              isFirst = false;
              rowmap_unzipped[row_idx] = row_idx;
              probs_unzipped[row_idx] = local_probs_topk;
              expert_idx[row_idx] = local_routemap_topk;
            } else {
              // 增广部分， 原子更新行偏置
              extended_row_offset =
                  atomicAdd(&atomic_extended_offset_counter[0], 1);
              int extended_row_idx =
                  total_zipped_tokens_num + extended_row_offset;
              // 立即唤起相关的线程组1，减少忙等, 也强保证rowmap_unzipped的变动对组1可见
              atomicExch(&rowmap_unzipped[extended_row_idx], row_idx);
              atomicExch(&row_valid[extended_row_offset], 1);
              probs_unzipped[extended_row_idx] = local_probs_topk;
              expert_idx[extended_row_idx] =local_routemap_topk;
            }
          }
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
        int origin_row = rowmap_unzipped[row_idx];
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
                           paddle::Tensor &expert_rowmap_unzipped,
                           paddle::Tensor &token_prob_unzipped,
                           paddle::Tensor &expert_idx,
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
  if (topk == 8) {
    token_unzip_kernel<8><<<grid, block, 0, X.stream()>>>(
        X.data<phi::bfloat16>(),
        expert_routemap_topk.data<int>(),
        expert_prob_topk.data<phi::bfloat16>(),
        X_unzipped.data<phi::bfloat16>(),
        expert_rowmap_unzipped.data<int>(),
        token_prob_unzipped.data<phi::bfloat16>(),
        expert_idx.data<int>(),
        atomic_extended_offset_counter.data<int>(),
        row_valid.data<int>(),
        total_zipped_tokens_num,
        total_unzipped_tokens_num,
        token_length,
        num_experts);
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

  //------------------------ 输出四张量 ------------------------
  auto X_unzipped =
      paddle::empty({total_unzipped_tokens_num, cols}, X.dtype(), X.place());
  auto token_rowmap_unzipped = paddle::empty(
      {total_unzipped_tokens_num}, paddle::DataType::INT32, X.place());
  auto token_prob_unzipped = paddle::empty(
      {total_unzipped_tokens_num}, paddle::DataType::BFLOAT16, X.place());
  auto expert_idx = paddle::empty({total_unzipped_tokens_num}, paddle::DataType::INT32, X.place());

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
                        token_rowmap_unzipped,
                        token_prob_unzipped,
                        expert_idx,
                        atomic_extended_offset_counter,
                        row_valid,
                        rows,
                        total_unzipped_tokens_num,
                        cols,
                        topk,
                        num_experts);
  return {X_unzipped, token_rowmap_unzipped, token_prob_unzipped, expert_idx};
}

PD_BUILD_OP(tokens_unzip)
    .Inputs({"X", "expert_routemap_topk", "expert_prob_topk"})
    .Outputs({"X_unzipped", "token_rowmap_unzipped", "token_prob_unzipped", "expert_idx"})
    .Attrs({"total_unzipped_tokens_num: int", "topk: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_unzip));
