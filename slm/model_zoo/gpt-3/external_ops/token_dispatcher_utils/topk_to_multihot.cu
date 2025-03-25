#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

template <bool containsProb, int topk, int expert>
__global__ void convert_to_multihot_specialized_kernel(
    const int *__restrict__ routemap_topk,  // 输入矩阵 [seqlen, topk]
    const float *__restrict__ probs_topk,   // 输入矩阵 [seqlen, topk]
    int *routemap_multihot,                 // 输出矩阵 [seqlen, expert]
    float *probs_multihot,                  // 输出矩阵 [seqlen, expert]
    int seqlen                              // 序列长度
) {
  // 每个线程处理一行数据
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_idx < seqlen) [[likely]] {
    // 寄存器加载、存储，消耗2x(topk + expert)个reg
    int local_routemap_topk[topk];
    float local_probs_topk[topk];
    int local_routemap_multihot[expert];
    float local_probs_multihot[expert];
    // 合并初始化 thread local
#pragma unroll
    for (int i = 0; i < expert; i++) {
      local_routemap_multihot[i] = 0;
      if constexpr (containsProb) {
        local_probs_multihot[i] = 0.0f;
      }
    }
    // local_topk 数据加载
#pragma unroll
    for (int i = 0; i < topk; i++) {
      local_routemap_topk[i] = routemap_topk[row_idx * topk + i];
      if constexpr (containsProb) {
        local_probs_topk[i] = probs_topk[row_idx * topk + i];
      }
    }
    // local_multihot 数据填充
#pragma unroll
    for (int i = 0; i < topk; i++) {
      int expert_idx = local_routemap_topk[i];
      if (expert_idx >= 0 && expert_idx < expert) {
        local_routemap_multihot[expert_idx] = 1;
        if constexpr (containsProb) {
          local_probs_multihot[expert_idx] = local_probs_topk[i];
        }
      }
    }

    // 连续访存结果写入，拆两个for以获取更好的写入带宽
#pragma unroll
    for (int i = 0; i < expert; i++) {
      routemap_multihot[row_idx * expert + i] = local_routemap_multihot[i];
    }

#pragma unroll
    for (int i = 0; i < expert; i++) {
      probs_multihot[row_idx * expert + i] = local_probs_multihot[i];
    }
  }
}

// 通用版本的 kernel dispatch逻辑， 覆盖两种case，可通过牺牲编译时间来扩展
template <bool containsProb>
void dispatch_fused_topk_to_multihot(
    const paddle::Tensor &routemap_topk,
    const paddle::optional<paddle::Tensor> &probs_topk,
    paddle::Tensor &routemap_multihot,
    paddle::Tensor &probs_multihot,
    int seqlen,
    int topk,
    int num_experts) {
  dim3 block, grid;
  block.x = 256;
  grid.x = (seqlen + block.x - 1) / block.x;

#define HANDLE_CASE(tk, exp)                                 \
  if (topk == tk && num_experts == exp) {                    \
    if constexpr (containsProb) {                            \
      convert_to_multihot_specialized_kernel<true, tk, exp>  \
          <<<grid, block, 0, routemap_topk.stream()>>>(      \
              routemap_topk.data<int>(),                     \
              probs_topk.get().data<float>(),                \
              routemap_multihot.data<int>(),                 \
              probs_multihot.data<float>(),                  \
              seqlen);                                       \
    } else {                                                 \
      convert_to_multihot_specialized_kernel<false, tk, exp> \
          <<<grid, block, 0, routemap_topk.stream()>>>(      \
              routemap_topk.data<int>(),                     \
              nullptr,                                       \
              routemap_multihot.data<int>(),                 \
              nullptr,                                       \
              seqlen);                                       \
    }                                                        \
    return;                                                  \
  }

  // 处理常见的topk, expert组合
  HANDLE_CASE(8, 8);
  HANDLE_CASE(8, 4);
#undef HANDLE_CASE
}

std::vector<paddle::Tensor> fused_topk_to_multihot(
    const paddle::Tensor &expert_routemap_topk,
    const paddle::optional<paddle::Tensor> &expert_probability_topk,
    const int &seqlen,
    const int &topk,
    const int &num_experts) {
  PD_CHECK(expert_routemap_topk.dtype() == paddle::DataType::INT32);
  if (expert_probability_topk)
    PD_CHECK(expert_probability_topk.get().dtype() ==
             paddle::DataType::FLOAT32);

  paddle::Tensor expert_routemap_multihot =
      paddle::empty({seqlen, num_experts},
                    expert_routemap_topk.dtype(),
                    expert_routemap_topk.place());
  paddle::Tensor expert_probability_multihot;

  if (expert_probability_topk) {  // 如果包含prob_topk，则为prob_multihot留出空间
    expert_probability_multihot =
        paddle::empty({seqlen, num_experts},
                      expert_probability_topk->dtype(),
                      expert_probability_topk->place());
    dispatch_fused_topk_to_multihot<true>(expert_routemap_topk,
                                          expert_probability_topk,
                                          expert_routemap_multihot,
                                          expert_probability_multihot,
                                          seqlen,
                                          topk,
                                          num_experts);
  } else {
    dispatch_fused_topk_to_multihot<false>(expert_routemap_topk,
                                           expert_probability_topk,
                                           expert_routemap_multihot,
                                           expert_probability_multihot,
                                           seqlen,
                                           topk,
                                           num_experts);
  }
  return {expert_routemap_multihot, expert_probability_multihot};
}

PD_BUILD_OP(fused_topk_to_multihot)
    .Inputs({"expert_routemap_topk",
             paddle::Optional("expert_probability_topk")})
    .Outputs({"expert_routemap_multihot",
              paddle::Optional("expert_probability_multihot")})
    .Attrs({"seqlen: int", "topk: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(fused_topk_to_multihot));
