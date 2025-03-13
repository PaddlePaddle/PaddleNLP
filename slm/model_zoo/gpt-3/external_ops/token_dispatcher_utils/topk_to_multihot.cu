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
    const int *__restrict__ routemap_topk, // 输入矩阵 [seqlen, topk]
    const float *__restrict__ probs_topk,  // 输入矩阵 [seqlen, topk]
    int *routemap_multihot, // 输出矩阵 [seqlen, expert], 已初始化为0
    float *probs_multihot, // 输出矩阵 [seqlen, expert], 已初始化为0.0f
    int seqlen             // 序列长度
) {
  // 每个线程处理一行数据
  int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_idx < seqlen) {
    // 寄存器加载一行数据
    int expert_indices_topk[topk];
    float expert_probs_topk[topk];

#pragma unroll
    for (int i = 0; i < topk; i++) {
      expert_indices_topk[i] = routemap_topk[row_idx * topk + i];
      if constexpr (containsProb) {
        expert_probs_topk[i] = probs_topk[row_idx * topk + i];
      }
    }

    // 使用位操作快速构建结果
    unsigned int mask = 0;
#pragma unroll
    for (int i = 0; i < topk; i++) {
      int expert_idx = expert_indices_topk[i];
      if (expert_idx >= 0 && expert_idx < expert) {
        mask |= (1u << expert_idx);
        routemap_multihot[row_idx * expert + expert_idx] = 1;
        if constexpr (containsProb) {
          probs_multihot[row_idx * expert + expert_idx] = expert_probs_topk[i];
        }
      }
    }

#pragma unroll
    for (int i = 0; i < expert; i++) {
      routemap_multihot[row_idx * expert + i] = (mask & (1u << i)) ? 1 : 0;
    }
  }
}

// 通用版本的 kernel 启动包装函数
template <bool containsProb>
void dispatch_fused_topk_to_multihot(
    const paddle::Tensor &routemap_topk,
    const paddle::optional<paddle::Tensor> &probs_topk,
    paddle::Tensor &routemap_multihot, paddle::Tensor &probs_multihot,
    int seqlen, int topk, int num_experts) {
  dim3 block, grid;
  block.x = 256;
  grid.x = (seqlen + block.x - 1) / block.x;

#define HANDLE_CASE(tk, exp)                                                   \
  if (topk == tk && num_experts == exp) {                                      \
    if constexpr (containsProb) {                                              \
      convert_to_multihot_specialized_kernel<true, tk, exp>                    \
          <<<grid, block, 0, routemap_topk.stream()>>>(                        \
              routemap_topk.data<int>(), probs_topk.get().data<float>(),       \
              routemap_multihot.data<int>(), probs_multihot.data<float>(),     \
              seqlen);                                                         \
    } else {                                                                   \
      convert_to_multihot_specialized_kernel<false, tk, exp>                   \
          <<<grid, block, 0, routemap_topk.stream()>>>(                        \
              routemap_topk.data<int>(), nullptr,                              \
              routemap_multihot.data<int>(), nullptr, seqlen);                 \
    }                                                                          \
    return;                                                                    \
  }

  // 处理常见的topk, expert组合
  HANDLE_CASE(8, 8);
  HANDLE_CASE(8, 4);
#undef HANDLE_CASE
}
std::vector<paddle::Tensor> fused_topk_to_multihot(
    const paddle::Tensor &expert_routemap_topk,
    const paddle::optional<paddle::Tensor> &expert_probability_topk,
    const int &seqlen, const int &topk, const int &num_experts) {

  PD_CHECK(expert_routemap_topk.dtype() == paddle::DataType::INT32);
  if (expert_probability_topk)
    PD_CHECK(expert_probability_topk.get().dtype() ==
             paddle::DataType::FLOAT32);

  paddle::Tensor expert_routemap_multihot =
      paddle::empty({seqlen, num_experts}, expert_routemap_topk.dtype(),
                    expert_routemap_topk.place());
  paddle::Tensor expert_probability_multihot;

  if (expert_probability_topk) { // 如果包含prob_topk，则为prob_multihot留出空间
    expert_probability_multihot =
        paddle::empty({seqlen, num_experts}, expert_probability_topk->dtype(),
                      expert_probability_topk->place());
    dispatch_fused_topk_to_multihot<true>(
        expert_routemap_topk, expert_probability_topk, expert_routemap_multihot,
        expert_probability_multihot, seqlen, topk, num_experts);
  } else {
    dispatch_fused_topk_to_multihot<false>(
        expert_routemap_topk, expert_probability_topk, expert_routemap_multihot,
        expert_probability_multihot, seqlen, topk, num_experts);
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
