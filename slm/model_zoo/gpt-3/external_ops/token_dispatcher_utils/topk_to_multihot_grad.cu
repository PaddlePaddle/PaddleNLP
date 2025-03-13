#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

template <int topk, int expert>
__global__ void convert_to_topk_specialized_kernel(
    const int *__restrict__ routemap_topk,     // 输入矩阵 [seqlen, topk]
    const float *__restrict__ probs_multihot,  // 输入矩阵 [seqlen, expert]
    float *probs_topk,                         // 输出矩阵 [seqlen, topk]
    int seqlen                                 // 序列长度
) {
  // 每个线程处理一行数据
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_idx < seqlen) [[likely]] {
    // 寄存器加载、存储，消耗2xtopk + expert 个reg
    int local_routemap_topk[topk];
    float local_probs_multihot[expert];
    float local_probs_topk[topk];
    // 合并初始化 thread local
#pragma unroll
    for (int i = 0; i < topk; i++) {
      local_probs_topk[i] = 0.0f;
      local_routemap_topk[i] = routemap_topk[row_idx * topk + i];
    }
    // local_multihot 数据填充
#pragma unroll
    for (int i = 0; i < topk; i++) {
      const int expert_idx = local_routemap_topk[i];
      if (expert_idx > -1 && expert_idx < expert) {
        local_probs_topk[i] = local_probs_multihot[expert_idx];
      }
    }

    // 写回
#pragma unroll
    for (int i = 0; i < topk; i++) {
      probs_topk[row_idx * topk + i] = local_probs_topk[i];
    }
  }
}

// 通用版本的 kernel dispatch逻辑， 覆盖两种case，可通过牺牲编译时间来扩展
void dispatch_multihot_prob_backto_topk(const paddle::Tensor &routemap_topk,
                                        const paddle::Tensor &probs_multihot,
                                        paddle::Tensor &probs_topk,
                                        int seqlen,
                                        int topk,
                                        int num_experts) {
  dim3 block, grid;
  block.x = 256;
  grid.x = (seqlen + block.x - 1) / block.x;

#define HANDLE_CASE(tk, exp)                          \
  if (topk == tk && num_experts == exp) {             \
    convert_to_topk_specialized_kernel<tk, exp>       \
        <<<grid, block, 0, routemap_topk.stream()>>>( \
            routemap_topk.data<int>(),                \
            probs_multihot.data<float>(),             \
            probs_topk.data<float>(),                 \
            seqlen);                                  \
    return;                                           \
  }

  // 处理常见的topk, expert组合
  HANDLE_CASE(8, 8);
  HANDLE_CASE(8, 4);
#undef HANDLE_CASE
}

std::vector<paddle::Tensor> multihot_prob_backto_topk(
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &expert_probability_multihot,
    const int &seqlen,
    const int &topk,
    const int &num_experts) {
  PD_CHECK(expert_routemap_topk.dtype() == paddle::DataType::INT32);
  PD_CHECK(expert_probability_multihot.dtype() == paddle::DataType::FLOAT32);

  paddle::Tensor expert_probability_topk =
      paddle::empty({seqlen, topk},
                    expert_probability_multihot.dtype(),
                    expert_probability_multihot.place());

  dispatch_multihot_prob_backto_topk(expert_routemap_topk,
                                     expert_probability_multihot,
                                     expert_probability_topk,
                                     seqlen,
                                     topk,
                                     num_experts);

  return {expert_probability_topk};
}

PD_BUILD_OP(fused_multihot_prob_backto_topk)
    .Inputs({"expert_routemap_topk", "expert_probability_multihot"})
    .Attrs({"seqlen: int", "topk: int", "num_experts: int"})
    .Outputs({"expert_probability_topk"})
    .SetKernelFn(PD_KERNEL(multihot_prob_backto_topk));
