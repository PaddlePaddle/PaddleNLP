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
__global__ void probs_topk_guided_unzip_kernel(
    const phi::bfloat16 *__restrict__ probs_topk_in,
    const int *__restrict__ expert_routemap_topk,
    const int *__restrict__ zipped_expertwise_rowmap,
    phi::bfloat16 *__restrict__ guided_unzipped_probs_1d_out,
    const int total_zipped_tokens_num) {
  const int this_row =
      blockIdx.x * blockDim.x + threadIdx.x;  // 一个线程处理一行topk
  if (this_row >= total_zipped_tokens_num) return;
  const __nv_bfloat16 *probs_topk =
      reinterpret_cast<const __nv_bfloat16 *>(probs_topk_in);

  __nv_bfloat16 token_prob_topk[topk];
  int token_expert_rowmap[num_experts];
  int token_expert_route[topk];
  __nv_bfloat16 *guided_unzipped_probs_1d =
      reinterpret_cast<__nv_bfloat16 *>(guided_unzipped_probs_1d_out);
// 使用该行的num_expert规模的行映射信息填充寄存器组，非法值为-1
#pragma unroll
  for (int i = 0; i < num_experts; i++) {
    token_expert_rowmap[i] =
        zipped_expertwise_rowmap[this_row * num_experts + i];
  }

// 使用该行的topk规模的prob和route信息填充寄存器组
#pragma unroll
  for (int i = 0; i < topk; ++i) {
    token_prob_topk[i] = probs_topk[this_row * topk + i];
    token_expert_route[i] = expert_routemap_topk[this_row * topk + i];
  }

#pragma unroll
  for (int i = 0; i < topk; ++i) {
    // 如果route不为-1，则probs亦合法，将该token的prob放入对应的unzipped_probs_1d中
    if (token_expert_route[i] != -1) {
      const int routed_expert = token_expert_route[i];
      const int mapped_row = token_expert_rowmap[routed_expert];
      guided_unzipped_probs_1d[mapped_row] = token_prob_topk[i];
    }
  }
}

void dispatch_probs_topk_guided_unzip(
    const paddle::Tensor &probs_topk,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &zipped_expertwise_rowmap,
    paddle::Tensor &guided_unzipped_probs_1d,
    const int num_experts,
    const int total_zipped_tokens_num,
    const int topk) {
  dim3 grid, block;
  block.x = 256;
  grid.x = (total_zipped_tokens_num + block.x - 1) /
           block.x;  // 每一个thread处理一个probs_topk
  if (num_experts == 4 && topk == 8) {
    probs_topk_guided_unzip_kernel<8, 4>
        <<<grid, block, 0, probs_topk.stream()>>>(
            probs_topk.data<phi::bfloat16>(),
            expert_routemap_topk.data<int>(),
            zipped_expertwise_rowmap.data<int>(),
            guided_unzipped_probs_1d.data<phi::bfloat16>(),
            total_zipped_tokens_num);
  }
}

std::vector<paddle::Tensor> probs_topk_guided_unzip(
    const paddle::Tensor &probs_topk,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const int &total_unzipped_tokens_num,
    const int &num_experts,
    const int &topk) {
  PD_CHECK(probs_topk.dtype() == paddle::DataType::BFLOAT16);
  int rows = probs_topk.shape()[0];  // seqlen
  int cols = probs_topk.shape()[1];  //一般为8
  PD_CHECK(topk == cols);

  //------------------------ 输出1张量 ------------------------
  auto guided_unzipped_probs_1d = paddle::empty(
      {total_unzipped_tokens_num}, probs_topk.dtype(), probs_topk.place());

  dispatch_probs_topk_guided_unzip(probs_topk,
                                   expert_routemap_topk,
                                   zipped_expertwise_rowmap,
                                   guided_unzipped_probs_1d,
                                   num_experts,
                                   rows,
                                   cols);
  return {guided_unzipped_probs_1d};
}

PD_BUILD_OP(probs_topk_guided_unzip)
    .Inputs({"probs_topk", "expert_routemap_topk", "zipped_expertwise_rowmap"})
    .Outputs({"guided_unzipped_probs_1d"})
    .Attrs({"total_unzipped_token_num: int", "num_experts: int", "topk: int"})
    .SetKernelFn(PD_KERNEL(probs_topk_guided_unzip));