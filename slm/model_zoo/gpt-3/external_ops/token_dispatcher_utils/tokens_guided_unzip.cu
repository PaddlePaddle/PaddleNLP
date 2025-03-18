#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

template <int num_experts>
__global__ void tokens_guided_unzip_kernel(
    const phi::bfloat16 *__restrict__ X_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    phi::bfloat16 *__restrict__ guided_unzipped_X_out,
    const int total_zipped_tokens_num,
    const int token_length) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;
  const __nv_bfloat16 *X = reinterpret_cast<const __nv_bfloat16 *>(X_in);
  __nv_bfloat16 *guided_unzipped_X =
      reinterpret_cast<__nv_bfloat16 *>(guided_unzipped_X_out);

  int local_row_pushlist[num_experts];
// 填充该行token被广播到的rows和对应的概率
#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    local_row_pushlist[expert] =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
  }
  for (int expert = 0; expert < num_experts; ++expert) {
    int push_row = local_row_pushlist[expert];
    // 该专家没被发到，跳过
    if (push_row == -1) continue;
    // 可通过向量化优化
    for (int i = threadIdx.x; i < token_length; i += blockDim.x) {
      guided_unzipped_X[push_row * token_length + i] =
          X[this_row * token_length + i];
    }
  }
}

void dispatch_tokens_guided_unzip(
    const paddle::Tensor &X,
    const paddle::Tensor &zipped_expertwise_rowmap,
    paddle::Tensor &guided_unzipped_X,
    const int num_experts,
    const int total_zipped_tokens_num,
    const int token_length) {
  dim3 grid, block;
  grid.x = total_zipped_tokens_num;
  block.x = 256;
  if (num_experts == 4) {
    tokens_guided_unzip_kernel<4><<<grid, block, 0, X.stream()>>>(
        X.data<phi::bfloat16>(),
        zipped_expertwise_rowmap.data<int>(),
        guided_unzipped_X.data<phi::bfloat16>(),
        total_zipped_tokens_num,
        token_length);
  }
}

std::vector<paddle::Tensor> tokens_guided_unzip(
    const paddle::Tensor &X,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const int &total_unzipped_tokens_num,
    const int &num_experts) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);
  int rows = X.shape()[0];  // seqlen
  int cols = X.shape()[1];  //一般为7168

  //------------------------ 输出1张量 ------------------------
  auto guided_unzipped_X =
      paddle::empty({total_unzipped_tokens_num, cols}, X.dtype(), X.place());

  dispatch_tokens_guided_unzip(
      X, zipped_expertwise_rowmap, guided_unzipped_X, num_experts, rows, cols);
  return {guided_unzipped_X};
}

PD_BUILD_OP(tokens_guided_unzip)
    .Inputs({"X", "zipped_expertwise_rowmap"})
    .Outputs({"guided_unzipped_X"})
    .Attrs({"total_unzipped_token_num: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_guided_unzip));