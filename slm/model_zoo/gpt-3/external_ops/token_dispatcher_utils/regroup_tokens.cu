#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

__global__ void regroup_tokens_kernel(const phi::bfloat16* __restrict__ X_in,
                                      const int* __restrict__ expert_idx,
                                      const int original_token_num,
                                      const int token_length,
                                      const int expert_token_num,
                                      int* __restrict__ atomic_offset_counters,
                                      phi::bfloat16* grouped_X_out) {
  int row_idx = blockIdx.x;
  if (row_idx >= original_token_num) return;

  const __nv_bfloat16* X = reinterpret_cast<const __nv_bfloat16*>(X_in);
  __nv_bfloat16* grouped_X = reinterpret_cast<__nv_bfloat16*>(grouped_X_out);

  extern __shared__ __nv_bfloat16* target_rowbase;
  if (threadIdx.x == 0) {
    int target_group = expert_idx[row_idx];
    int offset = target_group * expert_token_num * token_length;
    __nv_bfloat16* group_base = grouped_X + offset;
    int target_row_idx = atomicAdd(&(atomic_offset_counters[target_group]), 1);
    target_rowbase = group_base + target_row_idx * token_length;
  }
  __syncthreads();

  for (int col_offset = threadIdx.x; col_offset < token_length;
       col_offset += blockDim.x) {
    target_rowbase[col_offset] = X[row_idx * token_length + col_offset];
  }
}

void dispatch_regroup_tokens_kernel(const paddle::Tensor& X,
                                    const paddle::Tensor& expert_idx,
                                    const int expert_token_num,
                                    paddle::Tensor& atomic_offset_counters,
                                    paddle::Tensor grouped_X) {
  dim3 grid;
  dim3 block;
  int original_token_num = X.shape()[0];
  int token_length = X.shape()[1];
  grid.x = original_token_num;
  block.x = 256;  // 单block处理单token
  regroup_tokens_kernel<<<grid, block, 0, X.stream()>>>(
      X.data<phi::bfloat16>(),
      expert_idx.data<int>(),
      original_token_num,
      token_length,
      expert_token_num,
      atomic_offset_counters.data<int>(),
      grouped_X.data<phi::bfloat16>());
}
std::vector<paddle::Tensor> regroup_tokens(const paddle::Tensor& X,
                                           const paddle::Tensor& expert_idx,
                                           const int& expert_num,
                                           const int& token_max_per_expert) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);  // 当前只支持BFLOAT16
  int rows = X.shape()[0];
  int cols = X.shape()[1];
  paddle::Tensor out;

  // 待优化，padding时使用full算子默认将写不到的元素归为0
  out = paddle::zeros(
      {expert_num * token_max_per_expert, cols}, X.dtype(), X.place());
  //将原子计数数组初始化为0
  auto atomic_offset_counters =
      paddle::zeros({expert_num}, paddle::DataType::INT32, X.place());
  dispatch_regroup_tokens_kernel(
      X, expert_idx, token_max_per_expert, atomic_offset_counters, out);
  return {out};
}


PD_BUILD_OP(regroup_tokens)
    .Inputs({"X", "expert_idx"})
    .Outputs({"grouped_X"})
    .Attrs({"expert_num: int", "token_max_per_expert: int"})
    .SetKernelFn(PD_KERNEL(regroup_tokens));
