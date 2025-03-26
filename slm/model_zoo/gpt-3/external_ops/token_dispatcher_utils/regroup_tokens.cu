#include "utils.h"

__global__ void regroup_tokens_kernel(const phi::bfloat16* __restrict__ X_in,
                                      const phi::bfloat16* __restrict__ dout_in,
                                      const int* __restrict__ expert_idx,
                                      const int original_token_num,
                                      const int h1,
                                      const int h2,
                                      const int expert_token_num,
                                      int* __restrict__ atomic_offset_counters,
                                      phi::bfloat16* grouped_X_out,
                                      phi::bfloat16* grouped_dout_out) {
  int row_idx = blockIdx.x;
  if (row_idx >= original_token_num) return;

  const __nv_bfloat16* X = reinterpret_cast<const __nv_bfloat16*>(X_in);
  const __nv_bfloat16* dout = reinterpret_cast<const __nv_bfloat16*>(dout_in);
  __nv_bfloat16* grouped_X = reinterpret_cast<__nv_bfloat16*>(grouped_X_out);
  __nv_bfloat16* grouped_dout =
      reinterpret_cast<__nv_bfloat16*>(grouped_dout_out);

  extern __shared__ __nv_bfloat16* target_rowbase_X;
  extern __shared__ __nv_bfloat16* target_rowbase_dout;
  if (threadIdx.x == 0) {
    int target_group = expert_idx[row_idx];
    int offset_X = target_group * expert_token_num * h1;  // stride = token_len
    int offset_dout =
        target_group * expert_token_num * h2;  // stride = feature_len
    __nv_bfloat16* group_base_X = grouped_X + offset_X;
    __nv_bfloat16* group_base_dout = grouped_dout + offset_dout;
    int target_row_idx = atomicAdd(&(atomic_offset_counters[target_group]), 1);
    target_rowbase_X = group_base_X + target_row_idx * h1;
    target_rowbase_dout = group_base_dout + target_row_idx * h2;
  }
  __syncthreads();

  // X相关的数据搬移
  /*
  for (int col_offset = threadIdx.x; col_offset < h1;
       col_offset += blockDim.x) {
    target_rowbase_X[col_offset] = X[row_idx * h1 + col_offset];
  }
  */
  vectorized_memcpy(&X[row_idx * h1], target_rowbase_X, h1);
  // dout相关的数据搬移
  /*
  for (int col_offset = threadIdx.x; col_offset < h2;
       col_offset += blockDim.x) {
    target_rowbase_dout[col_offset] = dout[row_idx * h2 + col_offset];
  }
  */
  vectorized_memcpy(&dout[row_idx * h2], target_rowbase_dout, h2);
}

void dispatch_regroup_tokens_kernel(const paddle::Tensor& X,
                                    const paddle::Tensor& dout,
                                    const paddle::Tensor& expert_idx,
                                    const int expert_token_num,
                                    paddle::Tensor& atomic_offset_counters,
                                    paddle::Tensor grouped_X,
                                    paddle::Tensor grouped_dout) {
  dim3 grid;
  dim3 block;
  int original_token_num = X.shape()[0];
  int h1 = X.shape()[1];
  int h2 = dout.shape()[1];
  grid.x = original_token_num;
  block.x = 256;  // 单block处理单token+dout
  regroup_tokens_kernel<<<grid, block, 0, X.stream()>>>(
      X.data<phi::bfloat16>(),
      dout.data<phi::bfloat16>(),
      expert_idx.data<int>(),
      original_token_num,
      h1,
      h2,
      expert_token_num,
      atomic_offset_counters.data<int>(),
      grouped_X.data<phi::bfloat16>(),
      grouped_dout.data<phi::bfloat16>());
}
std::vector<paddle::Tensor> regroup_tokens(const paddle::Tensor& X,
                                           const paddle::Tensor& dout,
                                           const paddle::Tensor& expert_idx,
                                           const int& expert_num,
                                           const int& token_max_per_expert) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16);  // 当前只支持BFLOAT16
  int seqlen = X.shape()[0];
  int h1 = X.shape()[1];
  int h2 = dout.shape()[1];
  paddle::Tensor grouped_X, grouped_dout;

  // 待优化，padding时使用full算子默认将写不到的元素归为0
  grouped_X = paddle::zeros(
      {expert_num * token_max_per_expert, h1}, X.dtype(), X.place());
  grouped_dout = paddle::zeros(
      {expert_num * token_max_per_expert, h2}, X.dtype(), X.place());
  // 将原子计数数组初始化为0
  auto atomic_offset_counters =
      paddle::zeros({expert_num}, paddle::DataType::INT32, X.place());
  dispatch_regroup_tokens_kernel(X,
                                 dout,
                                 expert_idx,
                                 token_max_per_expert,
                                 atomic_offset_counters,
                                 grouped_X,
                                 grouped_dout);
  return {grouped_X, grouped_dout};
}

PD_BUILD_OP(regroup_tokens)
    .Inputs({"X", "dout", "expert_idx"})
    .Outputs({"grouped_X", "grouped_dout"})
    .Attrs({"expert_num: int", "token_max_per_expert: int"})
    .SetKernelFn(PD_KERNEL(regroup_tokens));
