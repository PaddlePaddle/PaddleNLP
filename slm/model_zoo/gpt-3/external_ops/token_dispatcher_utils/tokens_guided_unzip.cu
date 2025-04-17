#include "utils.h"

// --------------------------- kernels ------------------------
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


template <typename T, int num_experts>
__global__ void tokens_guided_unzip_kernel(
    const T *__restrict__ X_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    T *__restrict__ guided_unzipped_X_out,
    const int total_zipped_tokens_num,
    const int token_length) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;
  /*
  const __nv_bfloat16 *X = reinterpret_cast<const __nv_bfloat16 *>(X_in);
  __nv_bfloat16 *guided_unzipped_X =
      reinterpret_cast<__nv_bfloat16 *>(guided_unzipped_X_out);
  */

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
    /*
    for (int i = threadIdx.x; i < token_length; i += blockDim.x) {
      guided_unzipped_X_out[push_row * token_length + i] =
          X_in[this_row * token_length + i];
    }
    */
    vectorized_memcpy(&X_in[this_row * token_length],
                      &guided_unzipped_X_out[push_row * token_length],
                      token_length);
  }
}
// ------------------------------ Dispatch -----------------------------
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
    if (X.dtype() == paddle::DataType::BFLOAT16) {
      tokens_guided_unzip_kernel<phi::bfloat16, 4>
          <<<grid, block, 0, X.stream()>>>(
              X.data<phi::bfloat16>(),
              zipped_expertwise_rowmap.data<int>(),
              guided_unzipped_X.data<phi::bfloat16>(),
              total_zipped_tokens_num,
              token_length);
    } else if (X.dtype() == paddle::DataType::FLOAT32) {
      tokens_guided_unzip_kernel<float, 4>
          <<<grid, block, 0, X.stream()>>>(X.data<float>(),
                                           zipped_expertwise_rowmap.data<int>(),
                                           guided_unzipped_X.data<float>(),
                                           total_zipped_tokens_num,
                                           token_length);
    }
  }
}

// -----------------------------------  API ----------------------------------
std::vector<paddle::Tensor> probs_topk_guided_unzip(
    const paddle::Tensor &probs_topk,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const int &total_unzipped_tokens_num,
    const int &num_experts,
    const int &topk) {
  PD_CHECK(probs_topk.dtype() == paddle::DataType::BFLOAT16);
  int rows = probs_topk.shape()[0];  // seqlen
  int cols = probs_topk.shape()[1];  // 一般为8
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
std::vector<paddle::Tensor> tokens_guided_unzip(
    const paddle::Tensor &X,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const int &total_unzipped_tokens_num,
    const int &num_experts) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16 ||
           X.dtype() == paddle::DataType::FLOAT32);
  int rows = X.shape()[0];  // seqlen
  int cols = X.shape()[1];  // 一般为7168

  //------------------------ 输出1张量 ------------------------
  auto guided_unzipped_X =
      paddle::empty({total_unzipped_tokens_num, cols}, X.dtype(), X.place());

  dispatch_tokens_guided_unzip(
      X, zipped_expertwise_rowmap, guided_unzipped_X, num_experts, rows, cols);
  return {guided_unzipped_X};
}

// -----------------------------------  注册 ----------------------------------
PD_BUILD_OP(tokens_guided_unzip)
    .Inputs({"X", "zipped_expertwise_rowmap"})
    .Outputs({"guided_unzipped_X"})
    .Attrs({"total_unzipped_token_num: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_guided_unzip));

PD_BUILD_OP(probs_topk_guided_unzip)
    .Inputs({"probs_topk", "expert_routemap_topk", "zipped_expertwise_rowmap"})
    .Outputs({"guided_unzipped_probs_1d"})
    .Attrs({"total_unzipped_token_num: int", "num_experts: int", "topk: int"})
    .SetKernelFn(PD_KERNEL(probs_topk_guided_unzip));
