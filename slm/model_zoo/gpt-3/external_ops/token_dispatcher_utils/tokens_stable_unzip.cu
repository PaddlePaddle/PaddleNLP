#include "utils.h"

#define CUMSUM_BLOCK_SIZE 64
#define CUMSUM_INVALID_TAG -1  // 用于标记无效的cumsum值,可为啥不能是-114514？

// 多阶段算法，控制每block处理的行数来权衡额外开销
//  首先解析routemap来更新专家当前所收到的token数，然后check前一个block给的前缀和并更新给下一个block
//  随后，目的行号的信息已获取，立即开始搬运工作，直至任务完全完成
template <typename X_T,
          typename routemap_T,
          typename probs_T,
          int topk,
          int num_experts>
__global__ void tokens_unzip_stable_kernel(
    const X_T *__restrict__ X,
    const routemap_T *__restrict__ routemap_topk,
    const probs_T *__restrict__ probs_topk,
    X_T *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    probs_T *__restrict__ probs_unzipped,
    int *__restrict__ expert_idx_unzipped,
    int *global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int max_tokens_per_expert,
    const int token_length) {

  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  int cumsum_offset[num_experts];
  int expert_offset[num_experts];
  int local_cumsum[num_experts];
#pragma unroll
  for (int i = 0; i < num_experts; i++) {
    cumsum_offset[i] =
        (blockIdx.x == 0)
            ? 0
            : CUMSUM_INVALID_TAG;  // 除了第0个block，其他的都以非法值初始化,因为atomic忙等要用
    expert_offset[i] = i * max_tokens_per_expert;
    local_cumsum[i] = 0;
  }
  const int base_row_idx = blockIdx.x * CUMSUM_BLOCK_SIZE;
  __shared__ int shared_expert_rowmap[CUMSUM_BLOCK_SIZE][num_experts];
  __shared__ probs_T shared_expert_probmap[CUMSUM_BLOCK_SIZE][num_experts];

  // --------------------- thread0 单线程任务传递 -------------------------
  if (threadIdx.x == 0) [[unlikely]] {
    int local_expert_rowmap[CUMSUM_BLOCK_SIZE][num_experts];
    probs_T local_expert_probs[CUMSUM_BLOCK_SIZE][num_experts];
#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
#pragma unroll
      for (int j = 0; j < num_experts; j++) {
        local_expert_rowmap[i][j] =
            -1;  // 以非法值初始化，方便后续shared mem写入
        local_expert_probs[i][j] = (probs_T)0;
      }
    }
    // 将乱序访存限制在寄存器级别，后续shared_mem规整写入
    for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
         row++) {
      if (row >= total_zipped_tokens_num) break;
      const int internal_row = row - block_row_base;
#pragma unroll
      for (int k = 0; k < topk; k++) {
        const int expert = routemap_topk[row * topk + k];
        if (expert == -1) continue;
        local_expert_rowmap[internal_row][expert] =
            local_cumsum[expert] + expert_offset[expert];
        local_expert_probs[internal_row][expert] = probs_topk[row * topk + k];
        local_cumsum[expert] += 1;
      }
    }
// 块间通信逻辑，更新下一个block的cumsum_offset
#pragma unroll
    for (int i = 0; i < num_experts; i++) {
      if (blockIdx.x != 0) [[likely]] {
        while (cumsum_offset[i] == CUMSUM_INVALID_TAG) [[likely]] {
          cumsum_offset[i] = atomicExch(
              &global_expertwise_block_cumsum[blockIdx.x * num_experts + i],
              CUMSUM_INVALID_TAG);
        }
      }
      const int proposed_offset = cumsum_offset[i] + local_cumsum[i];
      global_expertwise_block_cumsum[(blockIdx.x + 1) * num_experts + i] =
          proposed_offset;
    }  // 至此，给下一个block的cumsum已经更新完毕，下一个block可以开始cumsum的计算了

#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
#pragma unroll
      for (int j = 0; j < num_experts; j++) {
        const int proposed_row =
            (local_expert_rowmap[i][j] == -1)
                ? -1
                : (local_expert_rowmap[i][j] + cumsum_offset[j]);
        shared_expert_rowmap[i][j] = proposed_row;
        shared_expert_probmap[i][j] = local_expert_probs[i][j];
      }
    }
  }  // 至此，
     // local_cumsum及其对应的rowmap、probs已经更新完毕，需要依据上一个block提供的信息将cumsum_offset更新并传递
     // 至此，本线程块内的shared_mem已经规整完毕，接下来是向量化的数据搬运
  __syncthreads();  // 其余线程等到了thread0，
                    // 具体工作已经与上一个block沟通完毕、存储在 shared中
  // ------------------------- 所有block内线程 -------------------------
  for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
       row++) {
    if (row >= total_zipped_tokens_num) return;
    const int internal_row = row - block_row_base;
#pragma unroll
    for (int expert = 0; expert < num_experts; expert++) {
      const int unzipped_row_idx = shared_expert_rowmap[internal_row][expert];
      zipped_expertwise_rowmap[row * num_experts + expert] = unzipped_row_idx;
      if (unzipped_row_idx == -1) continue;
      // 更新三个核心数据结构，可以只让thread0做，但引入if没必要
      if (threadIdx.x == 0) {
        probs_unzipped[unzipped_row_idx] =
            shared_expert_probmap[internal_row][expert];
        expert_idx_unzipped[unzipped_row_idx] = expert;
      }
      vectorized_memcpy(&X[row * token_length],
                        &X_unzipped[unzipped_row_idx * token_length],
                        token_length);
    }
  }
}
// ---------------------------- Dispatch ---------------------------------
void dispatch_tokens_unzip_stable(
    const paddle::Tensor &X,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &expert_prob_topk,
    paddle::Tensor &X_unzipped,
    paddle::Tensor &zipped_expertwise_rowmap,
    paddle::Tensor &token_prob_unzipped,
    paddle::Tensor &expert_idx_unzipped,
    paddle::Tensor &global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int token_length,
    const int topk,
    const int num_experts,
    const int max_tokens_per_expert) {
  dim3 grid, block;
  grid.x =
      (total_zipped_tokens_num + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  block.x = 256;

// 定义类型获取宏
#define DTYPE_CASE(dtype, type) dtype == paddle::DataType::type
#define GET_DATA(tensor, type) tensor.data<type>()

// 分发处理不同的类型组合
#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, TOPK, NUM_EXPERTS)             \
  auto kernel =                                                              \
      tokens_unzip_stable_kernel<TOKEN_T, INT_T, PROB_T, TOPK, NUM_EXPERTS>; \
  kernel<<<grid, block, 0, X.stream()>>>(                                    \
      GET_DATA(X, TOKEN_T),                                                  \
      GET_DATA(expert_routemap_topk, INT_T),                                 \
      GET_DATA(expert_prob_topk, PROB_T),                                    \
      GET_DATA(X_unzipped, TOKEN_T),                                         \
      GET_DATA(zipped_expertwise_rowmap, INT_T),                             \
      GET_DATA(token_prob_unzipped, PROB_T),                                 \
      expert_idx_unzipped.data<int>(),                                       \
      global_expertwise_block_cumsum.data<int>(),                            \
      total_zipped_tokens_num,                                               \
      max_tokens_per_expert,                                                 \
      token_length);

// 可扩展：处理特定的topk和num_experts组合,可根据之后需求进行扩展
#define HANDLE_EXPERT_CASE(TOKEN_T, PROB_T, INT_T) \
  if (topk == 8 && num_experts == 4) {             \
    DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, 8, 4)    \
  } else {                                         \
    std::__throw_invalid_argument;                 \
  }

#define HANDLE_TOKEN_TYPE(PROB_T, INT_T)                  \
  if (DTYPE_CASE(X.dtype(), BFLOAT16)) {                  \
    HANDLE_EXPERT_CASE(phi::bfloat16, PROB_T, INT_T)      \
  } else if (DTYPE_CASE(X.dtype(), FLOAT8_E4M3FN)) {      \
    HANDLE_EXPERT_CASE(phi::float8_e4m3fn, PROB_T, INT_T) \
  }

#define HANDLE_PROB_TYPE(INT_T)                               \
  if (DTYPE_CASE(expert_prob_topk.dtype(), BFLOAT16)) {       \
    HANDLE_TOKEN_TYPE(phi::bfloat16, INT_T)                   \
  } else if (DTYPE_CASE(expert_prob_topk.dtype(), FLOAT32)) { \
    HANDLE_TOKEN_TYPE(float, INT_T)                           \
  }

  // 可扩展：根据整型类型控制派发，未来可支持int8，但int64不行，因为下标开销太重了，建议在外面直接cast到int32
  if (DTYPE_CASE(zipped_expertwise_rowmap.dtype(), INT32)) {
    HANDLE_PROB_TYPE(int)
  }

#undef DTYPE_CASE
#undef GET_DATA
#undef DISPATCH_CASE
#undef HANDLE_EXPERT_CASE
#undef HANDLE_TOKEN_TYPE
#undef HANDLE_PROB_TYPE
}


std::vector<paddle::Tensor> tokens_unzip_stable(
    const paddle::Tensor &X,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &expert_prob_topk,
    const int &topk,
    const int &num_experts,
    const int &max_tokens_per_expert) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16 ||
           X.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  PD_CHECK(expert_prob_topk.dtype() == paddle::DataType::BFLOAT16 ||
           expert_prob_topk.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(expert_routemap_topk.dtype() == paddle::DataType::INT32);
  const int rows = X.shape()[0];  // 一般为seqlen
  const int cols = X.shape()[1];  // 一般为7168
  const int output_rows = num_experts * max_tokens_per_expert;
  //------------------------ 输出四张量 ------------------------
  auto X_unzipped = paddle::empty({output_rows, cols}, X.dtype(), X.place());

  // 暂时将所有输出缓冲区初始化为0用于padding，后续可融合进kernel逻辑中?
  if (X.dtype() == paddle::DataType::BFLOAT16) {
    auto X_unzipped_ptr =
        reinterpret_cast<void *>(X_unzipped.data<phi::bfloat16>());
    cudaMemsetAsync(X_unzipped_ptr,
                    0,
                    sizeof(phi::bfloat16) * output_rows * cols,
                    X.stream());
  } else if (X.dtype() == paddle::DataType::FLOAT8_E4M3FN) {
    auto X_unzipped_ptr =
        reinterpret_cast<void *>(X_unzipped.data<phi::float8_e4m3fn>());
    cudaMemsetAsync(X_unzipped_ptr,
                    0,
                    sizeof(phi::float8_e4m3fn) * output_rows * cols,
                    X.stream());
  }
  // 核心数据结构，用于后续zip 【seqlen x num_experts】, 行对应zipped
  // token，列对应专家，元素对应unzipped行号
  auto zipped_expertwise_rowmap =
      paddle::empty({rows, num_experts}, paddle::DataType::INT32, X.place());

  // 重要数据结构，用于FP8 grouped_gemm，指示行所属专家号,行对应prob值
  // ------------- expert_idx padding 相关逻辑 -------------
  auto expert_idx_unzipped =
      paddle::empty({output_rows}, paddle::DataType::INT32, X.place());
  auto expert_idx_unzipped_ptr =
      reinterpret_cast<void *>(expert_idx_unzipped.data<int>());
  // 置非法值-1用于padding
  cudaMemsetAsync(expert_idx_unzipped_ptr,
                  -1,
                  sizeof(int) * output_rows,
                  expert_idx_unzipped.stream());
  // ------------- token_prob padding 相关逻辑 -------------
  auto token_prob_unzipped = paddle::empty(
      {output_rows}, expert_prob_topk.dtype(), expert_prob_topk.place());
  if (expert_prob_topk.dtype() == paddle::DataType::BFLOAT16) {
    auto token_prob_unzipped_ptr =
        reinterpret_cast<void *>(X_unzipped.data<phi::bfloat16>());
    cudaMemsetAsync(token_prob_unzipped_ptr,
                    0,
                    sizeof(phi::bfloat16) * output_rows,
                    token_prob_unzipped.stream());
  } else if (expert_prob_topk.dtype() == paddle::DataType::FLOAT32) {
    auto token_prob_unzipped_ptr =
        reinterpret_cast<void *>(X_unzipped.data<float>());
    cudaMemsetAsync(token_prob_unzipped_ptr,
                    0,
                    sizeof(float) * output_rows,
                    token_prob_unzipped.stream());
  }
  // ------------ 前缀和辅助数组相关逻辑，“推”式block通信 ------------
  const int cumsum_blocknum =
      (rows + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  auto global_expertwise_block_cumsum = paddle::empty(
      {cumsum_blocknum + 1, num_experts}, paddle::DataType::INT32, X.place());
  auto global_expertwise_block_cumsum_ptr =
      reinterpret_cast<void *>(global_expertwise_block_cumsum.data<int>());
  // 设置为非法值CUMSUM_INVALID_TAG，用于线程块等待时使用
  cudaMemsetAsync(global_expertwise_block_cumsum_ptr,
                  CUMSUM_INVALID_TAG,
                  sizeof(int) * (cumsum_blocknum + 1) * num_experts,
                  global_expertwise_block_cumsum.stream());

  dispatch_tokens_unzip_stable(X,
                               expert_routemap_topk,
                               expert_prob_topk,
                               X_unzipped,
                               zipped_expertwise_rowmap,
                               token_prob_unzipped,
                               expert_idx_unzipped,
                               global_expertwise_block_cumsum,
                               rows,
                               cols,
                               topk,
                               num_experts,
                               max_tokens_per_expert);
  return {X_unzipped,
          zipped_expertwise_rowmap,
          token_prob_unzipped,
          expert_idx_unzipped,
          global_expertwise_block_cumsum};
}

PD_BUILD_OP(tokens_unzip_stable)
    .Inputs({"X", "expert_routemap_topk", "expert_prob_topk"})
    .Outputs({"X_unzipped",
              "zipped_expertwise_rowmap",
              "token_prob_unzipped",
              "expert_idx_unzipped",
              "debug_global_expertwise_block_cumsum"})
    .Attrs({"topk: int", "num_experts: int", "max_tokens_per_expert: int"})
    .SetKernelFn(PD_KERNEL(tokens_unzip_stable));


#undef CUMSUM_BLOCK_SIZE