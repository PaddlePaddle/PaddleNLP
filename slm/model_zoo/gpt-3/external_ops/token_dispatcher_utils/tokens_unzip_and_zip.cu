#include "utils.h"


template <typename X_T,
          typename routemap_T,
          typename probs_T,
          int topk,
          int num_experts>
__global__ void token_unzip_kernel(
    const X_T *__restrict__ X,
    const routemap_T *__restrict__ routemap_topk,
    const probs_T *__restrict__ probs_topk,
    X_T *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    probs_T *__restrict__ probs_unzipped,
    int *__restrict__ expert_idx_unzipped,
    int *__restrict__ atomic_extended_offset_counter,
    int *__restrict__ row_valid,
    const int total_zipped_tokens_num,
    const int total_unzipped_tokens_num,
    const int token_length) {
  // 每个线程处理一行数据
  const int row_idx = blockIdx.x;
  // 仅在线程组2中被更新，不初始化
  extern __shared__ int shared_original_row;

  if (row_idx < total_unzipped_tokens_num) [[likely]] {
    // 线程组0，
    // 主要处理topk和增广部分的行索引、处理专家广播后的行表、一对一搬移
    if (row_idx < total_zipped_tokens_num) [[likely]] {
      // ----------------- 增广行的任务派发逻辑，交给thread0 --------------
      if (threadIdx.x == 0) [[unlikely]] {
        // 寄存器加载、存储，消耗2xtopk 个reg
        // 每行只有一次非广播的机会
        bool isFirst = true;
        int local_expert_rowmap[num_experts];
// 寄存器填入非法值，避免误用（0为合法rowidx）
#pragma unroll
        for (int i = 0; i < num_experts; i++) {
          local_expert_rowmap[i] = -1;
        }
        for (int i = 0; i < topk; i++) {
          routemap_T this_expert_idx = routemap_topk[row_idx * topk + i];
          probs_T this_expert_prob = probs_topk[row_idx * topk + i];
          if (this_expert_idx < 0) [[likely]]
            continue;
          // 第一次出现，直接搬入
          if (isFirst) [[likely]] {
            isFirst = false;
            probs_unzipped[row_idx] = this_expert_prob;
            expert_idx_unzipped[row_idx] = this_expert_idx;
            local_expert_rowmap[this_expert_idx] = row_idx;
          } else {  // 增广部分, 原子更新行偏置,并计算扩展行索引
            int extended_row_offset;
            extended_row_offset =
                atomicAdd(&atomic_extended_offset_counter[0], 1);
            int extended_row_idx =
                total_zipped_tokens_num + extended_row_offset;
            probs_unzipped[extended_row_idx] = this_expert_prob;
            expert_idx_unzipped[extended_row_idx] = this_expert_idx;
            // 处理专家广播后的行表，用于zip进行收集
            local_expert_rowmap[this_expert_idx] = extended_row_idx;
          }
        }
// ------------------ 更新专家广播后的行表，用于zip进行收集 -----------
// 将合法值和未被触碰的非法值返回给zipped_expertwise_rowmap
#pragma unroll
        for (int i = 0; i < num_experts; i++) {
          zipped_expertwise_rowmap[row_idx * num_experts + i] =
              local_expert_rowmap[i];
          int valid_offset = local_expert_rowmap[i] - total_zipped_tokens_num;
          // 只给增广行传递信号量，非法值保持为0
          if (valid_offset >= 0) {
            atomicExch(&row_valid[valid_offset], row_idx);  // 发送任务信号量
          }
        }
      }
      //这个syncthread可能并不必要，但尽可能为了不让线程间差太多，还是这样吧。
      __syncthreads();
      // 处理完增广事务，对位搬搬移第一次出现的数据,可用inplace优化
      for (int i = threadIdx.x; i < token_length; i += blockDim.x) {
        X_unzipped[row_idx * token_length + i] = X[row_idx * token_length + i];
      }
    } else {  // 线程组1， 忙等、并发处理数据搬移
      if (threadIdx.x == 0) {
        int extended_row_offset = row_idx - total_zipped_tokens_num;
        int local_original_row = -1;
        // 忙等该行的 row_valid变为非-1的合法值
        while (local_original_row == -1) {
          local_original_row = atomicExch(&row_valid[extended_row_offset], -1);
        }
        // 传递给同组线程共享
        shared_original_row = local_original_row;
      }
      __syncthreads();  // 所有该组线程都等0号取任务，再搬移数据
      int original_row = shared_original_row;
      // 搬
      for (int i = threadIdx.x; i < token_length; i += blockDim.x) {
        X_unzipped[row_idx * token_length + i] =
            X[original_row * token_length + i];
      }
    }
  }
}


template <int num_experts>
__global__ void tokens_weighted_zip_kernel(
    const phi::bfloat16 *__restrict__ unzipped_tokens_in,
    const phi::bfloat16 *__restrict__ unzipped_token_probs_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    phi::bfloat16 *__restrict__ weighted_zipped_tokens_out,
    const int total_zipped_tokens_num,
    const int token_length) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;

  const __nv_bfloat16 *unzipped_tokens =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_tokens_in);
  const __nv_bfloat16 *probs_unzipped =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_token_probs_in);
  __nv_bfloat16 *weighted_zipped_tokens =
      reinterpret_cast<__nv_bfloat16 *>(weighted_zipped_tokens_out);

  int local_row_fetchlist[num_experts];
  __nv_bfloat16 local_expert_problist[num_experts];
// 填充该行token被广播到的rows和对应的概率
#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    local_row_fetchlist[expert] =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
    if (local_row_fetchlist[expert] >= 0)
      local_expert_problist[expert] =
          probs_unzipped[local_row_fetchlist[expert]];
  }

  for (int i = threadIdx.x; i < token_length; i += blockDim.x) {
// tensor内部元素加权和
#pragma unroll
    for (int expert = 0; expert < num_experts; ++expert) {
      const bool is_expert_taken = (local_row_fetchlist[expert] >= 0);
      const int fetch_row = local_row_fetchlist[expert];
      if (is_expert_taken) {
      }
      weighted_zipped_tokens[this_row * token_length + i] +=
          is_expert_taken ? local_expert_problist[expert] *
                                unzipped_tokens[fetch_row * token_length + i]
                          : (__nv_bfloat16)0;
    }
  }
}
// ---------------------------- Dispatch ---------------------------------
void dispatch_tokens_unzip(const paddle::Tensor &X,
                           const paddle::Tensor &expert_routemap_topk,
                           const paddle::Tensor &expert_prob_topk,
                           paddle::Tensor &X_unzipped,
                           paddle::Tensor &zipped_expertwise_rowmap,
                           paddle::Tensor &token_prob_unzipped,
                           paddle::Tensor &expert_idx_unzipped,
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

// 定义类型获取宏
#define DTYPE_CASE(dtype, type) dtype == paddle::DataType::type
#define GET_DATA(tensor, type) tensor.data<type>()

// 分发处理不同的类型组合
#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, TOPK, NUM_EXPERTS)               \
  auto kernel = token_unzip_kernel<TOKEN_T, INT_T, PROB_T, TOPK, NUM_EXPERTS>; \
  kernel<<<grid, block, 0, X.stream()>>>(                                      \
      GET_DATA(X, TOKEN_T),                                                    \
      GET_DATA(expert_routemap_topk, INT_T),                                   \
      GET_DATA(expert_prob_topk, PROB_T),                                      \
      GET_DATA(X_unzipped, TOKEN_T),                                           \
      GET_DATA(zipped_expertwise_rowmap, INT_T),                               \
      GET_DATA(token_prob_unzipped, PROB_T),                                   \
      expert_idx_unzipped.data<int>(),                                         \
      atomic_extended_offset_counter.data<int>(),                              \
      row_valid.data<int>(),                                                   \
      total_zipped_tokens_num,                                                 \
      total_unzipped_tokens_num,                                               \
      token_length);

// 可扩展：处理特定的topk和num_experts组合,可根据之后需求进行扩展
#define HANDLE_EXPERT_CASE(TOKEN_T, PROB_T, INT_T) \
  if (topk == 8 && num_experts == 4) {             \
    DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, 8, 4)    \
  } else {                                         \
    /* 超出当前任务范围，*/               \
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

  // 可扩展：根据整型类型控制派发，未来可支持int8，但int64不行，因为下标开销太重了，建议直接cast到int32
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

void dispatch_tokens_weighted_zip(
    const paddle::Tensor &unzipped_tokens,
    const paddle::Tensor &unzipped_token_probs,
    const paddle::Tensor &zipped_expertwise_rowmap,
    paddle::Tensor &weighted_zipped_tokens,
    const int total_zipped_tokens_num,
    const int num_experts,
    const int token_length) {
  dim3 grid, block;
  grid.x = total_zipped_tokens_num;
  block.x = 256;

  // Map data types to C++ types
  if (num_experts == 4) {
    tokens_weighted_zip_kernel<4><<<grid, block, 0, unzipped_tokens.stream()>>>(
        unzipped_tokens.data<phi::bfloat16>(),
        unzipped_token_probs.data<phi::bfloat16>(),
        zipped_expertwise_rowmap.data<int>(),
        weighted_zipped_tokens.data<phi::bfloat16>(),
        total_zipped_tokens_num,
        token_length);
  }
}

// -------------------------------- API -----------------------------------
std::vector<paddle::Tensor> tokens_weighted_zip(
    const paddle::Tensor &unzipped_tokens,
    const paddle::Tensor &unzipped_token_probs,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const int &total_zipped_tokens_num,
    const int &num_experts) {
  PD_CHECK(unzipped_tokens.dtype() == paddle::DataType::BFLOAT16);
  int rows = unzipped_tokens.shape()[0];  // seqlen
  int cols = unzipped_tokens.shape()[1];  //一般为7168

  //------------------------ 输出1张量 ------------------------
  auto weighted_zipped_tokens = paddle::empty({total_zipped_tokens_num, cols},
                                              unzipped_tokens.dtype(),
                                              unzipped_tokens.place());

  dispatch_tokens_weighted_zip(unzipped_tokens,
                               unzipped_token_probs,
                               zipped_expertwise_rowmap,
                               weighted_zipped_tokens,
                               total_zipped_tokens_num,
                               num_experts,
                               cols);
  return {weighted_zipped_tokens};
}

std::vector<paddle::Tensor> tokens_unzip(
    const paddle::Tensor &X,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &expert_prob_topk,
    const int &total_unzipped_tokens_num,
    const int &topk,
    const int &num_experts) {
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16 ||
           X.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  PD_CHECK(expert_prob_topk.dtype() == paddle::DataType::BFLOAT16 ||
           expert_prob_topk.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(expert_routemap_topk.dtype() == paddle::DataType::INT32);
  int rows = X.shape()[0];  // seqlen
  int cols = X.shape()[1];  //一般为7168
  int original_token_num = rows;

  //------------------------ 输出四张量 ------------------------
  auto X_unzipped =
      paddle::empty({total_unzipped_tokens_num, cols}, X.dtype(), X.place());
  // seqlen x num_experts, 每个token的每个专家(如果被发到)对应的行索引, 未初始化
  auto zipped_expertwise_rowmap = paddle::empty(
      {original_token_num, num_experts}, paddle::DataType::INT32, X.place());
  auto token_prob_unzipped = paddle::empty({total_unzipped_tokens_num},
                                           expert_prob_topk.dtype(),
                                           expert_prob_topk.place());
  auto expert_idx_unzipped = paddle::empty(
      {total_unzipped_tokens_num}, paddle::DataType::INT32, X.place());

  //------------------------ 辅助二张量 ------------------------
  //用于原子记录当前以增广的行数，其上限应为 total_unzipped_tokens_num - rows
  auto atomic_extended_offset_counter =
      paddle::zeros({1}, paddle::DataType::INT32, X.place());
  // 增广行数的合法性向量，用于线程组1唤起
  int extended_row_num = total_unzipped_tokens_num - rows;
  auto row_valid =
      paddle::empty({extended_row_num}, paddle::DataType::INT32, X.place());
  void *row_valid_gpu = reinterpret_cast<void *>(row_valid.data<int>());
  cudaMemsetAsync(
      row_valid_gpu, -1, sizeof(int) * extended_row_num, X.stream());


  dispatch_tokens_unzip(X,
                        expert_routemap_topk,
                        expert_prob_topk,
                        X_unzipped,
                        zipped_expertwise_rowmap,
                        token_prob_unzipped,
                        expert_idx_unzipped,
                        atomic_extended_offset_counter,
                        row_valid,
                        rows,
                        total_unzipped_tokens_num,
                        cols,
                        topk,
                        num_experts);
  return {X_unzipped,
          zipped_expertwise_rowmap,
          token_prob_unzipped,
          expert_idx_unzipped};
}

PD_BUILD_OP(tokens_unzip)
    .Inputs({"X", "expert_routemap_topk", "expert_prob_topk"})
    .Outputs({"X_unzipped",
              "zipped_expertwise_rowmap",
              "token_prob_unzipped",
              "expert_idx_unzipped"})
    .Attrs({"total_unzipped_tokens_num: int", "topk: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_unzip));

PD_BUILD_OP(tokens_weighted_zip)
    .Inputs({"unzipped_tokens",
             "unzipped_token_probs",
             "zipped_expertwise_rowmap"})
    .Outputs({"weighted_zipped_tokens"})
    .Attrs({"total_zipped_tokens: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_weighted_zip));

#undef DISPATCH_CASE
#undef DISPATCH_TOKEN_TYPE
#undef DISPATCH_PROB_TYPE