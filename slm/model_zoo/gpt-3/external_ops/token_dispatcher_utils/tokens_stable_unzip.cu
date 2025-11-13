/*
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/
#include "paddle/common/array.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "utils.h"

COMMON_DECLARE_bool(enable_pir_api);


static paddle::DataType TransToDataType(int64_t dtype) {
  if (FLAGS_enable_pir_api) {
    return static_cast<paddle::DataType>(dtype);
  } else {
    return phi::TransToPhiDataType(dtype);
  }
}

#define CUMSUM_BLOCK_SIZE 48  // cumsum开销和并行度之间的tradeoff的结果，勿动
#define CUMSUM_INVALID_TAG -1  // 用于标记无效的cumsum，尝试过-114514但失败了

template <int MAX_NUM_EXPERTS>
struct __align__(16) expert_base_offset {
  int data[MAX_NUM_EXPERTS];
};


// 多阶段算法，控制每block处理的行数来权衡额外开销
//  首先解析routemap来更新专家当前所收到的token数，然后check前一个block给的前缀和并更新给下一个block
//  随后，目的行号的信息已获取，立即开始搬运工作，直至任务完全完成
template <typename X_T,
          typename routemap_T,
          typename probs_T,
          bool has_scale,
          bool fill_x,
          int MAX_NUM_EXPERTS_C>
__global__ void tokens_unzip_stable_kernel(
    const X_T *__restrict__ X,
    const routemap_T *__restrict__ routemap_topk,
    const probs_T *__restrict__ probs_topk,
    const float *__restrict__ XScale,
    const expert_base_offset<MAX_NUM_EXPERTS_C> expert_base_offset,
    X_T *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    probs_T *__restrict__ probs_unzipped,
    float *__restrict__ XScale_unzipped,
    int *global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int token_length,
    const int scale_length,
    const int num_experts,
    const int topk) {
  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  int cumsum_offset[MAX_NUM_EXPERTS_C];
  int local_expert_offsets[MAX_NUM_EXPERTS_C];
  int local_cumsum[MAX_NUM_EXPERTS_C];
#pragma unroll
  for (int i = 0; i < num_experts; i++) {
    cumsum_offset[i] =
        (blockIdx.x == 0)
            ? 0
            : CUMSUM_INVALID_TAG;  // 除了第0个block，其他的都以非法值初始化,因为atomic忙等要用
    local_expert_offsets[i] = expert_base_offset.data[i];
    local_cumsum[i] = 0;
  }
  const int base_row_idx = blockIdx.x * CUMSUM_BLOCK_SIZE;
  __shared__ int shared_expert_rowmap[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS_C];
  __shared__ probs_T
      shared_expert_probmap[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS_C];

  // --------------------- thread0 单线程任务传递 -------------------------
  if (threadIdx.x == 0) [[unlikely]] {
    int local_expert_rowmap[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS_C];
    probs_T local_expert_probs[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS_C];
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
            local_cumsum[expert] + local_expert_offsets[expert];
        local_expert_probs[internal_row][expert] = probs_topk[row * topk + k];
        local_cumsum[expert] += 1;
      }
    }
// -------------------------- 块间通信逻辑 -----------------------------
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

// -------------------------- 块内通信逻辑 -----------------------------
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
  }  // 至此，本线程块内的shared_mem已经规整完毕，接下来是向量化的数据搬运
  __syncthreads();  // 其余线程等到了thread0，工作安排在shared_mem上
  // ------------------------- 所有block内线程 -------------------------
  for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
       row++) {
    if (row >= total_zipped_tokens_num) return;
    const int internal_row = row - block_row_base;
#pragma unroll
    for (int expert = 0; expert < num_experts; expert++) {
      const int unzipped_row_idx = shared_expert_rowmap[internal_row][expert];
      if (threadIdx.x == 0) {
        zipped_expertwise_rowmap[row * num_experts + expert] = unzipped_row_idx;
      }
      if (unzipped_row_idx == -1) continue;
      // 更新三个核心数据结构
      if (threadIdx.x == 0) {
        probs_unzipped[unzipped_row_idx] =
            shared_expert_probmap[internal_row][expert];
      }
      if (fill_x) {
        if constexpr (has_scale) {
          vectorized_memcpy(&XScale[(int64_t)row * (int64_t)scale_length],
                            &XScale_unzipped[(int64_t)unzipped_row_idx *
                                             (int64_t)scale_length],
                            scale_length);
        }
        vectorized_memcpy(
            &X[(int64_t)row * (int64_t)token_length],
            &X_unzipped[(int64_t)unzipped_row_idx * (int64_t)token_length],
            token_length);
      }
    }
  }
}
// ---------------------------- Dispatch ---------------------------------
template <int MAX_NUM_EXPERTS_C>
void dispatch_tokens_unzip_stable(
    const paddle::Tensor &X,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &expert_prob_topk,
    const paddle::optional<paddle::Tensor> &XScale,
    const expert_base_offset<MAX_NUM_EXPERTS_C> &expert_offsets,
    paddle::Tensor &X_unzipped,
    paddle::Tensor &zipped_expertwise_rowmap,
    paddle::Tensor &token_prob_unzipped,
    paddle::Tensor &XScale_unzipped,
    paddle::Tensor &global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int token_length,
    const int topk,  // deprecated
    const int num_experts,
    const int scale_length,
    const bool fill_x) {
  dim3 grid, block;
  grid.x =
      (total_zipped_tokens_num + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  block.x = 256;

  if (grid.x <= 0) return;

// 定义类型获取宏
#define DTYPE_CASE(dtype, type) dtype == paddle::DataType::type
#define GET_DATA(tensor, type) tensor.data<type>()

// 分发处理不同的类型组合
#define DISPATCH_CASE_IMPL(TOKEN_T, PROB_T, INT_T, HAS_SCALE, FILL_X) \
  do {                                                                \
    auto kernel = tokens_unzip_stable_kernel<TOKEN_T,                 \
                                             INT_T,                   \
                                             PROB_T,                  \
                                             HAS_SCALE,               \
                                             FILL_X,                  \
                                             MAX_NUM_EXPERTS_C>;      \
    kernel<<<grid, block, 0, X.stream()>>>(                           \
        GET_DATA(X, TOKEN_T),                                         \
        GET_DATA(expert_routemap_topk, INT_T),                        \
        GET_DATA(expert_prob_topk, PROB_T),                           \
        XScale ? XScale->data<float>() : nullptr,                     \
        expert_offsets,                                               \
        GET_DATA(X_unzipped, TOKEN_T),                                \
        GET_DATA(zipped_expertwise_rowmap, INT_T),                    \
        GET_DATA(token_prob_unzipped, PROB_T),                        \
        XScale_unzipped.data<float>(),                                \
        global_expertwise_block_cumsum.data<int>(),                   \
        total_zipped_tokens_num,                                      \
        token_length,                                                 \
        scale_length,                                                 \
        num_experts,                                                  \
        topk);                                                        \
  } while (0)

#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE)            \
  do {                                                              \
    if (fill_x) {                                                   \
      DISPATCH_CASE_IMPL(TOKEN_T, PROB_T, INT_T, HAS_SCALE, true);  \
    } else {                                                        \
      DISPATCH_CASE_IMPL(TOKEN_T, PROB_T, INT_T, HAS_SCALE, false); \
    }                                                               \
  } while (0)

// 可扩展：处理特定的topk和num_experts组合,可根据之后需求进行扩展
#define HANDLE_EXPERT_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE) \
  DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE);

#define HANDLE_TOKEN_TYPE(PROB_T, INT_T)                           \
  do {                                                             \
    if (DTYPE_CASE(X.dtype(), BFLOAT16)) {                         \
      HANDLE_EXPERT_CASE(phi::bfloat16, PROB_T, INT_T, false);     \
    } else if (DTYPE_CASE(X.dtype(), FLOAT8_E4M3FN)) {             \
      HANDLE_EXPERT_CASE(phi::float8_e4m3fn, PROB_T, INT_T, true); \
    }                                                              \
  } while (0)

#define HANDLE_PROB_TYPE(INT_T)                                 \
  do {                                                          \
    if (DTYPE_CASE(expert_prob_topk.dtype(), BFLOAT16)) {       \
      HANDLE_TOKEN_TYPE(phi::bfloat16, INT_T);                  \
    } else if (DTYPE_CASE(expert_prob_topk.dtype(), FLOAT32)) { \
      HANDLE_TOKEN_TYPE(float, INT_T);                          \
    }                                                           \
  } while (0)

  // 可扩展：根据整型类型控制派发，未来可支持int8，但int64不行，因为下标开销太重了，建议在外面直接cast到int32
  if (DTYPE_CASE(zipped_expertwise_rowmap.dtype(), INT32)) {
    HANDLE_PROB_TYPE(int);
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
    const paddle::optional<paddle::Tensor> &XScale,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &expert_prob_topk,
    const int &topk,
    const int &num_experts,
    const std::vector<int> &tokens_per_expert,
    const int padding_multiplex,
    const bool fill_x) {
  // --------------------- 输入检查与解析 --------------------
  PD_CHECK(X.dtype() == paddle::DataType::BFLOAT16 ||
           X.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  PD_CHECK(expert_routemap_topk.dtype() == paddle::DataType::INT32);
  PD_CHECK(expert_prob_topk.dtype() == paddle::DataType::BFLOAT16 ||
           expert_prob_topk.dtype() == paddle::DataType::FLOAT32);
  if (XScale) {
    PD_CHECK(XScale->dtype() == paddle::DataType::FLOAT32);
  }
  const int64_t rows = X.shape()[0];  // 一般为seqlen
  const int64_t cols = X.shape()[1];  // 一般为7168
  const int64_t quanted_cols = (XScale) ? XScale->shape()[1] : 0;
  /*
  const int max_tokens_per_expert =
      ((max_tokens_per_expert_in + 127) / 128) * 128;
  const int output_rows = num_experts * max_tokens_per_expert;
  */
  //------------------------ 输出缓冲区分配  ------------------------
  paddle::Tensor X_unzipped, XScale_unzipped, zipped_expertwise_rowmap,
      token_prob_unzipped;

  PD_SWITCH_NUM_EXPERTS(
      num_experts, ([&] {
        expert_base_offset<MAX_NUM_EXPERTS_C> expert_offset;
        int64_t tokens_cumulated = 0;
        for (int i = 0; i < MAX_NUM_EXPERTS_C; i++) {
          if (i < num_experts) {
            expert_offset.data[i] = tokens_cumulated;
            tokens_cumulated +=
                ((tokens_per_expert[i] + padding_multiplex - 1) /
                 padding_multiplex) *
                padding_multiplex;
          } else {
            expert_offset.data[i] = 0;
          }
        }

        const int64_t output_rows = tokens_cumulated;
        const int64_t topk_calculated = expert_routemap_topk.shape()[1];

        // FP8 scale unziped缓冲区分配
        if (XScale && fill_x) {
          XScale_unzipped = paddle::empty(
              {output_rows, quanted_cols}, XScale->dtype(), XScale->place());
        } else {  // 让输出时不报错，但实际不会用到
          XScale_unzipped =
              paddle::empty({0}, paddle::DataType::FLOAT32, X.place());
        }

        if (fill_x) {
          X_unzipped = paddle::empty({output_rows, cols}, X.dtype(), X.place());
        } else {
          X_unzipped = paddle::empty({0, cols}, X.dtype(), X.place());
        }
        zipped_expertwise_rowmap = paddle::empty(
            {rows, num_experts}, paddle::DataType::INT32, X.place());
        token_prob_unzipped = paddle::empty(
            {output_rows}, expert_prob_topk.dtype(), expert_prob_topk.place());

        // ------------------------ 缓冲区初始化（适配padding）----------------
        if (fill_x) {
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
          if (XScale) {
            auto XScale_unzipped_ptr =
                reinterpret_cast<void *>(XScale_unzipped.data<float>());
            cudaMemsetAsync(XScale_unzipped_ptr,
                            0,
                            sizeof(float) * output_rows * quanted_cols,
                            XScale_unzipped.stream());
          }
        }
        if (expert_prob_topk.dtype() == paddle::DataType::BFLOAT16) {
          auto token_prob_unzipped_ptr = reinterpret_cast<void *>(
              token_prob_unzipped.data<phi::bfloat16>());
          cudaMemsetAsync(token_prob_unzipped_ptr,
                          0,
                          sizeof(phi::bfloat16) * output_rows,
                          token_prob_unzipped.stream());
        } else if (expert_prob_topk.dtype() == paddle::DataType::FLOAT32) {
          auto token_prob_unzipped_ptr =
              reinterpret_cast<void *>(token_prob_unzipped.data<float>());
          cudaMemsetAsync(token_prob_unzipped_ptr,
                          0,
                          sizeof(float) * output_rows,
                          token_prob_unzipped.stream());
        }
        // ------------ 前缀和辅助数组相关逻辑，“推”式block通信
        // -------------------
        const int64_t cumsum_blocknum =
            (rows + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
        auto global_expertwise_block_cumsum =
            paddle::empty({cumsum_blocknum + 1, num_experts},
                          paddle::DataType::INT32,
                          X.place());
        auto global_expertwise_block_cumsum_ptr = reinterpret_cast<void *>(
            global_expertwise_block_cumsum.data<int>());
        // 设置为非法值CUMSUM_INVALID_TAG，用于线程块等待时使用
        cudaMemsetAsync(global_expertwise_block_cumsum_ptr,
                        CUMSUM_INVALID_TAG,
                        sizeof(int) * (cumsum_blocknum + 1) * num_experts,
                        global_expertwise_block_cumsum.stream());
        if (rows != 0) {
          dispatch_tokens_unzip_stable<MAX_NUM_EXPERTS_C>(
              X,
              expert_routemap_topk,
              expert_prob_topk,
              XScale,
              expert_offset,
              X_unzipped,
              zipped_expertwise_rowmap,
              token_prob_unzipped,
              XScale_unzipped,
              global_expertwise_block_cumsum,
              rows,
              cols,
              topk_calculated,
              num_experts,
              quanted_cols,
              fill_x);
        }
      }));
  return {X_unzipped,
          zipped_expertwise_rowmap,
          token_prob_unzipped,
          XScale_unzipped};
}

PD_BUILD_OP(tokens_unzip_stable)
    .Inputs({"X",
             paddle::Optional("Xscale"),
             "expert_routemap_topk",
             "expert_prob_topk"})
    .Outputs({"X_unzipped",
              "zipped_expertwise_rowmap",
              "token_prob_unzipped",
              paddle::Optional("XScale_unzipped")})
    .Attrs({"topk: int",
            "num_experts: int",
            "tokens_per_expert: std::vector<int>",
            "padding_multiplex: int",
            "fill_output: bool"})
    .SetKernelFn(PD_KERNEL(tokens_unzip_stable));


#undef CUMSUM_BLOCK_SIZE


static int LimitGridDim(int64_t n) {
  return static_cast<int>(std::min<int64_t>(n, 1024 * 1024));
}


template <typename T, bool has_scale>
__global__ void tokens_unzip_gather_kernel(
    const T *__restrict__ x,
    const float *__restrict__ x_scale,
    const int *__restrict__ zipped_expertwise_rowmap,
    T *__restrict__ x_unzipped,
    float *__restrict__ x_scale_unzipped,
    int64_t *__restrict__ index_unzipped,
    int64_t unzipped_rows,
    int64_t zipped_rows,
    int token_length,
    int scale_length,
    int num_experts,
    int expert_id,
    int64_t offset) {
  for (int64_t row = blockIdx.x; row < zipped_rows; row += gridDim.x) {
    int64_t unzipped_row_idx =
        zipped_expertwise_rowmap[row * num_experts + expert_id];
    if (unzipped_row_idx < 0) continue;

    unzipped_row_idx -= offset;
    index_unzipped[unzipped_row_idx] = row;
    if constexpr (has_scale) {
      vectorized_memcpy(x_scale + row * scale_length,
                        x_scale_unzipped + unzipped_row_idx * scale_length,
                        scale_length);
    }
    vectorized_memcpy(x + row * token_length,
                      x_unzipped + unzipped_row_idx * token_length,
                      token_length);
  }
}

std::vector<paddle::Tensor> tokens_unzip_gather(
    const paddle::Tensor &x,
    const paddle::optional<paddle::Tensor> &x_scale,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const int expert_id,
    const std::vector<int64_t> &tokens_per_expert,
    const int padding_multiplex) {
  int num_experts = tokens_per_expert.size();
  PD_CHECK(expert_id >= 0 && expert_id < num_experts);
  std::vector<int64_t> cumsum_tokens(num_experts + 1);
  cumsum_tokens[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    auto padded = (tokens_per_expert[i] + padding_multiplex - 1) /
                  padding_multiplex * padding_multiplex;
    cumsum_tokens[i + 1] = cumsum_tokens[i] + padded;
  }

  int64_t padded_num_tokens =
      cumsum_tokens[expert_id + 1] - cumsum_tokens[expert_id];
  int64_t offset = cumsum_tokens[expert_id];

  auto dtype = x.dtype();
  auto place = x.place();
  auto stream = x.stream();
  auto x_shape = x.shape();
  PD_CHECK(x_shape.size() == 2);
  int64_t zipped_rows = x_shape[0];
  int64_t hidden_size = x_shape[1];
  PADDLE_ENFORCE_LE(
      hidden_size,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("hidden_size should be less than "
                                      "INT_MAX, received hidden_size: (%ld)",
                                      hidden_size));

  std::vector<int64_t> x_scale_shape;
  int64_t quanted_hidden_size = 0;
  bool has_scale = (x_scale.get_ptr() != nullptr);
  if (has_scale) {
    x_scale_shape = x_scale.get().shape();
    PD_CHECK(x_scale_shape.size() == 2);
    PD_CHECK(x_scale_shape[0] == x_shape[0]);
    quanted_hidden_size = x_scale_shape[1];
  }
  PADDLE_ENFORCE_LE(quanted_hidden_size,
                    std::numeric_limits<int32_t>::max(),
                    common::errors::InvalidArgument(
                        "quanted_hidden_size should be less than "
                        "INT_MAX, received quanted_hidden_size: (%ld)",
                        quanted_hidden_size));

  auto x_unzipped =
      paddle::zeros({padded_num_tokens, hidden_size}, dtype, place);
  paddle::Tensor x_scale_unzipped;
  if (has_scale) {
    x_scale_unzipped = paddle::zeros(
        {padded_num_tokens, quanted_hidden_size}, x_scale.get().dtype(), place);
  } else {
    PD_CHECK(hidden_size % 128 == 0);
    quanted_hidden_size = hidden_size / 128;
    x_scale_unzipped = paddle::empty(
        {0, quanted_hidden_size}, paddle::DataType::FLOAT32, place);
  }

  auto index_unzipped = paddle::empty(
      {tokens_per_expert[expert_id]}, paddle::DataType::INT64, place);

  int block = 1024;
  int grid = LimitGridDim(zipped_rows);

#define LAUNCH_TOKENS_UNZIP_GATHER_KERNEL_IMPL(__cpp_dtype, __has_scale) \
  do {                                                                   \
    tokens_unzip_gather_kernel<__cpp_dtype, __has_scale>                 \
        <<<grid, block, 0, stream>>>(                                    \
            x.data<__cpp_dtype>(),                                       \
            __has_scale ? x_scale.get().data<float>() : nullptr,         \
            zipped_expertwise_rowmap.data<int>(),                        \
            x_unzipped.data<__cpp_dtype>(),                              \
            __has_scale ? x_scale_unzipped.data<float>() : nullptr,      \
            index_unzipped.data<int64_t>(),                              \
            tokens_per_expert[expert_id],                                \
            zipped_rows,                                                 \
            hidden_size,                                                 \
            quanted_hidden_size,                                         \
            num_experts,                                                 \
            expert_id,                                                   \
            offset);                                                     \
  } while (0)

#define LAUNCH_TOKENS_UNZIP_GATHER_KERNEL(__cpp_dtype)            \
  do {                                                            \
    if (has_scale) {                                              \
      LAUNCH_TOKENS_UNZIP_GATHER_KERNEL_IMPL(__cpp_dtype, true);  \
    } else {                                                      \
      LAUNCH_TOKENS_UNZIP_GATHER_KERNEL_IMPL(__cpp_dtype, false); \
    }                                                             \
  } while (0)

  if (grid > 0) {
    if (has_scale) {
      LAUNCH_TOKENS_UNZIP_GATHER_KERNEL(phi::float8_e4m3fn);
    } else {
      LAUNCH_TOKENS_UNZIP_GATHER_KERNEL(phi::bfloat16);
    }
  }
  return {x_unzipped, x_scale_unzipped, index_unzipped};
}

template <typename ZipT, typename UnzipT, typename ZipPtrsT, int VecSize>
__global__ void tokens_zip_unique_add_kernel(
    ZipPtrsT zipped_ptrs,
    const UnzipT *__restrict__ unzipped,
    const int64_t *__restrict__ index_unzipped,
    const int64_t unzipped_rows,
    const int64_t subbatch_rows,
    const int64_t hidden_size) {
  for (int64_t unzipped_row = blockIdx.x; unzipped_row < unzipped_rows;
       unzipped_row += gridDim.x) {
    int64_t zipped_row = index_unzipped[unzipped_row];
    auto *zipped_ptr = zipped_ptrs[zipped_row / subbatch_rows] +
                       (zipped_row % subbatch_rows) * hidden_size;
    const auto *unzipped_ptr = unzipped + unzipped_row * hidden_size;
    for (int64_t i = threadIdx.x * VecSize; i < hidden_size;
         i += blockDim.x * VecSize) {
      phi::AlignedVector<ZipT, VecSize> zipped_tmp;
      phi::AlignedVector<UnzipT, VecSize> unzipped_tmp;
      phi::Load(zipped_ptr + i, &zipped_tmp);
      phi::Load(unzipped_ptr + i, &unzipped_tmp);
#pragma unroll
      for (int j = 0; j < VecSize; ++j) {
        zipped_tmp[j] += static_cast<ZipT>(unzipped_tmp[j]);
      }
      phi::Store(zipped_tmp, zipped_ptr + i);
    }
  }
}


template <typename T>
T **GetTensorDevicePtrs(const std::vector<paddle::Tensor> &tensors,
                        paddle::Tensor *ptr_tensor,
                        cudaStream_t stream,
                        phi::Place place) {
  auto nbytes = tensors.size() * sizeof(T *);
  std::vector<const T *> cpu_ptrs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    cpu_ptrs[i] = tensors[i].data<T>();
  }
  *ptr_tensor = paddle::empty(
      {static_cast<int64_t>(nbytes)}, paddle::DataType::UINT8, place);
  auto *device_ptrs = reinterpret_cast<T **>(ptr_tensor->data());
  auto err = cudaMemcpyAsync(
      device_ptrs, cpu_ptrs.data(), nbytes, cudaMemcpyHostToDevice, stream);
  PD_CHECK(
      err == cudaSuccess, "cudaMemcpyAsync error", cudaGetErrorString(err));
  err = cudaStreamSynchronize(stream);
  PD_CHECK(err == cudaSuccess,
           "cudaStreamSynchronize error",
           cudaGetErrorString(err));
  return device_ptrs;
}


std::vector<paddle::Tensor> tokens_zip_unique_add_impl(
    const std::vector<paddle::Tensor> &zipped_origin,
    const paddle::Tensor &unzipped,
    const paddle::Tensor &index_unzipped,
    int64_t zipped_rows,
    int64_t subbatch_rows) {
  int64_t num_split = static_cast<int64_t>(zipped_origin.size());
  PD_CHECK(num_split >= 1, "num_split should be larger than or equal to 1");

  auto zipped_shape = zipped_origin[0].shape();
  auto unzipped_shape = unzipped.shape();
  PD_CHECK(zipped_shape.size() == 2);
  PD_CHECK(unzipped_shape.size() == 2);
  PD_CHECK(zipped_shape[1] == unzipped_shape[1]);

  auto hidden_size = zipped_shape[1];

  auto out_dtype = zipped_origin[0].dtype();
  auto in_dtype = unzipped.dtype();
  auto place = zipped_origin[0].place();

  if (zipped_rows <= 0) {
    return zipped_origin;
  }

  if (subbatch_rows <= 0) {
    subbatch_rows = zipped_rows;
  }
  subbatch_rows = std::min(zipped_rows, subbatch_rows);
  auto desired_num_split = (zipped_rows + subbatch_rows - 1) / subbatch_rows;
  auto remainder_rows = zipped_rows - (desired_num_split - 1) * subbatch_rows;

  std::vector<paddle::Tensor> zipped;
  zipped.reserve(desired_num_split);
  if (zipped_shape[0] == 0) {
    PD_CHECK(num_split == 1,
             "When input is 0-size tensor, it should be a single tensor "
             "instead of a tensor list");
    for (int64_t i = 0; i < desired_num_split; ++i) {
      auto tmp_rows =
          (i + 1 == desired_num_split ? remainder_rows : subbatch_rows);
      zipped.emplace_back(
          paddle::zeros({tmp_rows, hidden_size}, out_dtype, place));
    }
    num_split = desired_num_split;
  } else {
    PD_CHECK(num_split == desired_num_split);
    for (int64_t i = 0; i < desired_num_split; ++i) {
      auto tmp_shape = zipped_origin[i].shape();
      auto tmp_dtype = zipped_origin[i].dtype();
      PD_CHECK(tmp_shape.size() == 2);
      if (i + 1 == desired_num_split) {
        PD_CHECK(tmp_shape[0] == remainder_rows);
      } else {
        PD_CHECK(tmp_shape[0] == subbatch_rows);
      }
      PD_CHECK(tmp_shape[1] == hidden_size);
      PD_CHECK(tmp_dtype == out_dtype);

      zipped.emplace_back(zipped_origin[i]);
    }
  }

  auto index_shape = index_unzipped.shape();
  PD_CHECK(index_shape.size() == 1);
  auto unzipped_rows = index_shape[0];
  PD_CHECK(unzipped_rows <= zipped_rows);
  PD_CHECK(unzipped_rows <= unzipped_shape[0]);

  constexpr int kVecSize = 4;
  PD_CHECK(hidden_size % kVecSize == 0);

  int block = 1024;
  int grid = LimitGridDim(unzipped_rows);

  auto stream = unzipped.stream();
  paddle::Tensor ptr_tensor;

#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD_CASE_IMPL(__ZipT, __UnzipT, __out_ptrs)  \
  do {                                                                        \
    auto stream = unzipped.stream();                                          \
    tokens_zip_unique_add_kernel<                                             \
        __ZipT,                                                               \
        __UnzipT,                                                             \
        typename std::remove_reference<decltype(__out_ptrs)>::type,           \
        kVecSize><<<grid, block, 0, stream>>>(__out_ptrs,                     \
                                              unzipped.data<__UnzipT>(),      \
                                              index_unzipped.data<int64_t>(), \
                                              unzipped_rows,                  \
                                              subbatch_rows,                  \
                                              hidden_size);                   \
  } while (0)


#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, __num_split) \
  if (num_split <= __num_split) {                                            \
    phi::Array<__ZipT *, __num_split> array;                                 \
    for (int64_t i = 0; i < num_split; ++i) {                                \
      array[i] = zipped[i].data<__ZipT>();                                   \
    }                                                                        \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_CASE_IMPL(__ZipT, __UnzipT, array);         \
    break;                                                                   \
  }


#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD_DYNAMIC_CASE(__ZipT, __UnzipT)      \
  paddle::Tensor ptr_tensor;                                             \
  auto device_ptrs =                                                     \
      GetTensorDevicePtrs<__ZipT>(zipped, &ptr_tensor, stream, place);   \
  LAUNCH_TOKENS_ZIP_UNIQUE_ADD_CASE_IMPL(__ZipT, __UnzipT, device_ptrs); \
  break


#define LAUNCH_TOKENS_ZIP_UNIQUE_ADD(__ZipT, __UnzipT)           \
  do {                                                           \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 1);  \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 2);  \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 4);  \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 8);  \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_FIX_CASE(__ZipT, __UnzipT, 16); \
    LAUNCH_TOKENS_ZIP_UNIQUE_ADD_DYNAMIC_CASE(__ZipT, __UnzipT); \
  } while (0)


  if (grid > 0) {
    if (out_dtype == paddle::DataType::FLOAT32 &&
        in_dtype == paddle::DataType::BFLOAT16) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(float, phi::bfloat16);
    } else if (out_dtype == paddle::DataType::BFLOAT16 &&
               in_dtype == out_dtype) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(phi::bfloat16, phi::bfloat16);
    } else if (out_dtype == paddle::DataType::FLOAT32 &&
               in_dtype == out_dtype) {
      LAUNCH_TOKENS_ZIP_UNIQUE_ADD(float, float);
    } else {
      PD_THROW("Unsupported data type");
    }
  }
  return zipped;
}

std::vector<paddle::Tensor> tokens_zip_unique_add(
    const paddle::Tensor &zipped_origin,
    const paddle::Tensor &unzipped,
    const paddle::Tensor &index_unzipped,
    int64_t zipped_rows) {
  return tokens_zip_unique_add_impl(
      {zipped_origin}, unzipped, index_unzipped, zipped_rows, 0);
}

void tokens_zip_unique_add_subbatch(
    const std::vector<paddle::Tensor> &zipped_origin,
    const paddle::Tensor &unzipped,
    const paddle::Tensor &index_unzipped,
    int64_t zipped_rows,
    int64_t subbatch_rows) {
  tokens_zip_unique_add_impl(
      zipped_origin, unzipped, index_unzipped, zipped_rows, subbatch_rows);
}

template <typename T>
struct UnzippedProbInfo {
  const T *__restrict__ data;
  int64_t offset;
};

template <typename T, int MAX_NUM_EXPERTS_C>
__global__ void tokens_zip_prob_kernel(
    phi::Array<UnzippedProbInfo<T>, MAX_NUM_EXPERTS_C> unzipped_probs,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ dispatched_indices,
    T *zipped_probs,
    int64_t zipped_rows,
    int topk,
    int num_expert) {
  int64_t idx = threadIdx.x + static_cast<int64_t>(blockDim.x) * blockIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t limit = zipped_rows * topk;
  while (idx < limit) {
    auto zipped_row = idx / topk;
    auto topk_idx = idx % topk;
    auto expert_id = dispatched_indices[idx];
    T value = static_cast<T>(0);
    if (expert_id >= 0) {
      auto unzipped_row =
          zipped_expertwise_rowmap[zipped_row * num_expert + expert_id];
      if (unzipped_row >= 0) {
        unzipped_row -= unzipped_probs[expert_id].offset;
        value = unzipped_probs[expert_id].data[unzipped_row];
      }
    }
    zipped_probs[idx] = value;
    idx += stride;
  }
}

template <typename T>
std::vector<paddle::Tensor> tokens_zip_prob_impl(
    const std::vector<paddle::Tensor> &unzipped_probs,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const paddle::Tensor &dispatched_indices,
    paddle::DataType dtype) {
  auto zipped_expertwise_rowmap_shape = zipped_expertwise_rowmap.shape();
  auto dispatched_indices_shape = dispatched_indices.shape();
  PD_CHECK(zipped_expertwise_rowmap_shape.size() == 2);
  PD_CHECK(dispatched_indices_shape.size() == 2);
  PD_CHECK(zipped_expertwise_rowmap_shape[0] == dispatched_indices_shape[0]);

  int64_t zipped_rows = zipped_expertwise_rowmap_shape[0];
  int num_expert = zipped_expertwise_rowmap_shape[1];
  int topk = dispatched_indices_shape[1];
  PD_CHECK(unzipped_probs.size() == num_expert);

  auto zipped_probs =
      paddle::empty({zipped_rows, topk}, dtype, unzipped_probs[0].place());

  PD_SWITCH_NUM_EXPERTS(
      num_expert, ([&] {
        phi::Array<UnzippedProbInfo<T>, MAX_NUM_EXPERTS_C> unzipped_probs_info;
        int64_t offset = 0;
        for (int i = 0; i < num_expert; ++i) {
          auto shape = unzipped_probs[i].shape();
          PD_CHECK(shape.size() == 1);
          unzipped_probs_info[i].data = unzipped_probs[i].data<T>();
          unzipped_probs_info[i].offset = offset;
          offset += shape[0];
        }

        int thread = 1024;
        int grid = LimitGridDim((zipped_rows * topk + thread - 1) / thread);

        if (grid > 0) {
          tokens_zip_prob_kernel<T, MAX_NUM_EXPERTS_C>
              <<<grid, thread, 0, zipped_probs.stream()>>>(
                  unzipped_probs_info,
                  zipped_expertwise_rowmap.data<int>(),
                  dispatched_indices.data<int>(),
                  zipped_probs.data<T>(),
                  zipped_rows,
                  topk,
                  num_expert);
        }
      }));
  return {zipped_probs};
}


std::vector<paddle::Tensor> tokens_zip_prob(
    const std::vector<paddle::Tensor> &unzipped_probs,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const paddle::Tensor &dispatched_indices) {
  PD_CHECK(zipped_expertwise_rowmap.dtype() == paddle::DataType::INT32);
  PD_CHECK(dispatched_indices.dtype() == paddle::DataType::INT32);

  auto dtype = unzipped_probs[0].dtype();
  if (dtype == paddle::DataType::FLOAT32) {
    return tokens_zip_prob_impl<float>(
        unzipped_probs, zipped_expertwise_rowmap, dispatched_indices, dtype);
  } else if (dtype == paddle::DataType::BFLOAT16) {
    return tokens_zip_prob_impl<phi::bfloat16>(
        unzipped_probs, zipped_expertwise_rowmap, dispatched_indices, dtype);
  } else {
    PD_THROW("Unsupported data type: %s", dtype);
  }
}


template <typename InT, typename OutT, typename InPtrsT, int VecSize>
__global__ void merge_subbatch_cast_kernel(const InPtrsT in_ptrs,
                                           OutT *__restrict__ out,
                                           int64_t total_num,
                                           int64_t subbatch_num) {
  int64_t idx =
      (threadIdx.x + static_cast<int64_t>(blockDim.x) * blockIdx.x) * VecSize;
  int64_t stride = (static_cast<int64_t>(blockDim.x) * gridDim.x) * VecSize;

  while (idx < total_num) {
    const InT *x_ptr = in_ptrs[idx / subbatch_num] + idx % subbatch_num;
    phi::AlignedVector<InT, VecSize> in_data;
    phi::Load(x_ptr, &in_data);
    if constexpr (std::is_same<InT, OutT>::value) {
      phi::Store(in_data, out + idx);
    } else {
      phi::AlignedVector<OutT, VecSize> out_data;
#pragma unroll
      for (int i = 0; i < VecSize; ++i) {
        out_data[i] = static_cast<OutT>(in_data[i]);
      }
      phi::Store(out_data, out + idx);
    }
    idx += stride;
  }
}


std::vector<paddle::Tensor> merge_subbatch_cast(
    const std::vector<paddle::Tensor> &x, int64_t int_dtype) {
  if (x.empty()) return {};

  auto in_dtype = x[0].dtype();
  auto merged_dtype = TransToDataType(int_dtype);

  auto place = x[0].place();
  auto merged_shape = x[0].shape();
  int64_t subbatch_rows = merged_shape[0];
  for (size_t i = 1; i < x.size(); ++i) {
    auto tmp_shape = x[i].shape();
    PD_CHECK(tmp_shape.size() == merged_shape.size());
    for (size_t j = 1; j < tmp_shape.size(); ++j) {
      PD_CHECK(tmp_shape[j] == merged_shape[j]);
    }
    if (i + 1 != x.size()) {
      PD_CHECK(tmp_shape[0] == subbatch_rows);
    } else {
      PD_CHECK(tmp_shape[0] <= subbatch_rows);
    }
    merged_shape[0] += tmp_shape[0];

    PD_CHECK(x[i].dtype() == in_dtype);
  }

  auto output = paddle::empty(merged_shape, merged_dtype, place);
  int64_t hidden_size = 1;
  for (size_t i = 1; i < merged_shape.size(); ++i) {
    hidden_size *= merged_shape[i];
  }

  int64_t total_num = merged_shape[0] * hidden_size;
  int64_t subbatch_num = subbatch_rows * hidden_size;

  constexpr int kVecSize = 4;
  PD_CHECK(total_num % kVecSize == 0);
  PD_CHECK(subbatch_num % kVecSize == 0);
  auto stream = output.stream();

  int thread = 1024;
  int grid = LimitGridDim((total_num / kVecSize + thread - 1) / thread);
  auto num_split = static_cast<int64_t>(x.size());

#define LAUNCH_MERGE_SUBBATCH_CAST_CASE_IMPL(__InT, __OutT, __in_ptrs) \
  do {                                                                 \
    merge_subbatch_cast_kernel<                                        \
        __InT,                                                         \
        __OutT,                                                        \
        typename std::remove_reference<decltype(__in_ptrs)>::type,     \
        kVecSize><<<grid, thread, 0, stream>>>(                        \
        __in_ptrs, output.data<__OutT>(), total_num, subbatch_num);    \
  } while (0)

#define LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, __num_split) \
  if (num_split <= __num_split) {                                       \
    phi::Array<const __InT *, __num_split> array;                       \
    for (int64_t i = 0; i < num_split; ++i) {                           \
      array[i] = x[i].data<__InT>();                                    \
    }                                                                   \
    LAUNCH_MERGE_SUBBATCH_CAST_CASE_IMPL(__InT, __OutT, array);         \
    break;                                                              \
  }

#define LAUNCH_MERGE_SUBBATCH_CAST_DYNAMIC_CASE(__InT, __OutT)      \
  paddle::Tensor ptr_tensor;                                        \
  auto device_ptrs =                                                \
      GetTensorDevicePtrs<__InT>(x, &ptr_tensor, stream, place);    \
  LAUNCH_MERGE_SUBBATCH_CAST_CASE_IMPL(__InT, __OutT, device_ptrs); \
  break


#define LAUNCH_MERGE_SUBBATCH_CAST(__InT, __OutT)           \
  do {                                                      \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 1);  \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 2);  \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 4);  \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 8);  \
    LAUNCH_MERGE_SUBBATCH_CAST_FIX_CASE(__InT, __OutT, 16); \
    LAUNCH_MERGE_SUBBATCH_CAST_DYNAMIC_CASE(__InT, __OutT); \
  } while (0)


  if (grid > 0) {
    if (in_dtype == paddle::DataType::FLOAT32 &&
        merged_dtype == paddle::DataType::BFLOAT16) {
      LAUNCH_MERGE_SUBBATCH_CAST(float, paddle::bfloat16);
    } else if (in_dtype == paddle::DataType::BFLOAT16 &&
               merged_dtype == paddle::DataType::FLOAT32) {
      LAUNCH_MERGE_SUBBATCH_CAST(paddle::bfloat16, float);
    } else if (in_dtype == paddle::DataType::FLOAT32 &&
               merged_dtype == paddle::DataType::FLOAT32) {
      LAUNCH_MERGE_SUBBATCH_CAST(float, float);
    } else if (in_dtype == paddle::DataType::BFLOAT16 &&
               merged_dtype == paddle::DataType::BFLOAT16) {
      LAUNCH_MERGE_SUBBATCH_CAST(paddle::bfloat16, paddle::bfloat16);
    } else {
      PD_THROW("Unsupported data type");
    }
  }

  return {output};
}


PD_BUILD_OP(tokens_unzip_gather)
    .Inputs({"x", paddle::Optional("x_scale"), "zipped_expertwise_rowmap"})
    .Outputs({"x_unzipped",
              paddle::Optional("x_scale_unzipped"),
              "idx_unzipped"})
    .Attrs({"expert_id: int",
            "tokens_per_expert: std::vector<int64_t>",
            "padding_multiplex: int"})
    .SetKernelFn(PD_KERNEL(tokens_unzip_gather));


PD_BUILD_OP(tokens_zip_unique_add)
    .Inputs({"x_zipped", "x_unzipped", "idx_unzipped"})
    .Outputs({"y_zipped"})
    .Attrs({"zipped_rows: int64_t"})
    .SetKernelFn(PD_KERNEL(tokens_zip_unique_add));

PD_BUILD_OP(tokens_zip_unique_add_subbatch)
    .Inputs({paddle::Vec("x_zipped"), "x_unzipped", "idx_unzipped"})
    .Outputs({paddle::Vec("y_zipped")})
    .SetInplaceMap({{paddle::Vec("x_zipped"), paddle::Vec("y_zipped")}})
    .Attrs({"zipped_rows: int64_t", "subbatch_rows: int64_t"})
    .SetKernelFn(PD_KERNEL(tokens_zip_unique_add_subbatch));


PD_BUILD_OP(tokens_zip_prob)
    .Inputs({paddle::Vec("unzipped_prob"),
             "zipped_expertwise_rowmap",
             "dispatched_indices"})
    .Outputs({"zipped_prob"})
    .SetKernelFn(PD_KERNEL(tokens_zip_prob));


PD_BUILD_OP(merge_subbatch_cast)
    .Inputs({paddle::Vec("x")})
    .Outputs({"y"})
    .Attrs({"dtype: int64_t"})
    .SetKernelFn(PD_KERNEL(merge_subbatch_cast));
