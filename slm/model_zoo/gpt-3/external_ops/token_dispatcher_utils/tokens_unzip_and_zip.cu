// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
      // 这个syncthread可能并不必要，但尽可能为了不让线程间差太多，还是这样吧。
      __syncthreads();
      // 处理完增广事务，对位搬搬移第一次出现的数据,可用inplace优化
      vectorized_memcpy(&X[(int64_t)row_idx * (int64_t)token_length],
                        &X_unzipped[(int64_t)row_idx * (int64_t)token_length],
                        token_length);
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
      vectorized_memcpy(&X[(int64_t)original_row * (int64_t)token_length],
                        &X_unzipped[(int64_t)row_idx * (int64_t)token_length],
                        token_length);
    }
  }
}


template <int num_experts, bool MP = true>
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

// -------------------------初始化任务表 ------------------------
#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    const int fetch_row =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
    local_row_fetchlist[expert] = fetch_row;
    if (fetch_row >= 0) {
      local_expert_problist[expert] = probs_unzipped[fetch_row];
    }
  }

  constexpr int vecSize = 2;  // __nv_bfloat162 = 2 x bfloat16
  const int num_full_vec = token_length / vecSize;
  const int remaining_elems = token_length % vecSize;
  const int thread_stride = blockDim.x * vecSize;

  if constexpr (MP) {
    // ------------------------ 手动混合精度 ---------------------------------
    // 齐整区域向量化搬移
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      float2 sum = {0.0f, 0.0f};
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &weighted_zipped_tokens[(int64_t)this_row * (int64_t)token_length +
                                  x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        const int fetch_row_index = fetch_row >= 0 ? fetch_row : 0;
        // 手动类型提升
        float2 token_vec =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(
                &unzipped_tokens[(int64_t)fetch_row_index *
                                     (int64_t)token_length +
                                 x_offset]));
        float prob = fetch_row >= 0
                         ? __bfloat162float(local_expert_problist[expert])
                         : 0.0f;
        float2 prob_vec = {prob, prob};
        sum.x = __fmaf_rn(token_vec.x, prob_vec.x, sum.x);
        sum.y = __fmaf_rn(token_vec.y, prob_vec.y, sum.y);
      }
      // 类型下降为原有精度
      *out_ptr = __float22bfloat162_rn(sum);
    }

    // 剩余元素处理
    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      float sum = 0.0f;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        int fetch_row_index = fetch_row >= 0 ? fetch_row : 0;
        float token_val = __bfloat162float(
            unzipped_tokens[(int64_t)fetch_row_index * (int64_t)token_length +
                            i]);
        float prob = fetch_row >= 0
                         ? __bfloat162float(local_expert_problist[expert])
                         : 0.0f;
        sum += prob * token_val;
      }
      weighted_zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] =
          __float2bfloat16_rn(sum);
    }
  } else {
    // ------------------------ BF16 intrinsics 加权累加 -----------------------
    // 齐整区域向量化搬移
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      __nv_bfloat162 sum = {0, 0};
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &weighted_zipped_tokens[(int64_t)this_row * (int64_t)token_length +
                                  x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        const int fetch_row_index = fetch_row >= 0 ? fetch_row : 0;
        __nv_bfloat162 token_vec = *reinterpret_cast<const __nv_bfloat162 *>(
            &unzipped_tokens[(int64_t)fetch_row_index * (int64_t)token_length +
                             x_offset]);
        __nv_bfloat16 prob =
            fetch_row >= 0 ? local_expert_problist[expert] : (__nv_bfloat16)0;
        __nv_bfloat162 prob_vec = {prob, prob};
        sum = __hfma2(token_vec, prob_vec, sum);
      }
      *out_ptr = sum;
    }

    // 剩余元素处理
    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      __nv_bfloat16 sum = (__nv_bfloat16)0;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        int fetch_row_index = fetch_row >= 0 ? fetch_row : 0;
        __nv_bfloat16 token_val =
            unzipped_tokens[(int64_t)fetch_row_index * (int64_t)token_length +
                            i];
        __nv_bfloat16 prob =
            fetch_row >= 0 ? local_expert_problist[expert] : (__nv_bfloat16)0;
        sum += prob * token_val;
      }
      weighted_zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] =
          sum;
    }
  }
}

template <int MAX_NUM_EXPERTS_C, bool MP = true>
__global__ void tokens_zip_kernel(
    const phi::bfloat16 *__restrict__ unzipped_tokens_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ expert_routemap_topk,
    const phi::bfloat16 *__restrict__ unzipped_token_probs,
    phi::bfloat16 *__restrict__ zipped_tokens_out,
    phi::bfloat16 *__restrict__ zipped_probs_topk,
    const int total_zipped_tokens_num,
    const int token_length,
    const int num_experts,
    const int topk) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;

  const __nv_bfloat16 *unzipped_tokens =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_tokens_in);
  __nv_bfloat16 *zipped_tokens =
      reinterpret_cast<__nv_bfloat16 *>(zipped_tokens_out);

  int local_row_fetchlist[MAX_NUM_EXPERTS_C];

// -------------------------初始化任务表 ------------------------
#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    const int fetch_row =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
    local_row_fetchlist[expert] = fetch_row;
  }

#pragma unroll
  for (int k = 0; k < topk; ++k) {
    const int expert_idx = expert_routemap_topk[this_row * topk + k];
    if (expert_idx < 0) [[likely]]
      continue;
    const int expert_fetch_row = local_row_fetchlist[expert_idx];
    zipped_probs_topk[this_row * topk + k] =
        unzipped_token_probs[expert_fetch_row];
  }

  constexpr int vecSize = 2;  // __nv_bfloat162 = 2 x bfloat16
  const int num_full_vec = token_length / vecSize;
  const int remaining_elems = token_length % vecSize;
  const int thread_stride = blockDim.x * vecSize;

  if constexpr (MP) {
    // ------------------------ 手动混合精度 ---------------------------------
    // 齐整区域向量化搬移
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      float2 sum = {0.0f, 0.0f};
      __nv_bfloat162 raw = {0, 0};
      int aggreg_cnt = 0;
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &zipped_tokens[(int64_t)this_row * (int64_t)token_length + x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        aggreg_cnt++;
        // 手动类型提升
        raw = *reinterpret_cast<const __nv_bfloat162 *>(
            &unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                             x_offset]);
        float2 token_vec = __bfloat1622float2(raw);
        sum.x = __fadd_rn(token_vec.x, sum.x);
        sum.y = __fadd_rn(token_vec.y, sum.y);
      }
      // 选择性类型下降为原有精度
      *out_ptr = (aggreg_cnt > 1) ? __float22bfloat162_rn(sum) : raw;
    }

    // 剩余元素处理
    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      float sum = 0.0f;
      __nv_bfloat16 raw = 0;
      int aggreg_cnt = 0;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        aggreg_cnt++;
        raw = unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length + i];
        float token_val = __bfloat162float(raw);
        sum = __fadd_rn(token_val, sum);
      }
      zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] =
          (aggreg_cnt > 1) ? __float2bfloat16_rn(sum) : raw;
    }
  } else {
    // ------------------------ BF16 intrinsics 加权累加 -----------------------
    // 齐整区域向量化搬移
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      __nv_bfloat162 sum = {0, 0};
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &zipped_tokens[(int64_t)this_row * (int64_t)token_length + x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        __nv_bfloat162 token_vec = *reinterpret_cast<const __nv_bfloat162 *>(
            &unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                             x_offset]);
        sum = __hadd2(sum, token_vec);
      }
      *out_ptr = sum;
    }

    // 剩余元素处理
    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      __nv_bfloat16 sum = (__nv_bfloat16)0;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        __nv_bfloat16 token_val =
            unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length + i];
        sum = __hadd(sum, token_val);
      }
      zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] = sum;
    }
  }
}

template <int MAX_NUM_EXPERTS_C, bool MP = true>
__global__ void tokens_zip_kernel(
    const phi::bfloat16 *__restrict__ unzipped_tokens_in,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ expert_routemap_topk,
    const float *__restrict__ unzipped_token_probs,
    phi::bfloat16 *__restrict__ zipped_tokens_out,
    float *__restrict__ zipped_probs_topk,
    const int total_zipped_tokens_num,
    const int token_length,
    const int num_experts,
    const int topk) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;

  const __nv_bfloat16 *unzipped_tokens =
      reinterpret_cast<const __nv_bfloat16 *>(unzipped_tokens_in);
  __nv_bfloat16 *zipped_tokens =
      reinterpret_cast<__nv_bfloat16 *>(zipped_tokens_out);

  int local_row_fetchlist[MAX_NUM_EXPERTS_C];

// -------------------------初始化任务表 ------------------------
#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    const int fetch_row =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
    local_row_fetchlist[expert] = fetch_row;
  }

#pragma unroll
  for (int k = 0; k < topk; ++k) {
    const int expert_idx = expert_routemap_topk[this_row * topk + k];
    if (expert_idx < 0) [[likely]]
      continue;
    const int expert_fetch_row = local_row_fetchlist[expert_idx];
    zipped_probs_topk[this_row * topk + k] =
        unzipped_token_probs[expert_fetch_row];
  }

  constexpr int vecSize = 2;  // __nv_bfloat162 = 2 x bfloat16
  const int num_full_vec = token_length / vecSize;
  const int remaining_elems = token_length % vecSize;
  const int thread_stride = blockDim.x * vecSize;

  if constexpr (MP) {
    // ------------------------ 手动混合精度 ---------------------------------
    // 齐整区域向量化搬移
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      float2 sum = {0.0f, 0.0f};
      __nv_bfloat162 raw = {0, 0};
      int aggreg_cnt = 0;
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &zipped_tokens[(int64_t)this_row * (int64_t)token_length + x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        aggreg_cnt++;
        // 手动类型提升
        raw = *reinterpret_cast<const __nv_bfloat162 *>(
            &unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                             x_offset]);
        float2 token_vec = __bfloat1622float2(raw);
        sum.x = __fadd_rn(token_vec.x, sum.x);
        sum.y = __fadd_rn(token_vec.y, sum.y);
      }
      // 选择性类型下降为原有精度
      *out_ptr = (aggreg_cnt > 1) ? __float22bfloat162_rn(sum) : raw;
    }

    // 剩余元素处理
    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      float sum = 0.0f;
      __nv_bfloat16 raw = 0;
      int aggreg_cnt = 0;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        aggreg_cnt++;
        raw = unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length + i];
        float token_val = __bfloat162float(raw);
        sum = __fadd_rn(token_val, sum);
      }
      zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] =
          (aggreg_cnt > 1) ? __float2bfloat16_rn(sum) : raw;
    }
  } else {
    // ------------------------ BF16 intrinsics 加权累加 -----------------------
    // 齐整区域向量化搬移
    for (int x_offset = threadIdx.x * vecSize;
         x_offset < num_full_vec * vecSize;
         x_offset += thread_stride) {
      __nv_bfloat162 sum = {0, 0};
      __nv_bfloat162 *out_ptr = reinterpret_cast<__nv_bfloat162 *>(
          &zipped_tokens[(int64_t)this_row * (int64_t)token_length + x_offset]);
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        const int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        __nv_bfloat162 token_vec = *reinterpret_cast<const __nv_bfloat162 *>(
            &unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                             x_offset]);
        sum = __hadd2(sum, token_vec);
      }
      *out_ptr = sum;
    }

    // 剩余元素处理
    for (int i = num_full_vec * vecSize + threadIdx.x; i < token_length;
         i += blockDim.x) {
      __nv_bfloat16 sum = (__nv_bfloat16)0;
#pragma unroll
      for (int expert = 0; expert < num_experts; ++expert) {
        int fetch_row = local_row_fetchlist[expert];
        if (fetch_row < 0) continue;
        __nv_bfloat16 token_val =
            unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length + i];
        sum = __hadd(sum, token_val);
      }
      zipped_tokens[(int64_t)this_row * (int64_t)token_length + i] = sum;
    }
  }
}

template <int MAX_NUM_EXPERTS_C>
__global__ void tokens_zip_kernel(
    const float *__restrict__ unzipped_tokens,
    const int *__restrict__ zipped_expertwise_rowmap,
    const int *__restrict__ expert_routemap_topk,
    const float *__restrict__ unzipped_token_probs,
    float *__restrict__ zipped_tokens,
    float *__restrict__ zipped_probs_topk,
    const int total_zipped_tokens_num,
    const int token_length,
    const int num_experts,
    const int topk) {
  const int this_row = blockIdx.x;
  if (this_row >= total_zipped_tokens_num) return;
  int local_row_fetchlist[MAX_NUM_EXPERTS_C];

// -------------------------初始化任务表 ------------------------
#pragma unroll
  for (int expert = 0; expert < num_experts; ++expert) {
    const int fetch_row =
        zipped_expertwise_rowmap[this_row * num_experts + expert];
    local_row_fetchlist[expert] = fetch_row;
  }

#pragma unroll
  for (int k = 0; k < topk; ++k) {
    const int expert_idx = expert_routemap_topk[this_row * topk + k];
    if (expert_idx < 0) [[likely]]
      continue;
    const int expert_fetch_row = local_row_fetchlist[expert_idx];
    zipped_probs_topk[this_row * topk + k] =
        unzipped_token_probs[expert_fetch_row];
  }

  const int thread_stride = blockDim.x;

  // ------------------------ 手动混合精度 ---------------------------------
  // 齐整区域向量化搬移
  for (int x_offset = threadIdx.x; x_offset < token_length;
       x_offset += thread_stride) {
    float sum = 0.0f;
#pragma unroll
    for (int expert = 0; expert < num_experts; ++expert) {
      const int fetch_row = local_row_fetchlist[expert];
      if (fetch_row < 0) continue;
      // 手动类型提升
      sum += unzipped_tokens[(int64_t)fetch_row * (int64_t)token_length +
                             x_offset];
    }
    zipped_tokens[(int64_t)this_row * (int64_t)token_length + x_offset] = sum;
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

template <int MAX_NUM_EXPERTS_C>
void dispatch_tokens_zip(const paddle::Tensor &unzipped_tokens,
                         const paddle::Tensor &zipped_expertwise_rowmap,
                         const paddle::Tensor &expert_routemap_topk,
                         const paddle::Tensor &unzipped_token_probs,
                         paddle::Tensor &zipped_tokens,
                         paddle::Tensor &zipped_probs_topk,
                         const int total_zipped_tokens_num,
                         const int num_experts,
                         const int token_length,
                         const int topk) {
  dim3 grid, block;
  grid.x = total_zipped_tokens_num;
  block.x = 256;

  // Map data types to C++ types

  if (unzipped_tokens.dtype() == paddle::DataType::BFLOAT16) {
    if (zipped_probs_topk.dtype() == paddle::DataType::FLOAT32) {
      tokens_zip_kernel<MAX_NUM_EXPERTS_C>
          <<<grid, block, 0, unzipped_tokens.stream()>>>(
              unzipped_tokens.data<phi::bfloat16>(),
              zipped_expertwise_rowmap.data<int>(),
              expert_routemap_topk.data<int>(),
              unzipped_token_probs.data<float>(),
              zipped_tokens.data<phi::bfloat16>(),
              zipped_probs_topk.data<float>(),
              total_zipped_tokens_num,
              token_length,
              num_experts,
              topk);
    } else if (zipped_probs_topk.dtype() == paddle::DataType::BFLOAT16) {
      tokens_zip_kernel<MAX_NUM_EXPERTS_C>
          <<<grid, block, 0, unzipped_tokens.stream()>>>(
              unzipped_tokens.data<phi::bfloat16>(),
              zipped_expertwise_rowmap.data<int>(),
              expert_routemap_topk.data<int>(),
              unzipped_token_probs.data<phi::bfloat16>(),
              zipped_tokens.data<phi::bfloat16>(),
              zipped_probs_topk.data<phi::bfloat16>(),
              total_zipped_tokens_num,
              token_length,
              num_experts,
              topk);
    }
  } else if (unzipped_tokens.dtype() == paddle::DataType::FLOAT32) {
    tokens_zip_kernel<MAX_NUM_EXPERTS_C>
        <<<grid, block, 0, unzipped_tokens.stream()>>>(
            unzipped_tokens.data<float>(),
            zipped_expertwise_rowmap.data<int>(),
            expert_routemap_topk.data<int>(),
            unzipped_token_probs.data<float>(),
            zipped_tokens.data<float>(),
            zipped_probs_topk.data<float>(),
            total_zipped_tokens_num,
            token_length,
            num_experts,
            topk);
  }
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
  int cols = unzipped_tokens.shape()[1];  // 一般为7168

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

/*
PD_BUILD_OP(tokens_zip)
    .Inputs({"unzipped_tokens", "zipped_expertwise_rowmap",
"expert_routemap_topk","unzipped_token_probs"}) .Outputs({"zipped_tokens",
"zipped_probs_topk"}) .Attrs({"total_zipped_tokens: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_zip));
*/
std::vector<paddle::Tensor> tokens_zip(
    const paddle::Tensor &unzipped_tokens,
    const paddle::Tensor &zipped_expertwise_rowmap,
    const paddle::Tensor &expert_routemap_topk,
    const paddle::Tensor &unzipped_token_probs,
    const int &total_zipped_tokens_num,
    const int &num_experts) {
  PD_CHECK(unzipped_tokens.dtype() == paddle::DataType::BFLOAT16 ||
           unzipped_tokens.dtype() == paddle::DataType::FLOAT32);
  const int rows = unzipped_tokens.shape()[0];       // seqlen
  const int cols = unzipped_tokens.shape()[1];       // 一般为7168
  const int topk = expert_routemap_topk.shape()[1];  // 一般为8


  //------------------------ 输出1张量 ------------------------
  auto zipped_tokens = paddle::empty({total_zipped_tokens_num, cols},
                                     unzipped_tokens.dtype(),
                                     unzipped_tokens.place());
  auto zipped_probs_topk = paddle::empty({total_zipped_tokens_num, topk},
                                         unzipped_token_probs.dtype(),
                                         unzipped_token_probs.place());
  // ----------------------- 0初始化 zipped_probs_topk ------------------
  if (unzipped_token_probs.dtype() == paddle::DataType::FLOAT32) {
    void *zipped_probs_topk_ptr =
        reinterpret_cast<void *>(zipped_probs_topk.data<float>());
    cudaMemsetAsync(zipped_probs_topk_ptr,
                    0,
                    sizeof(float) * total_zipped_tokens_num * topk,
                    unzipped_token_probs.stream());
  } else if (unzipped_token_probs.dtype() == paddle::DataType::BFLOAT16) {
    void *zipped_probs_topk_ptr =
        reinterpret_cast<void *>(zipped_probs_topk.data<phi::bfloat16>());
    cudaMemsetAsync(zipped_probs_topk_ptr,
                    0,
                    sizeof(phi::bfloat16) * total_zipped_tokens_num * topk,
                    unzipped_token_probs.stream());
  }
  if (rows != 0) {
    PD_SWITCH_NUM_EXPERTS(num_experts, ([&] {
                            dispatch_tokens_zip<MAX_NUM_EXPERTS_C>(
                                unzipped_tokens,
                                zipped_expertwise_rowmap,
                                expert_routemap_topk,
                                unzipped_token_probs,
                                zipped_tokens,
                                zipped_probs_topk,
                                total_zipped_tokens_num,
                                num_experts,
                                cols,
                                topk);
                          }));
  }
  return {zipped_tokens, zipped_probs_topk};
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
  int cols = X.shape()[1];  // 一般为7168
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
  // 用于原子记录当前以增广的行数，其上限应为 total_unzipped_tokens_num - rows
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

PD_BUILD_OP(tokens_zip)
    .Inputs({"unzipped_tokens",
             "zipped_expertwise_rowmap",
             "expert_routemap_topk",
             "unzipped_token_probs"})
    .Outputs({"zipped_tokens", "zipped_probs_topk"})
    .Attrs({"total_zipped_tokens: int", "num_experts: int"})
    .SetKernelFn(PD_KERNEL(tokens_zip));

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
