// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include "helper.h"
#include "mem_util.cuh"
#include "mma_tensor_op.cuh"
#include "utils.cuh"

template <typename T, int VecSize = 1>
__global__ void append_decode_cache_T_rope_kernel(
    const T* __restrict__ quant_qkv,  // [bsz, num_heads + 2 * kv_num_heads,
                                      // head_size]
    T* __restrict__ key_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // head_size // 2]
    T* __restrict__ value_cache,  // [num_blocks, kv_num_heads, block_size,
                                  // head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int kv_num_heads) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadKVT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT out_vec;
  LoadKVT cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  // const int64_t offset = 2 * hidden_size;
  const int half_head_size = head_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / hidden_size;
    const int bias = linear_index % hidden_size;
    const int hi = bias / head_size;  // q + k + v
    const int h_bias = bias % head_size;
    const int start_token_idx = ori_bi * max_seq_len - cum_offsets[ori_bi];
    if (seq_lens_encoder[ori_bi] > 0) return;
    const int write_seq_id = seq_lens[ori_bi];
    if (write_seq_id == 0) continue;

    const int* block_table_now = nullptr;

    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    const uint32_t ori_idx =
        start_token_idx * hidden_size + hi * head_size + h_bias;

    const int bias_idx = hi * head_size + h_bias;
    Load<T, VecSize>(&quant_qkv[ori_idx], &src_vec);
    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * half_head_size + h_bias / 2;
      Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      // dequant + add_bias + rope
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);

      if (hi < num_heads + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        out_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        out_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        out_vec[2 * i] = src_vec[2 * i];
        out_vec[2 * i + 1] = src_vec[2 * i + 1];
      }
    }
    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(out_vec, &qkv_out[ori_idx]);
    } else {
      // quant + write k/v
      const uint32_t kv_head_idx = (hi - num_heads) % kv_num_heads;
      const uint32_t tgt_idx =
          block_idx * kv_num_heads * block_size * head_size +
          kv_head_idx * block_size * head_size + block_offset * head_size +
          h_bias;
      if (hi < num_heads + kv_num_heads) {
        Store<T, VecSize>(out_vec, &key_cache[tgt_idx]);
      } else {
        Store<T, VecSize>(out_vec, &value_cache[tgt_idx]);
      }
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void append_decode_cache_T_rope_kernel(
    const int* __restrict__ quant_qkv,  // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    T* __restrict__ key_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // head_size // 2]
    T* __restrict__ value_cache,  // [num_blocks, kv_num_heads, block_size,
                                  // head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const float* __restrict__ qkv_out_scales,  // [num_head + 2 *
                                               // kv_num_heads, dim_head]
    const T* __restrict__ qkv_biases,  // [num_head + 2 * kv_num_heads,
                                       // dim_head]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int kv_num_heads) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadOutScaleT = AlignedVector<float, VecSize>;
  using LoadKVT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadOutScaleT out_scale_vec;
  LoadKVT cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  // const int64_t offset = 2 * hidden_size;
  const int half_head_size = head_size / 2;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / hidden_size;
    const int bias = linear_index % hidden_size;
    const int hi = bias / head_size;  // q + k + v
    const int h_bias = bias % head_size;
    const int start_token_idx = ori_bi * max_seq_len - cum_offsets[ori_bi];
    if (seq_lens_encoder[ori_bi] > 0) return;
    const int write_seq_id = seq_lens[ori_bi];
    if (write_seq_id == 0) continue;

    const int* block_table_now = nullptr;

    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    const uint32_t ori_idx =
        start_token_idx * hidden_size + hi * head_size + h_bias;

    const int bias_idx = hi * head_size + h_bias;
    Load<int, VecSize>(&quant_qkv[ori_idx], &src_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    }
    Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * half_head_size + h_bias / 2;
      Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      // dequant + add_bias + rope
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
                                    static_cast<float>(bias_vec[2 * i])
                              : input_left * out_scale_vec[2 * i];
      input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
                                     static_cast<float>(bias_vec[2 * i + 1])
                               : input_right * out_scale_vec[2 * i + 1];
      if (hi < num_heads + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(bias_vec, &qkv_out[ori_idx]);
    } else {
      // quant + write k/v
      const uint32_t kv_head_idx = (hi - num_heads) % kv_num_heads;
      const uint32_t tgt_idx =
          block_idx * kv_num_heads * block_size * head_size +
          kv_head_idx * block_size * head_size + block_offset * head_size +
          h_bias;
      if (hi < num_heads + kv_num_heads) {
        Store<T, VecSize>(bias_vec, &key_cache[tgt_idx]);
      } else {
        Store<T, VecSize>(bias_vec, &value_cache[tgt_idx]);
      }
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void append_decode_cache_T_neox_rope_kernel(
    const T* __restrict__ qkv,    // [bsz, num_heads + 2 * kv_num_heads,
                                  // head_size]
    T* __restrict__ key_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // head_size // 2]
    T* __restrict__ value_cache,  // [num_blocks, kv_num_heads, block_size,
                                  // head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int kv_num_heads) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadKVT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, VecSize>;

  LoadT left_vec, right_vec;
  LoadBiasT left_bias_vec, right_bias_vec;
  LoadKVT left_cache_vec, right_cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_head_size = head_size / 2;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  const int64_t half_hidden_size = hidden_size / 2;
  // const int64_t offset = 2 * hidden_size;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / half_hidden_size;
    const int bias = linear_index % half_hidden_size;
    const int hi = bias / half_head_size;  // q + k + v
    const int h_bias = bias % half_head_size;
    const int start_token_idx = ori_bi * max_seq_len - cum_offsets[ori_bi];
    if (seq_lens_encoder[ori_bi] > 0) return;
    const int write_seq_id = seq_lens[ori_bi];
    if (write_seq_id == 0) continue;

    const int* block_table_now = nullptr;

    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    const uint32_t ori_idx_left =
        start_token_idx * hidden_size + hi * head_size + h_bias;
    const uint32_t ori_idx_right = ori_idx_left + half_head_size;

    Load<T, VecSize>(&qkv[ori_idx_left], &left_vec);
    Load<T, VecSize>(&qkv[ori_idx_right], &right_vec);

    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * head_size + h_bias;
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      // rope
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      if (hi < num_heads + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }
    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(left_bias_vec, &qkv_out[ori_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv_out[ori_idx_right]);
    } else {
      // write k/v
      const uint32_t kv_head_idx = (hi - num_heads) % kv_num_heads;
      const uint32_t tgt_idx_left =
          block_idx * kv_num_heads * block_size * head_size +
          kv_head_idx * block_size * head_size + block_offset * head_size +
          h_bias;
      const uint32_t tgt_idx_right = tgt_idx_left + half_head_size;
      if (hi < num_heads + kv_num_heads) {
        Store<T, VecSize>(left_bias_vec, &key_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &key_cache[tgt_idx_right]);
      } else {
        Store<T, VecSize>(left_bias_vec, &value_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &value_cache[tgt_idx_right]);
      }
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void append_decode_cache_T_neox_rope_kernel(
    const int* __restrict__ quant_qkv,  // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    T* __restrict__ key_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // head_size // 2]
    T* __restrict__ value_cache,  // [num_blocks, kv_num_heads, block_size,
                                  // head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const float* __restrict__ qkv_out_scales,  // [num_head + 2 *
                                               // kv_num_heads, dim_head]
    const T* __restrict__ qkv_biases,  // [num_head + 2 * kv_num_heads,
                                       // dim_head]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const uint32_t elem_cnt,
    const int kv_num_heads) {
  using LoadT = AlignedVector<int, VecSize>;
  using LoadBiasT = AlignedVector<T, VecSize>;
  using LoadOutScaleT = AlignedVector<float, VecSize>;
  using LoadKVT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT left_vec, right_vec;
  LoadBiasT left_bias_vec, right_bias_vec;
  LoadOutScaleT left_out_scale_vec, right_out_scale_vec;
  LoadKVT left_cache_vec, right_cache_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_head_size = head_size / 2;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * head_size;
  const int64_t half_hidden_size = hidden_size / 2;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / half_hidden_size;
    const int bias = linear_index % half_hidden_size;
    const int hi = bias / half_head_size;  // q + k + v
    const int h_bias = bias % half_head_size;
    const int start_token_idx = ori_bi * max_seq_len - cum_offsets[ori_bi];
    if (seq_lens_encoder[ori_bi] > 0) return;
    const int write_seq_id = seq_lens[ori_bi];
    if (write_seq_id == 0) continue;

    const int* block_table_now = nullptr;

    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;
    const uint32_t ori_idx_left =
        start_token_idx * hidden_size + hi * head_size + h_bias;
    const uint32_t ori_idx_right = ori_idx_left + half_head_size;

    const int bias_idx_left = hi * head_size + h_bias;
    const int bias_idx_right = bias_idx_left + half_head_size;

    Load<int, VecSize>(&quant_qkv[ori_idx_left], &left_vec);
    Load<int, VecSize>(&quant_qkv[ori_idx_right], &right_vec);
    if (qkv_biases) {
      Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
      Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
    }

    Load<float, VecSize>(&qkv_out_scales[bias_idx_left], &left_out_scale_vec);
    Load<float, VecSize>(&qkv_out_scales[bias_idx_right], &right_out_scale_vec);

    if (hi < num_heads + kv_num_heads) {
      // q k rope
      const uint32_t emb_idx = write_seq_id * head_size + h_bias;
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      // dequant + add_bias + rope
      float input_left = static_cast<float>(left_vec[i]);
      float input_right = static_cast<float>(right_vec[i]);
      input_left = qkv_biases ? input_left * left_out_scale_vec[i] +
                                    static_cast<float>(left_bias_vec[i])
                              : input_left * left_out_scale_vec[i];
      input_right = qkv_biases ? input_right * right_out_scale_vec[i] +
                                     static_cast<float>(right_bias_vec[i])
                               : input_right * right_out_scale_vec[i];
      if (hi < num_heads + kv_num_heads) {
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        left_bias_vec[i] = static_cast<T>(input_left);
        right_bias_vec[i] = static_cast<T>(input_right);
      }
    }
    if (hi < num_heads) {
      // write q
      Store<T, VecSize>(left_bias_vec, &qkv_out[ori_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv_out[ori_idx_right]);
    } else {
      // quant + write k/v
      const uint32_t kv_head_idx = (hi - num_heads) % kv_num_heads;
      const uint32_t tgt_idx_left =
          block_idx * kv_num_heads * block_size * head_size +
          kv_head_idx * block_size * head_size + block_offset * head_size +
          h_bias;
      const uint32_t tgt_idx_right = tgt_idx_left + half_head_size;
      if (hi < num_heads + kv_num_heads) {
        Store<T, VecSize>(left_bias_vec, &key_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &key_cache[tgt_idx_right]);
      } else {
        Store<T, VecSize>(left_bias_vec, &value_cache[tgt_idx_left]);
        Store<T, VecSize>(right_bias_vec, &value_cache[tgt_idx_right]);
      }
    }
  }
}

template <typename T, int VecSize = 4, int RoundType = 0, int HeadDim = 128>
__global__ void append_decode_cache_int8_rope_kernel(
    const T* __restrict__ quant_qkv,    // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    uint8_t* __restrict__ key_cache,    // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    uint8_t* __restrict__ value_cache,  // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const T* __restrict__ cache_k_scale,
    const T* __restrict__ cache_v_scale,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int block_size,
    const float max_bound,
    const float min_bound,
    const int kv_num_heads) {
  static_assert(HeadDim == 128, "just support HeadDim be 128 now!");
  static_assert(VecSize == 4, "just support VecSize be 4 now, 32 * 4!");
  constexpr int NUM_WARPS = 4;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane_id = tid % 32;
  const int bid = blockIdx.x, head_idx = blockIdx.y * NUM_WARPS + wid;
  int q_head_idx, k_head_idx, v_idx;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * HeadDim;
  constexpr int half_head_size = HeadDim / 2;
  const int start_token_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (seq_lens_encoder[bid] > 0) return;
  const int write_seq_id = seq_lens[bid];
  if (write_seq_id == 0) return;
  const int* block_table_now = nullptr;

  block_table_now = block_tables + bid * max_blocks_per_seq;
  const int block_idx = __ldg(&block_table_now[write_seq_id / block_size]);
  const int block_offset = write_seq_id % block_size;

  if (head_idx < num_heads) {
    // q
    using LoadT = AlignedVector<T, VecSize>;
    using LoadBiasT = AlignedVector<T, VecSize>;
    using LoadOutScaleT = AlignedVector<float, VecSize>;
    constexpr int HalfVecSize = VecSize / 2;
    using LoadEmbT = AlignedVector<float, HalfVecSize>;

    LoadT src_vec;
    LoadBiasT out_vec;
    LoadEmbT cos_emb_vec;
    LoadEmbT sin_emb_vec;
    const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
    T* qkv_out_now = qkv_out + start_token_idx * hidden_size;
#pragma unroll
    for (uint32_t head_bias = lane_id * VecSize; head_bias < HeadDim;
         head_bias += 32 * VecSize) {
      const int bias_idx = head_idx * HeadDim + head_bias;
      Load<T, VecSize>(&qkv_now[bias_idx], &src_vec);

      // q rope
      const uint32_t emb_idx = write_seq_id * half_head_size + head_bias / 2;
      Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
      for (int i = 0; i < HalfVecSize; i++) {
        // dequant + add_bias + rope
        float input_left = static_cast<float>(src_vec[2 * i]);
        float input_right = static_cast<float>(src_vec[2 * i + 1]);

        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        out_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        out_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
      Store<T, VecSize>(out_vec, &qkv_out_now[bias_idx]);
    }
  } else if (head_idx < num_heads + 2 * kv_num_heads) {
    // k
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    const uint32_t kv_head_idx = (head_idx - num_heads) % kv_num_heads;
    if (block_offset == 0) {
      // pad zero for this kv_head_idx for this block
      LoadPadKVT pad_cache_vec;
      *(reinterpret_cast<uint4*>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
      if (head_idx < num_heads + kv_num_heads) {
        constexpr int num_vecs_per_head_dim = HeadDim / KV_VEC_SIZE;
        constexpr int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx =
            (block_idx * kv_num_heads + kv_head_idx) * block_size * HeadDim +
            lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim;
             block_i < block_size;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                      &key_cache[tgt_idx + block_i * HeadDim]);
        }
      } else {
        const int num_vecs_per_head_dim = block_size / KV_VEC_SIZE;
        const int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx =
            (block_idx * kv_num_heads + kv_head_idx) * HeadDim * block_size +
            lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim; block_i < HeadDim;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &value_cache[tgt_idx + block_i * block_size]);
        }
      }
    }

    constexpr int K_VEC_SIZE = 4;
    constexpr int HALF_K_VEC_SIZE = 2;
    using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
    using LoadKVT = AlignedVector<uint8_t, HALF_K_VEC_SIZE>;
    using LoadT = AlignedVector<T, HALF_K_VEC_SIZE>;
    using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
    using LoadOutScaleT = AlignedVector<float, HALF_K_VEC_SIZE>;
    using LoadEmbT = AlignedVector<float, 1>;
    LoadKVResT cache_vec;
    LoadT src_vec1, src_vec2;
    LoadBiasT out_vec1, out_vec2;
    LoadEmbT cos_emb_vec1, cos_emb_vec2;
    LoadEmbT sin_emb_vec1, sin_emb_vec2;

    const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
    const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
    const int bias_idx = head_idx * HeadDim + head_bias;
    Load<T, HALF_K_VEC_SIZE>(&qkv_now[bias_idx], &src_vec1);
    Load<T, HALF_K_VEC_SIZE>(&qkv_now[bias_idx + 8], &src_vec2);
    T scale;
    if (head_idx < num_heads + kv_num_heads) {
      const uint32_t emb_idx = write_seq_id * half_head_size + head_bias / 2;
      Load<float, 1>(&cos_emb[emb_idx], &cos_emb_vec1);
      Load<float, 1>(&cos_emb[emb_idx + 4], &cos_emb_vec2);
      Load<float, 1>(&sin_emb[emb_idx], &sin_emb_vec1);
      Load<float, 1>(&sin_emb[emb_idx + 4], &sin_emb_vec2);
      scale = __ldg(&cache_k_scale[kv_head_idx]);
    } else {
      scale = __ldg(&cache_v_scale[kv_head_idx]);
    }

    float input_left = static_cast<float>(src_vec1[0]);
    float input_right = static_cast<float>(src_vec1[1]);
    if (head_idx < num_heads + kv_num_heads) {
      float cos_tmp = cos_emb_vec1[0];
      float sin_tmp = sin_emb_vec1[0];
      out_vec1[0] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      out_vec1[1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    } else {
      out_vec1[0] = src_vec1[0];
      out_vec1[1] = src_vec1[1];
    }

    input_left = static_cast<float>(src_vec2[0]);
    input_right = static_cast<float>(src_vec2[1]);
    if (head_idx < num_heads + kv_num_heads) {
      float cos_tmp = cos_emb_vec2[0];
      float sin_tmp = sin_emb_vec2[0];
      out_vec2[0] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      out_vec2[1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    } else {
      out_vec2[0] = src_vec2[0];
      out_vec2[1] = src_vec2[1];
    }
#pragma unroll
    for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
      float quant_value1 = static_cast<float>(scale * out_vec1[i]);
      float quant_value2 = static_cast<float>(scale * out_vec2[i]);
      if constexpr (RoundType == 0) {
        quant_value1 = static_cast<float>(roundWithTiesToEven(quant_value1));
        quant_value2 = static_cast<float>(roundWithTiesToEven(quant_value2));
      } else {
        quant_value1 = static_cast<float>(round(quant_value1));
        quant_value2 = static_cast<float>(round(quant_value2));
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      cache_vec[i] = static_cast<uint8_t>(quant_value1 + 128.0f);
      cache_vec[i + HALF_K_VEC_SIZE] =
          static_cast<uint8_t>(quant_value2 + 128.0f);
    }
    if (head_idx < num_heads + kv_num_heads) {
      const int start_block_16 =
          block_offset / 16 * 16 + block_offset % 8 + lane_id / 4 % 2 * 8;
      const uint32_t tgt_cache_idx =
          block_idx * kv_num_heads * block_size * HeadDim +
          kv_head_idx * block_size * HeadDim + start_block_16 * HeadDim +
          lane_id / 4 / 2 * 32 + (block_offset % 16) / 8 * 16 + lane_id % 4 * 4;
      Store<uint8_t, K_VEC_SIZE>(cache_vec, &key_cache[tgt_cache_idx]);
    } else {
      const uint32_t base_tgt_cache_idx =
          block_idx * kv_num_heads * HeadDim * block_size +
          kv_head_idx * HeadDim * block_size +
          (lane_id / 4 * 16 + lane_id % 4 * 2) * block_size +
          block_offset / 16 % 2 * 8 * block_size + block_offset / 16 / 2 * 32;
      const uint32_t tgt_cache_idx1 = base_tgt_cache_idx +
                                      block_offset % 8 / 2 * 4     // per 4
                                      + block_offset % 16 / 8 * 2  // per 2
                                      + block_offset % 2;          // per 1
      const uint32_t tgt_cache_idx2 = tgt_cache_idx1 + block_size;
      const uint32_t tgt_cache_idx3 = tgt_cache_idx1 + 16;
      const uint32_t tgt_cache_idx4 = tgt_cache_idx3 + block_size;
      value_cache[tgt_cache_idx1] = cache_vec[0];
      value_cache[tgt_cache_idx2] = cache_vec[1];
      value_cache[tgt_cache_idx3] = cache_vec[2];
      value_cache[tgt_cache_idx4] = cache_vec[3];
    }
  }
}

template <typename T, int VecSize = 4, int RoundType = 0, int HeadDim = 128>
__global__ void append_decode_cache_int8_rope_kernel(
    const int* __restrict__ quant_qkv,  // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    uint8_t* __restrict__ key_cache,    // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    uint8_t* __restrict__ value_cache,  // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const float* __restrict__ qkv_out_scales,  // [num_head + 2 *
                                               // kv_num_heads, dim_head]
    const T* __restrict__ qkv_biases,  // [num_head + 2 * kv_num_heads,
                                       // dim_head]
    const T* __restrict__ cache_k_scales,
    const T* __restrict__ cache_v_scales,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int block_size,
    const float max_bound,
    const float min_bound,
    const int kv_num_heads) {
  static_assert(HeadDim == 128, "just support HeadDim be 128 now!");
  static_assert(VecSize == 4, "just support VecSize be 4 now, 32 * 4!");
  constexpr int NUM_WARPS = 4;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane_id = tid % 32;
  const int by = blockIdx.y;
  const int bid = blockIdx.x, head_idx = blockIdx.y * NUM_WARPS + wid;
  int q_head_idx, k_head_idx, v_idx;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * HeadDim;
  constexpr int half_head_size = HeadDim / 2;
  const int start_token_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (seq_lens_encoder[bid] > 0) return;
  const int write_seq_id = seq_lens[bid];
  if (write_seq_id == 0) return;
  const int* block_table_now = nullptr;

  block_table_now = block_tables + bid * max_blocks_per_seq;
  const int block_idx = __ldg(&block_table_now[write_seq_id / block_size]);
  const int block_offset = write_seq_id % block_size;

  if (head_idx < num_heads) {
    // q
    using LoadT = AlignedVector<int, VecSize>;
    using LoadBiasT = AlignedVector<T, VecSize>;
    using LoadOutScaleT = AlignedVector<float, VecSize>;
    constexpr int HalfVecSize = VecSize / 2;
    using LoadEmbT = AlignedVector<float, HalfVecSize>;

    LoadT src_vec;
    LoadBiasT bias_vec;
    LoadOutScaleT out_scale_vec;
    LoadEmbT cos_emb_vec;
    LoadEmbT sin_emb_vec;
    const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
    T* qkv_out_now = qkv_out + start_token_idx * hidden_size;
#pragma unroll
    for (uint32_t head_bias = lane_id * VecSize; head_bias < HeadDim;
         head_bias += 32 * VecSize) {
      const int bias_idx = head_idx * HeadDim + head_bias;
      Load<int, VecSize>(&qkv_now[bias_idx], &src_vec);

      if (qkv_biases) {
        Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
      }
      Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);

      // q rope
      const uint32_t emb_idx = write_seq_id * half_head_size + head_bias / 2;
      Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);

      Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);

#pragma unroll
      for (int i = 0; i < HalfVecSize; i++) {
        // dequant + add_bias + rope
        float input_left = static_cast<float>(src_vec[2 * i]);
        float input_right = static_cast<float>(src_vec[2 * i + 1]);
        input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
                                      static_cast<float>(bias_vec[2 * i])
                                : input_left * out_scale_vec[2 * i];
        input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
                                       static_cast<float>(bias_vec[2 * i + 1])
                                 : input_right * out_scale_vec[2 * i + 1];
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
      Store<T, VecSize>(bias_vec, &qkv_out_now[bias_idx]);
    }
  } else if (head_idx < num_heads + 2 * kv_num_heads) {
    // k
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    const uint32_t kv_head_idx = (head_idx - num_heads) % kv_num_heads;

    if (block_offset == 0) {
      // pad zero for this kv_head_idx for this block
      LoadPadKVT pad_cache_vec;
      *(reinterpret_cast<uint4*>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
      if (head_idx < num_heads + kv_num_heads) {
        constexpr int num_vecs_per_head_dim = HeadDim / KV_VEC_SIZE;
        constexpr int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx =
            (block_idx * kv_num_heads + kv_head_idx) * block_size * HeadDim +
            lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim;
             block_i < block_size;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                      &key_cache[tgt_idx + block_i * HeadDim]);
        }
      } else {
        const int num_vecs_per_head_dim = block_size / KV_VEC_SIZE;
        const int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx =
            (block_idx * kv_num_heads + kv_head_idx) * HeadDim * block_size +
            lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim; block_i < HeadDim;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &value_cache[tgt_idx + block_i * block_size]);
        }
      }
    }

    constexpr int K_VEC_SIZE = 4;
    constexpr int HALF_K_VEC_SIZE = 2;
    using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
    using LoadKVT = AlignedVector<uint8_t, HALF_K_VEC_SIZE>;
    using LoadT = AlignedVector<int, HALF_K_VEC_SIZE>;
    using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
    using LoadOutScaleT = AlignedVector<float, HALF_K_VEC_SIZE>;
    using LoadEmbT = AlignedVector<float, 1>;
    LoadKVResT cache_vec;
    LoadT src_vec1, src_vec2;
    LoadBiasT bias_vec1, bias_vec2;
    LoadOutScaleT out_scale_vec1, out_scale_vec2;
    LoadEmbT cos_emb_vec1, cos_emb_vec2;
    LoadEmbT sin_emb_vec1, sin_emb_vec2;

    const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
    const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
    const int bias_idx = head_idx * HeadDim + head_bias;
    Load<int, HALF_K_VEC_SIZE>(&qkv_now[bias_idx], &src_vec1);
    Load<int, HALF_K_VEC_SIZE>(&qkv_now[bias_idx + 8], &src_vec2);
    if (qkv_biases) {
      Load<T, HALF_K_VEC_SIZE>(&qkv_biases[bias_idx], &bias_vec1);
      Load<T, HALF_K_VEC_SIZE>(&qkv_biases[bias_idx + 8], &bias_vec2);
    }
    Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[bias_idx], &out_scale_vec1);
    Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[bias_idx + 8],
                                 &out_scale_vec2);

    T scale;
    if (head_idx < num_heads + kv_num_heads) {
      const uint32_t emb_idx = write_seq_id * half_head_size + head_bias / 2;
      Load<float, 1>(&cos_emb[emb_idx], &cos_emb_vec1);
      Load<float, 1>(&cos_emb[emb_idx + 4], &cos_emb_vec2);
      Load<float, 1>(&sin_emb[emb_idx], &sin_emb_vec1);
      Load<float, 1>(&sin_emb[emb_idx + 4], &sin_emb_vec2);
      scale = __ldg(&cache_k_scales[kv_head_idx]);
    } else {
      scale = __ldg(&cache_v_scales[kv_head_idx]);
    }

    float input_left = static_cast<float>(src_vec1[0]);
    float input_right = static_cast<float>(src_vec1[1]);
    input_left = qkv_biases ? input_left * out_scale_vec1[0] +
                                  static_cast<float>(bias_vec1[0])
                            : input_left * out_scale_vec1[0];
    input_right = qkv_biases ? input_right * out_scale_vec1[1] +
                                   static_cast<float>(bias_vec1[1])
                             : input_right * out_scale_vec1[1];
    if (head_idx < num_heads + kv_num_heads) {
      float cos_tmp = cos_emb_vec1[0];
      float sin_tmp = sin_emb_vec1[0];
      bias_vec1[0] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      bias_vec1[1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    } else {
      bias_vec1[0] = static_cast<T>(input_left);
      bias_vec1[1] = static_cast<T>(input_right);
    }

    input_left = static_cast<float>(src_vec2[0]);
    input_right = static_cast<float>(src_vec2[1]);
    input_left = qkv_biases ? input_left * out_scale_vec2[0] +
                                  static_cast<float>(bias_vec2[0])
                            : input_left * out_scale_vec2[0];
    input_right = qkv_biases ? input_right * out_scale_vec2[1] +
                                   static_cast<float>(bias_vec2[1])
                             : input_right * out_scale_vec2[1];
    if (head_idx < num_heads + kv_num_heads) {
      float cos_tmp = cos_emb_vec2[0];
      float sin_tmp = sin_emb_vec2[0];
      bias_vec2[0] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      bias_vec2[1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    } else {
      bias_vec2[0] = static_cast<T>(input_left);
      bias_vec2[1] = static_cast<T>(input_right);
    }
#pragma unroll
    for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
      float quant_value1 = static_cast<float>(scale * bias_vec1[i]);
      float quant_value2 = static_cast<float>(scale * bias_vec2[i]);
      if constexpr (RoundType == 0) {
        quant_value1 = static_cast<float>(roundWithTiesToEven(quant_value1));
        quant_value2 = static_cast<float>(roundWithTiesToEven(quant_value2));
      } else {
        quant_value1 = static_cast<float>(round(quant_value1));
        quant_value2 = static_cast<float>(round(quant_value2));
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      cache_vec[i] = static_cast<uint8_t>(quant_value1 + 128.0f);
      cache_vec[i + HALF_K_VEC_SIZE] =
          static_cast<uint8_t>(quant_value2 + 128.0f);
    }
    if (head_idx < num_heads + kv_num_heads) {
      const int start_block_16 =
          block_offset / 16 * 16 + block_offset % 8 + lane_id / 4 % 2 * 8;
      const uint32_t tgt_cache_idx =
          block_idx * kv_num_heads * block_size * HeadDim +
          kv_head_idx * block_size * HeadDim + start_block_16 * HeadDim +
          lane_id / 4 / 2 * 32 + (block_offset % 16) / 8 * 16 + lane_id % 4 * 4;
      Store<uint8_t, K_VEC_SIZE>(cache_vec, &key_cache[tgt_cache_idx]);
    } else {
      const uint32_t base_tgt_cache_idx =
          block_idx * kv_num_heads * HeadDim * block_size +
          kv_head_idx * HeadDim * block_size +
          (lane_id / 4 * 16 + lane_id % 4 * 2) * block_size +
          block_offset / 16 % 2 * 8 * block_size + block_offset / 16 / 2 * 32;
      const uint32_t tgt_cache_idx1 = base_tgt_cache_idx +
                                      block_offset % 8 / 2 * 4     // per 4
                                      + block_offset % 16 / 8 * 2  // per 2
                                      + block_offset % 2;          // per 1
      const uint32_t tgt_cache_idx2 = tgt_cache_idx1 + block_size;
      const uint32_t tgt_cache_idx3 = tgt_cache_idx1 + 16;
      const uint32_t tgt_cache_idx4 = tgt_cache_idx3 + block_size;
      value_cache[tgt_cache_idx1] = cache_vec[0];
      value_cache[tgt_cache_idx2] = cache_vec[1];
      value_cache[tgt_cache_idx3] = cache_vec[2];
      value_cache[tgt_cache_idx4] = cache_vec[3];
    }
  }
}


template <typename T, int VecSize = 4, int RoundType = 0, int HeadDim = 128>
__global__ void append_decode_cache_int8_neox_rope_kernel(
    const T* __restrict__ quant_qkv,    // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    uint8_t* __restrict__ key_cache,    // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    uint8_t* __restrict__ value_cache,  // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const T* __restrict__ cache_k_scales,
    const T* __restrict__ cache_v_scales,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int block_size,
    const float max_bound,
    const float min_bound,
    const int kv_num_heads) {
  static_assert(HeadDim == 128, "just support HeadDim be 128 now!");
  static_assert(VecSize == 4, "just support VecSize be 4 now, 32 * 4!");
  constexpr int NUM_WARPS = 4;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane_id = tid % 32;
  const int by = blockIdx.y;
  const int bid = blockIdx.x, head_idx = blockIdx.y * NUM_WARPS + wid;
  int q_head_idx, k_head_idx, v_idx;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * HeadDim;
  constexpr int half_head_size = HeadDim / 2;
  const int start_token_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (seq_lens_encoder[bid] > 0) return;
  const int write_seq_id = seq_lens[bid];
  if (write_seq_id == 0) return;
  const int* block_table_now = nullptr;

  block_table_now = block_tables + bid * max_blocks_per_seq;
  const int block_idx = __ldg(&block_table_now[write_seq_id / block_size]);
  const int block_offset = write_seq_id % block_size;

  if (head_idx < num_heads) {
    // q
    using LoadT = AlignedVector<T, VecSize>;
    using LoadBiasT = AlignedVector<T, VecSize>;
    constexpr int HalfVecSize = VecSize / 2;
    using LoadEmbT = AlignedVector<float, VecSize>;

    LoadT left_vec;
    LoadT right_vec;
    LoadBiasT left_bias_vec;
    LoadBiasT right_bias_vec;
    LoadEmbT cos_emb_vec;
    LoadEmbT sin_emb_vec;
    const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
    T* qkv_out_now = qkv_out + start_token_idx * hidden_size;
#pragma unroll
    for (uint32_t head_bias = lane_id * VecSize; head_bias < half_head_size;
         head_bias += 32 * VecSize) {
      const int bias_idx_left = head_idx * HeadDim + head_bias;
      const int bias_idx_right = bias_idx_left + half_head_size;

      Load<T, VecSize>(&qkv_now[bias_idx_left], &left_vec);
      Load<T, VecSize>(&qkv_now[bias_idx_right], &right_vec);

      // q rope
      const uint32_t emb_idx = write_seq_id * HeadDim + head_bias;
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);

#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        // dequant + add_bias + rope
        float input_left = static_cast<float>(left_vec[i]);
        float input_right = static_cast<float>(right_vec[i]);
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
      Store<T, VecSize>(left_bias_vec, &qkv_out_now[bias_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv_out_now[bias_idx_right]);
    }
  } else if (head_idx < num_heads + 2 * kv_num_heads) {
    // k v
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    const uint32_t kv_head_idx = (head_idx - num_heads) % kv_num_heads;
    if (block_offset == 0) {
      // pad zero for this kv_head_idx for this block
      LoadPadKVT pad_cache_vec;
      *(reinterpret_cast<uint4*>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
      if (head_idx < num_heads + kv_num_heads) {
        constexpr int num_vecs_per_head_dim = HeadDim / KV_VEC_SIZE;
        constexpr int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx =
            (block_idx * kv_num_heads + kv_head_idx) * block_size * HeadDim +
            lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim;
             block_i < block_size;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                      &key_cache[tgt_idx + block_i * HeadDim]);
        }
      } else {
        const int num_vecs_per_head_dim = block_size / KV_VEC_SIZE;
        const int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx =
            (block_idx * kv_num_heads + kv_head_idx) * HeadDim * block_size +
            lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim; block_i < HeadDim;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &value_cache[tgt_idx + block_i * block_size]);
        }
      }
    }
    if (head_idx < num_heads + kv_num_heads) {
      // k
      const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
      if (head_bias < half_head_size) {
        constexpr int K_VEC_SIZE = 4;
        constexpr int HALF_K_VEC_SIZE = 2;
        using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
        using LoadKVT = AlignedVector<uint8_t, HALF_K_VEC_SIZE>;
        using LoadT = AlignedVector<T, HALF_K_VEC_SIZE>;
        using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
        using LoadEmbT = AlignedVector<float, HALF_K_VEC_SIZE>;

        LoadKVResT left_cache_vec, right_cache_vec;
        LoadT left_src_vec1, left_src_vec2, right_src_vec1, right_src_vec2;
        LoadBiasT left_bias_vec1, left_bias_vec2, right_bias_vec1,
            right_bias_vec2;
        LoadEmbT cos_emb_vec1, cos_emb_vec2;
        LoadEmbT sin_emb_vec1, sin_emb_vec2;

        const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
        const int left_bias_idx = head_idx * HeadDim + head_bias;
        const int right_bias_idx = left_bias_idx + half_head_size;

        Load<T, HALF_K_VEC_SIZE>(&qkv_now[left_bias_idx], &left_src_vec1);
        Load<T, HALF_K_VEC_SIZE>(&qkv_now[left_bias_idx + 8], &left_src_vec2);
        Load<T, HALF_K_VEC_SIZE>(&qkv_now[right_bias_idx], &right_src_vec1);
        Load<T, HALF_K_VEC_SIZE>(&qkv_now[right_bias_idx + 8], &right_src_vec2);

        T scale;
        const uint32_t emb_idx = write_seq_id * HeadDim + head_bias;
        Load<float, HALF_K_VEC_SIZE>(&cos_emb[emb_idx], &cos_emb_vec1);
        Load<float, HALF_K_VEC_SIZE>(&cos_emb[emb_idx + 8], &cos_emb_vec2);
        Load<float, HALF_K_VEC_SIZE>(&sin_emb[emb_idx], &sin_emb_vec1);
        Load<float, HALF_K_VEC_SIZE>(&sin_emb[emb_idx + 8], &sin_emb_vec2);
        scale = __ldg(&cache_k_scales[kv_head_idx]);
#pragma unroll
        for (int i = 0; i < HALF_K_VEC_SIZE; i++) {
          float input_left = static_cast<float>(left_src_vec1[i]);
          float input_right = static_cast<float>(right_src_vec1[i]);

          float cos_tmp = cos_emb_vec1[i];
          float sin_tmp = sin_emb_vec1[i];
          left_bias_vec1[i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          right_bias_vec1[i] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);

          input_left = static_cast<float>(left_src_vec2[i]);
          input_right = static_cast<float>(right_src_vec2[i]);
          cos_tmp = cos_emb_vec2[i];
          sin_tmp = sin_emb_vec2[i];
          left_bias_vec2[i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          right_bias_vec2[i] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);

          float quant_value1 = static_cast<float>(scale * left_bias_vec1[i]);
          float quant_value2 = static_cast<float>(scale * left_bias_vec2[i]);
          if constexpr (RoundType == 0) {
            quant_value1 =
                static_cast<float>(roundWithTiesToEven(quant_value1));
            quant_value2 =
                static_cast<float>(roundWithTiesToEven(quant_value2));
          } else {
            quant_value1 = static_cast<float>(round(quant_value1));
            quant_value2 = static_cast<float>(round(quant_value2));
          }
          quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
          quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
          quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
          quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
          left_cache_vec[i] = static_cast<uint8_t>(quant_value1 + 128.0f);
          left_cache_vec[i + HALF_K_VEC_SIZE] =
              static_cast<uint8_t>(quant_value2 + 128.0f);

          quant_value1 = static_cast<float>(scale * right_bias_vec1[i]);
          quant_value2 = static_cast<float>(scale * right_bias_vec2[i]);
          if constexpr (RoundType == 0) {
            quant_value1 =
                static_cast<float>(roundWithTiesToEven(quant_value1));
            quant_value2 =
                static_cast<float>(roundWithTiesToEven(quant_value2));
          } else {
            quant_value1 = static_cast<float>(round(quant_value1));
            quant_value2 = static_cast<float>(round(quant_value2));
          }
          quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
          quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
          quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
          quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
          right_cache_vec[i] = static_cast<uint8_t>(quant_value1 + 128.0f);
          right_cache_vec[i + HALF_K_VEC_SIZE] =
              static_cast<uint8_t>(quant_value2 + 128.0f);
        }
        const int left_start_block_16 =
            block_offset / 16 * 16 + block_offset % 8 + lane_id / 4 % 2 * 8;
        const uint32_t left_tgt_cache_idx =
            block_idx * kv_num_heads * block_size * HeadDim +
            kv_head_idx * block_size * HeadDim + left_start_block_16 * HeadDim +
            lane_id / 4 / 2 * 32 + (block_offset % 16) / 8 * 16 +
            lane_id % 4 * 4;

        const int right_lane_id = lane_id + 16;
        const int right_start_block_16 = block_offset / 16 * 16 +
                                         block_offset % 8 +
                                         right_lane_id / 4 % 2 * 8;
        const uint32_t right_tgt_cache_idx =
            block_idx * kv_num_heads * block_size * HeadDim +
            kv_head_idx * block_size * HeadDim +
            right_start_block_16 * HeadDim + right_lane_id / 4 / 2 * 32 +
            (block_offset % 16) / 8 * 16 + right_lane_id % 4 * 4;
        Store<uint8_t, K_VEC_SIZE>(left_cache_vec,
                                   &key_cache[left_tgt_cache_idx]);
        Store<uint8_t, K_VEC_SIZE>(right_cache_vec,
                                   &key_cache[right_tgt_cache_idx]);
      }
    } else {
      // v
      constexpr int K_VEC_SIZE = 4;
      constexpr int HALF_K_VEC_SIZE = 2;
      using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
      using LoadKVT = AlignedVector<uint8_t, HALF_K_VEC_SIZE>;
      using LoadT = AlignedVector<T, HALF_K_VEC_SIZE>;
      using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
      LoadKVResT cache_vec;
      LoadT src_vec1, src_vec2;
      LoadBiasT bias_vec1, bias_vec2;

      const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
      const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
      const int bias_idx = head_idx * HeadDim + head_bias;
      Load<T, HALF_K_VEC_SIZE>(&qkv_now[bias_idx], &src_vec1);
      Load<T, HALF_K_VEC_SIZE>(&qkv_now[bias_idx + 8], &src_vec2);

      T scale = __ldg(&cache_v_scales[kv_head_idx]);

#pragma unroll
      for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
        float quant_value1 = static_cast<float>(scale * src_vec1[i]);
        float quant_value2 = static_cast<float>(scale * src_vec2[i]);
        if constexpr (RoundType == 0) {
          quant_value1 = static_cast<float>(roundWithTiesToEven(quant_value1));
          quant_value2 = static_cast<float>(roundWithTiesToEven(quant_value2));
        } else {
          quant_value1 = static_cast<float>(round(quant_value1));
          quant_value2 = static_cast<float>(round(quant_value2));
        }
        quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
        quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
        quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
        quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
        cache_vec[i] = static_cast<uint8_t>(quant_value1 + 128.0f);
        cache_vec[i + HALF_K_VEC_SIZE] =
            static_cast<uint8_t>(quant_value2 + 128.0f);
      }
      const uint32_t base_tgt_cache_idx =
          block_idx * kv_num_heads * HeadDim * block_size +
          kv_head_idx * HeadDim * block_size +
          (lane_id / 4 * 16 + lane_id % 4 * 2) * block_size +
          block_offset / 16 % 2 * 8 * block_size + block_offset / 16 / 2 * 32;
      const uint32_t tgt_cache_idx1 = base_tgt_cache_idx +
                                      block_offset % 8 / 2 * 4     // per 4
                                      + block_offset % 16 / 8 * 2  // per 2
                                      + block_offset % 2;          // per 1
      const uint32_t tgt_cache_idx2 = tgt_cache_idx1 + block_size;
      const uint32_t tgt_cache_idx3 = tgt_cache_idx1 + 16;
      const uint32_t tgt_cache_idx4 = tgt_cache_idx3 + block_size;
      value_cache[tgt_cache_idx1] = cache_vec[0];
      value_cache[tgt_cache_idx2] = cache_vec[1];
      value_cache[tgt_cache_idx3] = cache_vec[2];
      value_cache[tgt_cache_idx4] = cache_vec[3];
    }
  }
}

template <typename T, int VecSize = 4, int RoundType = 0, int HeadDim = 128>
__global__ void append_decode_cache_int8_neox_rope_kernel(
    const int* __restrict__ quant_qkv,  // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    uint8_t* __restrict__ key_cache,    // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    uint8_t* __restrict__ value_cache,  // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const float* __restrict__ qkv_out_scales,  // [num_head + 2 *
                                               // kv_num_heads, dim_head]
    const T* __restrict__ qkv_biases,  // [num_head + 2 * kv_num_heads,
                                       // dim_head]
    const T* __restrict__ cache_k_scales,
    const T* __restrict__ cache_v_scales,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int block_size,
    const float max_bound,
    const float min_bound,
    const int kv_num_heads) {
  static_assert(HeadDim == 128, "just support HeadDim be 128 now!");
  static_assert(VecSize == 4, "just support VecSize be 4 now, 32 * 4!");
  constexpr int NUM_WARPS = 4;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane_id = tid % 32;
  const int by = blockIdx.y;
  const int bid = blockIdx.x, head_idx = blockIdx.y * NUM_WARPS + wid;
  int q_head_idx, k_head_idx, v_idx;

  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * HeadDim;
  constexpr int half_head_size = HeadDim / 2;
  const int start_token_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (seq_lens_encoder[bid] > 0) return;
  const int write_seq_id = seq_lens[bid];
  if (write_seq_id == 0) return;
  const int* block_table_now = nullptr;

  block_table_now = block_tables + bid * max_blocks_per_seq;
  const int block_idx = __ldg(&block_table_now[write_seq_id / block_size]);
  const int block_offset = write_seq_id % block_size;

  if (head_idx < num_heads) {
    // q
    using LoadT = AlignedVector<int, VecSize>;
    using LoadBiasT = AlignedVector<T, VecSize>;
    using LoadOutScaleT = AlignedVector<float, VecSize>;
    constexpr int HalfVecSize = VecSize / 2;
    using LoadEmbT = AlignedVector<float, VecSize>;

    LoadT left_vec;
    LoadT right_vec;
    LoadBiasT left_bias_vec;
    LoadBiasT right_bias_vec;
    LoadOutScaleT left_out_scale_vec;
    LoadOutScaleT right_out_scale_vec;
    LoadEmbT cos_emb_vec;
    LoadEmbT sin_emb_vec;
    const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
    T* qkv_out_now = qkv_out + start_token_idx * hidden_size;
#pragma unroll
    for (uint32_t head_bias = lane_id * VecSize; head_bias < half_head_size;
         head_bias += 32 * VecSize) {
      const int bias_idx_left = head_idx * HeadDim + head_bias;
      const int bias_idx_right = bias_idx_left + half_head_size;

      Load<int, VecSize>(&qkv_now[bias_idx_left], &left_vec);
      Load<int, VecSize>(&qkv_now[bias_idx_right], &right_vec);

      if (qkv_biases) {
        Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
        Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
      }
      Load<float, VecSize>(&qkv_out_scales[bias_idx_left], &left_out_scale_vec);
      Load<float, VecSize>(&qkv_out_scales[bias_idx_right],
                           &right_out_scale_vec);

      // q rope
      const uint32_t emb_idx = write_seq_id * HeadDim + head_bias;
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);

#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        // dequant + add_bias + rope
        float input_left = static_cast<float>(left_vec[i]);
        float input_right = static_cast<float>(right_vec[i]);
        input_left = qkv_biases ? input_left * left_out_scale_vec[i] +
                                      static_cast<float>(left_bias_vec[i])
                                : input_left * left_out_scale_vec[i];
        input_right = qkv_biases ? input_right * right_out_scale_vec[i] +
                                       static_cast<float>(right_bias_vec[i])
                                 : input_right * right_out_scale_vec[i];
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
      Store<T, VecSize>(left_bias_vec, &qkv_out_now[bias_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv_out_now[bias_idx_right]);
    }
  } else if (head_idx < num_heads + 2 * kv_num_heads) {
    // k v
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    const uint32_t kv_head_idx = (head_idx - num_heads) % kv_num_heads;
    if (block_offset == 0) {
      // pad zero for this kv_head_idx for this block
      LoadPadKVT pad_cache_vec;
      *(reinterpret_cast<uint4*>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
      if (head_idx < num_heads + kv_num_heads) {
        constexpr int num_vecs_per_head_dim = HeadDim / KV_VEC_SIZE;
        constexpr int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx =
            (block_idx * kv_num_heads + kv_head_idx) * block_size * HeadDim +
            lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim;
             block_i < block_size;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(pad_cache_vec,
                                      &key_cache[tgt_idx + block_i * HeadDim]);
        }
      } else {
        const int num_vecs_per_head_dim = block_size / KV_VEC_SIZE;
        const int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx =
            (block_idx * kv_num_heads + kv_head_idx) * HeadDim * block_size +
            lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim; block_i < HeadDim;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &value_cache[tgt_idx + block_i * block_size]);
        }
      }
    }
    if (head_idx < num_heads + kv_num_heads) {
      // k
      const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
      if (head_bias < half_head_size) {
        constexpr int K_VEC_SIZE = 4;
        constexpr int HALF_K_VEC_SIZE = 2;
        using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
        using LoadKVT = AlignedVector<uint8_t, HALF_K_VEC_SIZE>;
        using LoadT = AlignedVector<int, HALF_K_VEC_SIZE>;
        using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
        using LoadOutScaleT = AlignedVector<float, HALF_K_VEC_SIZE>;
        using LoadEmbT = AlignedVector<float, HALF_K_VEC_SIZE>;

        LoadKVResT left_cache_vec, right_cache_vec;
        LoadT left_src_vec1, left_src_vec2, right_src_vec1, right_src_vec2;
        LoadBiasT left_bias_vec1, left_bias_vec2, right_bias_vec1,
            right_bias_vec2;
        LoadOutScaleT left_out_scale_vec1, left_out_scale_vec2,
            right_out_scale_vec1, right_out_scale_vec2;
        LoadEmbT cos_emb_vec1, cos_emb_vec2;
        LoadEmbT sin_emb_vec1, sin_emb_vec2;

        const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
        const int left_bias_idx = head_idx * HeadDim + head_bias;
        const int right_bias_idx = left_bias_idx + half_head_size;

        Load<int, HALF_K_VEC_SIZE>(&qkv_now[left_bias_idx], &left_src_vec1);
        Load<int, HALF_K_VEC_SIZE>(&qkv_now[left_bias_idx + 8], &left_src_vec2);
        Load<int, HALF_K_VEC_SIZE>(&qkv_now[right_bias_idx], &right_src_vec1);
        Load<int, HALF_K_VEC_SIZE>(&qkv_now[right_bias_idx + 8],
                                   &right_src_vec2);
        if (qkv_biases) {
          Load<T, HALF_K_VEC_SIZE>(&qkv_biases[left_bias_idx], &left_bias_vec1);
          Load<T, HALF_K_VEC_SIZE>(&qkv_biases[left_bias_idx + 8],
                                   &left_bias_vec2);
          Load<T, HALF_K_VEC_SIZE>(&qkv_biases[right_bias_idx],
                                   &right_bias_vec1);
          Load<T, HALF_K_VEC_SIZE>(&qkv_biases[right_bias_idx + 8],
                                   &right_bias_vec2);
        }
        Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[left_bias_idx],
                                     &left_out_scale_vec1);
        Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[left_bias_idx + 8],
                                     &left_out_scale_vec2);
        Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[right_bias_idx],
                                     &right_out_scale_vec1);
        Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[right_bias_idx + 8],
                                     &right_out_scale_vec2);

        T scale;
        const uint32_t emb_idx = write_seq_id * HeadDim + head_bias;
        Load<float, HALF_K_VEC_SIZE>(&cos_emb[emb_idx], &cos_emb_vec1);
        Load<float, HALF_K_VEC_SIZE>(&cos_emb[emb_idx + 8], &cos_emb_vec2);
        Load<float, HALF_K_VEC_SIZE>(&sin_emb[emb_idx], &sin_emb_vec1);
        Load<float, HALF_K_VEC_SIZE>(&sin_emb[emb_idx + 8], &sin_emb_vec2);
        scale = __ldg(&cache_k_scales[kv_head_idx]);
#pragma unroll
        for (int i = 0; i < HALF_K_VEC_SIZE; i++) {
          float input_left = static_cast<float>(left_src_vec1[i]);
          float input_right = static_cast<float>(right_src_vec1[i]);
          input_left = qkv_biases ? input_left * left_out_scale_vec1[i] +
                                        static_cast<float>(left_bias_vec1[i])
                                  : input_left * left_out_scale_vec1[i];
          input_right = qkv_biases ? input_right * right_out_scale_vec1[i] +
                                         static_cast<float>(right_bias_vec1[i])
                                   : input_right * right_out_scale_vec1[i];

          float cos_tmp = cos_emb_vec1[i];
          float sin_tmp = sin_emb_vec1[i];
          left_bias_vec1[i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          right_bias_vec1[i] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);

          input_left = static_cast<float>(left_src_vec2[i]);
          input_right = static_cast<float>(right_src_vec2[i]);
          input_left = qkv_biases ? input_left * left_out_scale_vec2[i] +
                                        static_cast<float>(left_bias_vec2[i])
                                  : input_left * left_out_scale_vec2[i];
          input_right = qkv_biases ? input_right * right_out_scale_vec2[i] +
                                         static_cast<float>(right_bias_vec2[i])
                                   : input_right * right_out_scale_vec2[i];
          cos_tmp = cos_emb_vec2[i];
          sin_tmp = sin_emb_vec2[i];
          left_bias_vec2[i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          right_bias_vec2[i] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);

          float quant_value1 = static_cast<float>(scale * left_bias_vec1[i]);
          float quant_value2 = static_cast<float>(scale * left_bias_vec2[i]);
          if constexpr (RoundType == 0) {
            quant_value1 =
                static_cast<float>(roundWithTiesToEven(quant_value1));
            quant_value2 =
                static_cast<float>(roundWithTiesToEven(quant_value2));
          } else {
            quant_value1 = static_cast<float>(round(quant_value1));
            quant_value2 = static_cast<float>(round(quant_value2));
          }
          quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
          quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
          quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
          quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
          left_cache_vec[i] = static_cast<uint8_t>(quant_value1 + 128.0f);
          left_cache_vec[i + HALF_K_VEC_SIZE] =
              static_cast<uint8_t>(quant_value2 + 128.0f);

          quant_value1 = static_cast<float>(scale * right_bias_vec1[i]);
          quant_value2 = static_cast<float>(scale * right_bias_vec2[i]);
          if constexpr (RoundType == 0) {
            quant_value1 =
                static_cast<float>(roundWithTiesToEven(quant_value1));
            quant_value2 =
                static_cast<float>(roundWithTiesToEven(quant_value2));
          } else {
            quant_value1 = static_cast<float>(round(quant_value1));
            quant_value2 = static_cast<float>(round(quant_value2));
          }
          quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
          quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
          quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
          quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
          right_cache_vec[i] = static_cast<uint8_t>(quant_value1 + 128.0f);
          right_cache_vec[i + HALF_K_VEC_SIZE] =
              static_cast<uint8_t>(quant_value2 + 128.0f);
        }
        // write k
        // 大分块 lane_id / 4 / 2
        // 上下 lane_id / 4 % 2
        // 左16还是右16 (block_offset % 16) / 8
        // 小偏移 lane_id % 4 * 2
        const int left_start_block_16 =
            block_offset / 16 * 16 + block_offset % 8 + lane_id / 4 % 2 * 8;
        const uint32_t left_tgt_cache_idx =
            block_idx * kv_num_heads * block_size * HeadDim +
            kv_head_idx * block_size * HeadDim + left_start_block_16 * HeadDim +
            lane_id / 4 / 2 * 32 + (block_offset % 16) / 8 * 16 +
            lane_id % 4 * 4;

        const int right_lane_id = lane_id + 16;
        const int right_start_block_16 = block_offset / 16 * 16 +
                                         block_offset % 8 +
                                         right_lane_id / 4 % 2 * 8;
        const uint32_t right_tgt_cache_idx =
            block_idx * kv_num_heads * block_size * HeadDim +
            kv_head_idx * block_size * HeadDim +
            right_start_block_16 * HeadDim + right_lane_id / 4 / 2 * 32 +
            (block_offset % 16) / 8 * 16 + right_lane_id % 4 * 4;
        Store<uint8_t, K_VEC_SIZE>(left_cache_vec,
                                   &key_cache[left_tgt_cache_idx]);
        Store<uint8_t, K_VEC_SIZE>(right_cache_vec,
                                   &key_cache[right_tgt_cache_idx]);
      }
    } else {
      // v
      constexpr int K_VEC_SIZE = 4;
      constexpr int HALF_K_VEC_SIZE = 2;
      using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
      using LoadKVT = AlignedVector<uint8_t, HALF_K_VEC_SIZE>;
      using LoadT = AlignedVector<int, HALF_K_VEC_SIZE>;
      using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
      using LoadOutScaleT = AlignedVector<float, HALF_K_VEC_SIZE>;
      LoadKVResT cache_vec;
      LoadT src_vec1, src_vec2;
      LoadBiasT bias_vec1, bias_vec2;
      LoadOutScaleT out_scale_vec1, out_scale_vec2;

      const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
      const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
      const int bias_idx = head_idx * HeadDim + head_bias;
      Load<int, HALF_K_VEC_SIZE>(&qkv_now[bias_idx], &src_vec1);
      Load<int, HALF_K_VEC_SIZE>(&qkv_now[bias_idx + 8], &src_vec2);
      if (qkv_biases) {
        Load<T, HALF_K_VEC_SIZE>(&qkv_biases[bias_idx], &bias_vec1);
        Load<T, HALF_K_VEC_SIZE>(&qkv_biases[bias_idx + 8], &bias_vec2);
      }
      Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[bias_idx], &out_scale_vec1);
      Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[bias_idx + 8],
                                   &out_scale_vec2);

      T scale = __ldg(&cache_v_scales[kv_head_idx]);

      float input_left = static_cast<float>(src_vec1[0]);
      float input_right = static_cast<float>(src_vec1[1]);
      input_left = qkv_biases ? input_left * out_scale_vec1[0] +
                                    static_cast<float>(bias_vec1[0])
                              : input_left * out_scale_vec1[0];
      input_right = qkv_biases ? input_right * out_scale_vec1[1] +
                                     static_cast<float>(bias_vec1[1])
                               : input_right * out_scale_vec1[1];

      bias_vec1[0] = static_cast<T>(input_left);
      bias_vec1[1] = static_cast<T>(input_right);

      input_left = static_cast<float>(src_vec2[0]);
      input_right = static_cast<float>(src_vec2[1]);
      input_left = qkv_biases ? input_left * out_scale_vec2[0] +
                                    static_cast<float>(bias_vec2[0])
                              : input_left * out_scale_vec2[0];
      input_right = qkv_biases ? input_right * out_scale_vec2[1] +
                                     static_cast<float>(bias_vec2[1])
                               : input_right * out_scale_vec2[1];

      bias_vec2[0] = static_cast<T>(input_left);
      bias_vec2[1] = static_cast<T>(input_right);

#pragma unroll
      for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
        float quant_value1 = static_cast<float>(scale * bias_vec1[i]);
        float quant_value2 = static_cast<float>(scale * bias_vec2[i]);
        if constexpr (RoundType == 0) {
          quant_value1 = static_cast<float>(roundWithTiesToEven(quant_value1));
          quant_value2 = static_cast<float>(roundWithTiesToEven(quant_value2));
        } else {
          quant_value1 = static_cast<float>(round(quant_value1));
          quant_value2 = static_cast<float>(round(quant_value2));
        }
        quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
        quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
        quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
        quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
        cache_vec[i] = static_cast<uint8_t>(quant_value1 + 128.0f);
        cache_vec[i + HALF_K_VEC_SIZE] =
            static_cast<uint8_t>(quant_value2 + 128.0f);
      }
      const uint32_t base_tgt_cache_idx =
          block_idx * kv_num_heads * HeadDim * block_size +
          kv_head_idx * HeadDim * block_size +
          (lane_id / 4 * 16 + lane_id % 4 * 2) * block_size +
          block_offset / 16 % 2 * 8 * block_size + block_offset / 16 / 2 * 32;
      const uint32_t tgt_cache_idx1 = base_tgt_cache_idx +
                                      block_offset % 8 / 2 * 4     // per 4
                                      + block_offset % 16 / 8 * 2  // per 2
                                      + block_offset % 2;          // per 1
      const uint32_t tgt_cache_idx2 = tgt_cache_idx1 + block_size;
      const uint32_t tgt_cache_idx3 = tgt_cache_idx1 + 16;
      const uint32_t tgt_cache_idx4 = tgt_cache_idx3 + block_size;
      value_cache[tgt_cache_idx1] = cache_vec[0];
      value_cache[tgt_cache_idx2] = cache_vec[1];
      value_cache[tgt_cache_idx3] = cache_vec[2];
      value_cache[tgt_cache_idx4] = cache_vec[3];
    }
  }
}


template <typename T, int VecSize = 4, int RoundType = 0, int HeadDim = 128>
__global__ void append_decode_cache_int4_rope_kernel(
    const T* __restrict__ quant_qkv,    // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    uint8_t* __restrict__ key_cache,    // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    uint8_t* __restrict__ value_cache,  // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const T* __restrict__ cache_k_scale,
    const T* __restrict__ cache_v_scale,
    const T* __restrict__ cache_k_zero_points,
    const T* __restrict__ cache_v_zero_points,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int block_size,
    const float max_bound,
    const float min_bound,
    const int kv_num_heads) {
  static_assert(HeadDim == 128, "just support HeadDim be 128 now!");
  static_assert(VecSize == 4, "just support VecSize be 4 now, 32 * 4!");
  constexpr int NUM_WARPS = 4;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane_id = tid % 32;
  const int bid = blockIdx.x, head_idx = blockIdx.y * NUM_WARPS + wid;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * HeadDim;
  constexpr int half_head_size = HeadDim / 2;
  const int half_block_size = block_size / 2;
  const int start_token_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (seq_lens_encoder[bid] > 0) return;
  const int write_seq_id = seq_lens[bid];
  if (write_seq_id == 0) return;
  const int* block_table_now = nullptr;

  block_table_now = block_tables + bid * max_blocks_per_seq;

  const int block_idx = __ldg(&block_table_now[write_seq_id / block_size]);
  const int block_offset = write_seq_id % block_size;

  if (head_idx < num_heads) {
    // q
    using LoadT = AlignedVector<T, VecSize>;
    using LoadBiasT = AlignedVector<T, VecSize>;
    using LoadOutScaleT = AlignedVector<float, VecSize>;
    constexpr int HalfVecSize = VecSize / 2;
    using LoadEmbT = AlignedVector<float, HalfVecSize>;

    LoadT src_vec;
    LoadBiasT out_vec;
    LoadEmbT cos_emb_vec;
    LoadEmbT sin_emb_vec;
    const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
    T* qkv_out_now = qkv_out + start_token_idx * hidden_size;
#pragma unroll
    for (uint32_t head_bias = lane_id * VecSize; head_bias < HeadDim;
         head_bias += 32 * VecSize) {
      const int bias_idx = head_idx * HeadDim + head_bias;
      Load<T, VecSize>(&qkv_now[bias_idx], &src_vec);

      // q rope
      const uint32_t emb_idx = write_seq_id * half_head_size + head_bias / 2;
      Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
      for (int i = 0; i < HalfVecSize; i++) {
        // dequant + add_bias + rope
        float input_left = static_cast<float>(src_vec[2 * i]);
        float input_right = static_cast<float>(src_vec[2 * i + 1]);

        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        out_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        out_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
      Store<T, VecSize>(out_vec, &qkv_out_now[bias_idx]);
    }
  } else if (head_idx < num_heads + 2 * kv_num_heads) {
    // k
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    const uint32_t kv_head_idx = (head_idx - num_heads) % kv_num_heads;
    if (block_offset == 0) {
      // pad zero for this kv_head_idx for this block
      LoadPadKVT pad_cache_vec;
      *(reinterpret_cast<uint4*>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
      if (head_idx < num_heads + kv_num_heads) {
        constexpr int num_vecs_per_head_dim = half_head_size / KV_VEC_SIZE;
        constexpr int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx = (block_idx * kv_num_heads + kv_head_idx) *
                                     block_size * half_head_size +
                                 lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim;
             block_i < block_size;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &key_cache[tgt_idx + block_i * half_head_size]);
        }
      } else {
        const int num_vecs_per_head_dim = half_block_size / KV_VEC_SIZE;
        const int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx = (block_idx * kv_num_heads + kv_head_idx) *
                                     HeadDim * half_block_size +
                                 lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim; block_i < HeadDim;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &value_cache[tgt_idx + block_i * half_block_size]);
        }
      }
    }

    constexpr int K_VEC_SIZE = 4;
    constexpr int HALF_K_VEC_SIZE = 2;
    using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
    using LoadT = AlignedVector<T, HALF_K_VEC_SIZE>;
    using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
    using LoadOutScaleT = AlignedVector<float, HALF_K_VEC_SIZE>;
    using LoadScaleT = AlignedVector<T, HALF_K_VEC_SIZE>;
    using LoadEmbT = AlignedVector<float, 1>;
    LoadT src_vec1, src_vec2;
    LoadBiasT out_vec1, out_vec2;
    LoadScaleT scale_vec1, scale_vec2;
    LoadScaleT zp_vec1, zp_vec2;
    LoadEmbT cos_emb_vec1, cos_emb_vec2;
    LoadEmbT sin_emb_vec1, sin_emb_vec2;

    const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
    const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
    const uint32_t cache_idx = kv_head_idx * HeadDim + head_bias;
    const int bias_idx = head_idx * HeadDim + head_bias;
    Load<T, HALF_K_VEC_SIZE>(&qkv_now[bias_idx], &src_vec1);
    Load<T, HALF_K_VEC_SIZE>(&qkv_now[bias_idx + 8], &src_vec2);
    if (head_idx < num_heads + kv_num_heads) {
      const uint32_t emb_idx = write_seq_id * half_head_size + head_bias / 2;
      Load<float, 1>(&cos_emb[emb_idx], &cos_emb_vec1);
      Load<float, 1>(&cos_emb[emb_idx + 4], &cos_emb_vec2);
      Load<float, 1>(&sin_emb[emb_idx], &sin_emb_vec1);
      Load<float, 1>(&sin_emb[emb_idx + 4], &sin_emb_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[cache_idx], &scale_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[cache_idx + 8], &scale_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[cache_idx], &zp_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[cache_idx + 8], &zp_vec2);
    } else {
      Load<T, HALF_K_VEC_SIZE>(&cache_v_scale[cache_idx], &scale_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_scale[cache_idx + 8], &scale_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_zero_points[cache_idx], &zp_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_zero_points[cache_idx + 8], &zp_vec2);
    }

    float input_left = static_cast<float>(src_vec1[0]);
    float input_right = static_cast<float>(src_vec1[1]);
    if (head_idx < num_heads + kv_num_heads) {
      float cos_tmp = cos_emb_vec1[0];
      float sin_tmp = sin_emb_vec1[0];
      out_vec1[0] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      out_vec1[1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    } else {
      out_vec1[0] = src_vec1[0];
      out_vec1[1] = src_vec1[1];
    }

    input_left = static_cast<float>(src_vec2[0]);
    input_right = static_cast<float>(src_vec2[1]);
    if (head_idx < num_heads + kv_num_heads) {
      float cos_tmp = cos_emb_vec2[0];
      float sin_tmp = sin_emb_vec2[0];
      out_vec2[0] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      out_vec2[1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    } else {
      out_vec2[0] = src_vec2[0];
      out_vec2[1] = src_vec2[1];
    }
    if (head_idx < num_heads + kv_num_heads) {
      LoadKVResT cache_vec;
      const int start_block_16 =
          block_offset / 16 * 16 + block_offset % 8 + lane_id / 4 % 4 / 2 * 8;
      const uint32_t tgt_cache_idx =
          block_idx * kv_num_heads * block_size * half_head_size +
          kv_head_idx * block_size * half_head_size +
          start_block_16 * half_head_size + lane_id / 4 / 4 * 32 +
          lane_id / 4 % 2 * 16 + lane_id % 4 * 4;
      Load<uint8_t, K_VEC_SIZE>(&key_cache[tgt_cache_idx], &cache_vec);

#pragma unroll
      for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
        float quant_value =
            static_cast<float>(scale_vec1[i] * out_vec1[i] + zp_vec1[i]);
        if constexpr (RoundType == 0) {
          quant_value = roundWithTiesToEven(quant_value);
        } else {
          quant_value = round(quant_value);
        }
        quant_value = quant_value > max_bound ? max_bound : quant_value;
        quant_value = quant_value < min_bound ? min_bound : quant_value;
        uint8_t uint_quant_value = static_cast<uint8_t>(quant_value + 8.0f);
        uint8_t ano_uint_quant_value = 0;
        if (block_offset % 16 / 8 == 0) {
          cache_vec[i] |= ((ano_uint_quant_value) | (uint_quant_value & 0x0F));
        } else {
          cache_vec[i] |= ((uint_quant_value << 4) | (ano_uint_quant_value));
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
        float quant_value =
            static_cast<float>(scale_vec2[i] * out_vec2[i] + zp_vec2[i]);
        if constexpr (RoundType == 0) {
          quant_value = roundWithTiesToEven(quant_value);
        } else {
          quant_value = round(quant_value);
        }
        quant_value = quant_value > max_bound ? max_bound : quant_value;
        quant_value = quant_value < min_bound ? min_bound : quant_value;
        uint8_t uint_quant_value = static_cast<uint8_t>(quant_value + 8.0f);
        uint8_t ano_uint_quant_value = 0;
        if (block_offset % 16 / 8 == 0) {
          cache_vec[i + HALF_K_VEC_SIZE] |=
              ((ano_uint_quant_value) | (uint_quant_value & 0x0F));
        } else {
          cache_vec[i + HALF_K_VEC_SIZE] |=
              ((uint_quant_value << 4) | (ano_uint_quant_value));
        }
      }
      Store<uint8_t, K_VEC_SIZE>(cache_vec, &key_cache[tgt_cache_idx]);
    } else {
      const uint32_t base_tgt_cache_idx =
          block_idx * kv_num_heads * HeadDim * half_block_size +
          kv_head_idx * HeadDim * half_block_size +
          (lane_id / 4 * 16 + lane_id % 4 * 2) * half_block_size +
          block_offset / 16 % 4 / 2 * 8 * half_block_size +
          block_offset / 16 / 4 * 32 + block_offset / 16 % 2 * 16;
      const uint32_t tgt_cache_idx1 = base_tgt_cache_idx +
                                      block_offset % 8 / 2 * 4     // per 4
                                      + block_offset % 16 / 8 * 2  // per 2
                                      + block_offset % 2;          // per 1
      const uint32_t tgt_cache_idx2 = tgt_cache_idx1 + half_block_size;

      float quant_value1 =
          static_cast<float>(scale_vec1[0] * out_vec1[0] + zp_vec1[0]);
      float quant_value2 =
          static_cast<float>(scale_vec2[0] * out_vec2[0] + zp_vec2[0]);
      if constexpr (RoundType == 0) {
        quant_value1 = roundWithTiesToEven(quant_value1);
        quant_value2 = roundWithTiesToEven(quant_value2);
      } else {
        quant_value1 = round(quant_value1);
        quant_value2 = round(quant_value2);
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      uint8_t uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
      uint8_t uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
      value_cache[tgt_cache_idx1] =
          (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);

      quant_value1 =
          static_cast<float>(scale_vec1[1] * out_vec1[1] + zp_vec1[1]);
      quant_value2 =
          static_cast<float>(scale_vec2[1] * out_vec2[1] + zp_vec2[1]);
      if constexpr (RoundType == 0) {
        quant_value1 = roundWithTiesToEven(quant_value1);
        quant_value2 = roundWithTiesToEven(quant_value2);
      } else {
        quant_value1 = round(quant_value1);
        quant_value2 = round(quant_value2);
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
      uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
      value_cache[tgt_cache_idx2] =
          (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
    }
  }
}

template <typename T, int VecSize = 4, int RoundType = 0, int HeadDim = 128>
__global__ void append_decode_cache_int4_rope_kernel(
    const int* __restrict__ quant_qkv,  // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    uint8_t* __restrict__ key_cache,    // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    uint8_t* __restrict__ value_cache,  // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const float* __restrict__ qkv_out_scales,  // [num_head + 2 *
                                               // kv_num_heads, dim_head]
    const T* __restrict__ qkv_biases,  // [num_head + 2 * kv_num_heads,
                                       // dim_head]
    const T* __restrict__ cache_k_scale,
    const T* __restrict__ cache_v_scale,
    const T* __restrict__ cache_k_zero_points,
    const T* __restrict__ cache_v_zero_points,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int block_size,
    const float max_bound,
    const float min_bound,
    const int kv_num_heads) {
  static_assert(HeadDim == 128, "just support HeadDim be 128 now!");
  static_assert(VecSize == 4, "just support VecSize be 4 now, 32 * 4!");
  constexpr int NUM_WARPS = 4;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane_id = tid % 32;
  const int bid = blockIdx.x, head_idx = blockIdx.y * NUM_WARPS + wid;

  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * HeadDim;
  constexpr int half_head_size = HeadDim / 2;
  const int half_block_size = block_size / 2;
  const int start_token_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (seq_lens_encoder[bid] > 0) return;
  const int write_seq_id = seq_lens[bid];
  if (write_seq_id == 0) return;
  const int* block_table_now = nullptr;

  block_table_now = block_tables + bid * max_blocks_per_seq;

  const int block_idx = __ldg(&block_table_now[write_seq_id / block_size]);
  const int block_offset = write_seq_id % block_size;

  if (head_idx < num_heads) {
    // q
    using LoadT = AlignedVector<int, VecSize>;
    using LoadBiasT = AlignedVector<T, VecSize>;
    using LoadOutScaleT = AlignedVector<float, VecSize>;
    constexpr int HalfVecSize = VecSize / 2;
    using LoadEmbT = AlignedVector<float, HalfVecSize>;

    LoadT src_vec;
    LoadBiasT bias_vec;
    LoadOutScaleT out_scale_vec;
    LoadEmbT cos_emb_vec;
    LoadEmbT sin_emb_vec;
    const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
    T* qkv_out_now = qkv_out + start_token_idx * hidden_size;
#pragma unroll
    for (uint32_t head_bias = lane_id * VecSize; head_bias < HeadDim;
         head_bias += 32 * VecSize) {
      const int bias_idx = head_idx * HeadDim + head_bias;
      Load<int, VecSize>(&qkv_now[bias_idx], &src_vec);
      if (qkv_biases) {
        Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
      }
      Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
      // q rope
      const uint32_t emb_idx = write_seq_id * half_head_size + head_bias / 2;
      Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
      for (int i = 0; i < HalfVecSize; i++) {
        // dequant + add_bias + rope
        float input_left = static_cast<float>(src_vec[2 * i]);
        float input_right = static_cast<float>(src_vec[2 * i + 1]);
        input_left = qkv_biases ? input_left * out_scale_vec[2 * i] +
                                      static_cast<float>(bias_vec[2 * i])
                                : input_left * out_scale_vec[2 * i];
        input_right = qkv_biases ? input_right * out_scale_vec[2 * i + 1] +
                                       static_cast<float>(bias_vec[2 * i + 1])
                                 : input_right * out_scale_vec[2 * i + 1];
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
      Store<T, VecSize>(bias_vec, &qkv_out_now[bias_idx]);
    }
  } else if (head_idx < num_heads + 2 * kv_num_heads) {
    // k
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    const uint32_t kv_head_idx = (head_idx - num_heads) % kv_num_heads;
    if (block_offset == 0) {
      // pad zero for this kv_head_idx for this block
      LoadPadKVT pad_cache_vec;
      *(reinterpret_cast<uint4*>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
      if (head_idx < num_heads + kv_num_heads) {
        constexpr int num_vecs_per_head_dim = half_head_size / KV_VEC_SIZE;
        constexpr int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx = (block_idx * kv_num_heads + kv_head_idx) *
                                     block_size * half_head_size +
                                 lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim;
             block_i < block_size;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &key_cache[tgt_idx + block_i * half_head_size]);
        }
      } else {
        const int num_vecs_per_head_dim = half_block_size / KV_VEC_SIZE;
        const int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx = (block_idx * kv_num_heads + kv_head_idx) *
                                     HeadDim * half_block_size +
                                 lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim; block_i < HeadDim;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &value_cache[tgt_idx + block_i * half_block_size]);
        }
      }
    }

    constexpr int K_VEC_SIZE = 4;
    constexpr int HALF_K_VEC_SIZE = 2;
    using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
    using LoadT = AlignedVector<int, HALF_K_VEC_SIZE>;
    using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
    using LoadOutScaleT = AlignedVector<float, HALF_K_VEC_SIZE>;
    using LoadScaleT = AlignedVector<T, HALF_K_VEC_SIZE>;
    using LoadEmbT = AlignedVector<float, 1>;
    LoadT src_vec1, src_vec2;
    LoadBiasT bias_vec1, bias_vec2;
    LoadOutScaleT out_scale_vec1, out_scale_vec2;
    LoadScaleT scale_vec1, scale_vec2;
    LoadScaleT zp_vec1, zp_vec2;
    LoadEmbT cos_emb_vec1, cos_emb_vec2;
    LoadEmbT sin_emb_vec1, sin_emb_vec2;

    const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
    const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
    const uint32_t cache_idx = kv_head_idx * HeadDim + head_bias;
    const int bias_idx = head_idx * HeadDim + head_bias;
    Load<int, HALF_K_VEC_SIZE>(&qkv_now[bias_idx], &src_vec1);
    Load<int, HALF_K_VEC_SIZE>(&qkv_now[bias_idx + 8], &src_vec2);
    if (qkv_biases) {
      Load<T, HALF_K_VEC_SIZE>(&qkv_biases[bias_idx], &bias_vec1);
      Load<T, HALF_K_VEC_SIZE>(&qkv_biases[bias_idx + 8], &bias_vec2);
    }
    Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[bias_idx], &out_scale_vec1);
    Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[bias_idx + 8],
                                 &out_scale_vec2);
    if (head_idx < num_heads + kv_num_heads) {
      const uint32_t emb_idx = write_seq_id * half_head_size + head_bias / 2;
      Load<float, 1>(&cos_emb[emb_idx], &cos_emb_vec1);
      Load<float, 1>(&cos_emb[emb_idx + 4], &cos_emb_vec2);
      Load<float, 1>(&sin_emb[emb_idx], &sin_emb_vec1);
      Load<float, 1>(&sin_emb[emb_idx + 4], &sin_emb_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[cache_idx], &scale_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[cache_idx + 8], &scale_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[cache_idx], &zp_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[cache_idx + 8], &zp_vec2);
    } else {
      Load<T, HALF_K_VEC_SIZE>(&cache_v_scale[cache_idx], &scale_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_scale[cache_idx + 8], &scale_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_zero_points[cache_idx], &zp_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_zero_points[cache_idx + 8], &zp_vec2);
    }

    float input_left = static_cast<float>(src_vec1[0]);
    float input_right = static_cast<float>(src_vec1[1]);
    input_left = qkv_biases ? input_left * out_scale_vec1[0] +
                                  static_cast<float>(bias_vec1[0])
                            : input_left * out_scale_vec1[0];
    input_right = qkv_biases ? input_right * out_scale_vec1[1] +
                                   static_cast<float>(bias_vec1[1])
                             : input_right * out_scale_vec1[1];
    if (head_idx < num_heads + kv_num_heads) {
      float cos_tmp = cos_emb_vec1[0];
      float sin_tmp = sin_emb_vec1[0];
      bias_vec1[0] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      bias_vec1[1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    } else {
      bias_vec1[0] = static_cast<T>(input_left);
      bias_vec1[1] = static_cast<T>(input_right);
    }

    input_left = static_cast<float>(src_vec2[0]);
    input_right = static_cast<float>(src_vec2[1]);
    input_left = qkv_biases ? input_left * out_scale_vec2[0] +
                                  static_cast<float>(bias_vec2[0])
                            : input_left * out_scale_vec2[0];
    input_right = qkv_biases ? input_right * out_scale_vec2[1] +
                                   static_cast<float>(bias_vec2[1])
                             : input_right * out_scale_vec2[1];
    if (head_idx < num_heads + kv_num_heads) {
      float cos_tmp = cos_emb_vec2[0];
      float sin_tmp = sin_emb_vec2[0];
      bias_vec2[0] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      bias_vec2[1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    } else {
      bias_vec2[0] = static_cast<T>(input_left);
      bias_vec2[1] = static_cast<T>(input_right);
    }
    if (head_idx < num_heads + kv_num_heads) {
      LoadKVResT cache_vec;
      const int start_block_16 =
          block_offset / 16 * 16 + block_offset % 8 + lane_id / 4 % 4 / 2 * 8;
      const uint32_t tgt_cache_idx =
          block_idx * kv_num_heads * block_size * half_head_size +
          kv_head_idx * block_size * half_head_size +
          start_block_16 * half_head_size + lane_id / 4 / 4 * 32 +
          lane_id / 4 % 2 * 16 + lane_id % 4 * 4;
      Load<uint8_t, K_VEC_SIZE>(&key_cache[tgt_cache_idx], &cache_vec);
#pragma unroll
      for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
        float quant_value =
            static_cast<float>(scale_vec1[i] * bias_vec1[i] + zp_vec1[i]);
        if constexpr (RoundType == 0) {
          quant_value = roundWithTiesToEven(quant_value);
        } else {
          quant_value = round(quant_value);
        }
        quant_value = quant_value > max_bound ? max_bound : quant_value;
        quant_value = quant_value < min_bound ? min_bound : quant_value;
        uint8_t uint_quant_value = static_cast<uint8_t>(quant_value + 8.0f);
        uint8_t ano_uint_quant_value = 0;
        if (block_offset % 16 / 8 == 0) {
          cache_vec[i] |= ((ano_uint_quant_value) | (uint_quant_value & 0x0F));
        } else {
          cache_vec[i] |= ((uint_quant_value << 4) | (ano_uint_quant_value));
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
        float quant_value =
            static_cast<float>(scale_vec2[i] * bias_vec2[i] + zp_vec2[i]);
        if constexpr (RoundType == 0) {
          quant_value = roundWithTiesToEven(quant_value);
        } else {
          quant_value = round(quant_value);
        }
        quant_value = quant_value > max_bound ? max_bound : quant_value;
        quant_value = quant_value < min_bound ? min_bound : quant_value;
        uint8_t uint_quant_value = static_cast<uint8_t>(quant_value + 8.0f);
        uint8_t ano_uint_quant_value = 0;
        if (block_offset % 16 / 8 == 0) {
          cache_vec[i + HALF_K_VEC_SIZE] |=
              ((ano_uint_quant_value) | (uint_quant_value & 0x0F));
        } else {
          cache_vec[i + HALF_K_VEC_SIZE] |=
              ((uint_quant_value << 4) | (ano_uint_quant_value));
        }
      }
      Store<uint8_t, K_VEC_SIZE>(cache_vec, &key_cache[tgt_cache_idx]);
    } else {
      const uint32_t base_tgt_cache_idx =
          block_idx * kv_num_heads * HeadDim * half_block_size +
          kv_head_idx * HeadDim * half_block_size +
          (lane_id / 4 * 16 + lane_id % 4 * 2) * half_block_size +
          block_offset / 16 % 4 / 2 * 8 * half_block_size +
          block_offset / 16 / 4 * 32 + block_offset / 16 % 2 * 16;
      const uint32_t tgt_cache_idx1 = base_tgt_cache_idx +
                                      block_offset % 8 / 2 * 4     // per 4
                                      + block_offset % 16 / 8 * 2  // per 2
                                      + block_offset % 2;          // per 1
      const uint32_t tgt_cache_idx2 = tgt_cache_idx1 + half_block_size;

      float quant_value1 =
          static_cast<float>(scale_vec1[0] * bias_vec1[0] + zp_vec1[0]);
      float quant_value2 =
          static_cast<float>(scale_vec2[0] * bias_vec2[0] + zp_vec2[0]);
      if constexpr (RoundType == 0) {
        quant_value1 = roundWithTiesToEven(quant_value1);
        quant_value2 = roundWithTiesToEven(quant_value2);
      } else {
        quant_value1 = round(quant_value1);
        quant_value2 = round(quant_value2);
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      uint8_t uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
      uint8_t uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
      value_cache[tgt_cache_idx1] =
          (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);

      quant_value1 =
          static_cast<float>(scale_vec1[1] * bias_vec1[1] + zp_vec1[1]);
      quant_value2 =
          static_cast<float>(scale_vec2[1] * bias_vec2[1] + zp_vec2[1]);
      if constexpr (RoundType == 0) {
        quant_value1 = roundWithTiesToEven(quant_value1);
        quant_value2 = roundWithTiesToEven(quant_value2);
      } else {
        quant_value1 = round(quant_value1);
        quant_value2 = round(quant_value2);
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
      uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
      value_cache[tgt_cache_idx2] =
          (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
    }
  }
}

template <typename T, int VecSize = 4, int RoundType = 0, int HeadDim = 128>
__global__ void append_decode_cache_int4_neox_rope_kernel(
    const T* __restrict__ quant_qkv,    // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    uint8_t* __restrict__ key_cache,    // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    uint8_t* __restrict__ value_cache,  // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const T* __restrict__ cache_k_scale,
    const T* __restrict__ cache_v_scale,
    const T* __restrict__ cache_k_zero_points,
    const T* __restrict__ cache_v_zero_points,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int block_size,
    const float max_bound,
    const float min_bound,
    const int kv_num_heads) {
  static_assert(HeadDim == 128, "just support HeadDim be 128 now!");
  static_assert(VecSize == 4, "just support VecSize be 4 now, 32 * 4!");
  constexpr int NUM_WARPS = 4;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane_id = tid % 32;
  const int bid = blockIdx.x, head_idx = blockIdx.y * NUM_WARPS + wid;
  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * HeadDim;
  constexpr int half_head_size = HeadDim / 2;
  const int half_block_size = block_size / 2;
  const int start_token_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (seq_lens_encoder[bid] > 0) return;
  const int write_seq_id = seq_lens[bid];
  if (write_seq_id == 0) return;
  const int* block_table_now = nullptr;

  block_table_now = block_tables + bid * max_blocks_per_seq;

  const int block_idx = __ldg(&block_table_now[write_seq_id / block_size]);
  const int block_offset = write_seq_id % block_size;

  if (head_idx < num_heads) {
    // q
    using LoadT = AlignedVector<T, VecSize>;
    using LoadBiasT = AlignedVector<T, VecSize>;
    constexpr int HalfVecSize = VecSize / 2;
    using LoadEmbT = AlignedVector<float, VecSize>;

    LoadT left_vec;
    LoadT right_vec;
    LoadBiasT left_out_vec;
    LoadBiasT right_out_vec;
    LoadEmbT cos_emb_vec;
    LoadEmbT sin_emb_vec;
    const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
    T* qkv_out_now = qkv_out + start_token_idx * hidden_size;
#pragma unroll
    for (uint32_t head_bias = lane_id * VecSize; head_bias < half_head_size;
         head_bias += 32 * VecSize) {
      const int bias_idx_left = head_idx * HeadDim + head_bias;
      const int bias_idx_right = bias_idx_left + half_head_size;
      Load<T, VecSize>(&qkv_now[bias_idx_left], &left_vec);
      Load<T, VecSize>(&qkv_now[bias_idx_right], &right_vec);

      // q rope
      const uint32_t emb_idx = write_seq_id * HeadDim + head_bias;
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        // dequant + add_bias + rope
        float input_left = static_cast<float>(left_vec[i]);
        float input_right = static_cast<float>(right_vec[i]);

        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_out_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_out_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
      Store<T, VecSize>(left_out_vec, &qkv_out_now[bias_idx_left]);
      Store<T, VecSize>(right_out_vec, &qkv_out_now[bias_idx_right]);
    }
  } else if (head_idx < num_heads + 2 * kv_num_heads) {
    // k
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    const uint32_t kv_head_idx = (head_idx - num_heads) % kv_num_heads;
    if (block_offset == 0) {
      // pad zero for this kv_head_idx for this block
      LoadPadKVT pad_cache_vec;
      *(reinterpret_cast<uint4*>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
      if (head_idx < num_heads + kv_num_heads) {
        constexpr int num_vecs_per_head_dim = half_head_size / KV_VEC_SIZE;
        constexpr int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx = (block_idx * kv_num_heads + kv_head_idx) *
                                     block_size * half_head_size +
                                 lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim;
             block_i < block_size;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &key_cache[tgt_idx + block_i * half_head_size]);
        }
      } else {
        const int num_vecs_per_head_dim = half_block_size / KV_VEC_SIZE;
        const int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx = (block_idx * kv_num_heads + kv_head_idx) *
                                     HeadDim * half_block_size +
                                 lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim; block_i < HeadDim;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &value_cache[tgt_idx + block_i * half_block_size]);
        }
      }
    }
    if (head_idx < num_heads + kv_num_heads) {
      const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
      if (head_bias < half_head_size) {
        constexpr int K_VEC_SIZE = 4;
        constexpr int HALF_K_VEC_SIZE = 2;
        using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
        using LoadT = AlignedVector<T, HALF_K_VEC_SIZE>;
        using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
        using LoadScaleT = AlignedVector<T, HALF_K_VEC_SIZE>;
        using LoadEmbT = AlignedVector<float, HALF_K_VEC_SIZE>;
        LoadT left_src_vec1, left_src_vec2, right_src_vec1, right_src_vec2;
        LoadBiasT left_out_vec1, left_out_vec2, right_out_vec1, right_out_vec2;
        LoadScaleT left_scale_vec1, left_scale_vec2, right_scale_vec1,
            right_scale_vec2;
        LoadScaleT left_zp_vec1, left_zp_vec2, right_zp_vec1, right_zp_vec2;
        LoadEmbT cos_emb_vec1, cos_emb_vec2;
        LoadEmbT sin_emb_vec1, sin_emb_vec2;

        const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
        const int left_bias_idx = head_idx * HeadDim + head_bias;
        const int right_bias_idx = left_bias_idx + half_head_size;

        const uint32_t left_cache_idx = kv_head_idx * HeadDim + head_bias;
        const uint32_t right_cache_idx = left_cache_idx + half_head_size;

        Load<T, HALF_K_VEC_SIZE>(&qkv_now[left_bias_idx], &left_src_vec1);
        Load<T, HALF_K_VEC_SIZE>(&qkv_now[left_bias_idx + 8], &left_src_vec2);
        Load<T, HALF_K_VEC_SIZE>(&qkv_now[right_bias_idx], &right_src_vec1);
        Load<T, HALF_K_VEC_SIZE>(&qkv_now[right_bias_idx + 8], &right_src_vec2);
        const uint32_t emb_idx = write_seq_id * HeadDim + head_bias;
        Load<float, HALF_K_VEC_SIZE>(&cos_emb[emb_idx], &cos_emb_vec1);
        Load<float, HALF_K_VEC_SIZE>(&cos_emb[emb_idx + 8], &cos_emb_vec2);
        Load<float, HALF_K_VEC_SIZE>(&sin_emb[emb_idx], &sin_emb_vec1);
        Load<float, HALF_K_VEC_SIZE>(&sin_emb[emb_idx + 8], &sin_emb_vec2);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[left_cache_idx],
                                 &left_scale_vec1);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[left_cache_idx + 8],
                                 &left_scale_vec2);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[left_cache_idx],
                                 &left_zp_vec1);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[left_cache_idx + 8],
                                 &left_zp_vec2);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[right_cache_idx],
                                 &right_scale_vec1);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[right_cache_idx + 8],
                                 &right_scale_vec2);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[right_cache_idx],
                                 &right_zp_vec1);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[right_cache_idx + 8],
                                 &right_zp_vec2);

        for (int i = 0; i < HALF_K_VEC_SIZE; i++) {
          float input_left = static_cast<float>(left_src_vec1[i]);
          float input_right = static_cast<float>(right_src_vec1[i]);
          float cos_tmp = cos_emb_vec1[0];
          float sin_tmp = sin_emb_vec1[0];
          left_out_vec1[i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          right_out_vec1[i] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);


          input_left = static_cast<float>(left_src_vec2[i]);
          input_right = static_cast<float>(right_src_vec2[i]);
          cos_tmp = cos_emb_vec2[i];
          sin_tmp = sin_emb_vec2[i];
          left_out_vec2[i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          right_out_vec2[i] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
          // quant + write k
        }
        LoadKVResT left_cache_vec, right_cache_vec;
        const int left_start_block_16 =
            block_offset / 16 * 16 + block_offset % 8 + lane_id / 4 % 4 / 2 * 8;
        const uint32_t left_tgt_cache_idx =
            block_idx * kv_num_heads * block_size * half_head_size +
            kv_head_idx * block_size * half_head_size +
            left_start_block_16 * half_head_size + lane_id / 4 / 4 * 32 +
            lane_id / 4 % 2 * 16 + lane_id % 4 * 4;

        const int right_lane_id = lane_id + 16;
        const int right_start_block_16 = block_offset / 16 * 16 +
                                         block_offset % 8 +
                                         right_lane_id / 4 % 4 / 2 * 8;
        const uint32_t right_tgt_cache_idx =
            block_idx * kv_num_heads * block_size * half_head_size +
            kv_head_idx * block_size * half_head_size +
            right_start_block_16 * half_head_size + right_lane_id / 4 / 4 * 32 +
            right_lane_id / 4 % 2 * 16 + right_lane_id % 4 * 4;
        Load<uint8_t, K_VEC_SIZE>(&key_cache[left_tgt_cache_idx],
                                  &left_cache_vec);
        Load<uint8_t, K_VEC_SIZE>(&key_cache[right_tgt_cache_idx],
                                  &right_cache_vec);

#pragma unroll
        for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
          float quant_value1 = static_cast<float>(
              left_scale_vec1[i] * left_out_vec1[i] + left_zp_vec1[i]);
          float quant_value2 = static_cast<float>(
              left_scale_vec2[i] * left_out_vec2[i] + left_zp_vec2[i]);
          if constexpr (RoundType == 0) {
            quant_value1 = roundWithTiesToEven(quant_value1);
            quant_value2 = roundWithTiesToEven(quant_value2);
          } else {
            quant_value1 = round(quant_value1);
            quant_value2 = round(quant_value2);
          }
          quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
          quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
          quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
          quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
          uint8_t uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
          uint8_t uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
          uint8_t ano_uint_quant_value = 0;
          if (block_offset % 16 / 8 == 0) {
            left_cache_vec[i] |=
                ((ano_uint_quant_value) | (uint_quant_value1 & 0x0F));
            left_cache_vec[i + HALF_K_VEC_SIZE] |=
                ((ano_uint_quant_value) | (uint_quant_value2 & 0x0F));
          } else {
            left_cache_vec[i] |=
                ((uint_quant_value1 << 4) | (ano_uint_quant_value));
            left_cache_vec[i + HALF_K_VEC_SIZE] |=
                ((uint_quant_value2 << 4) | (ano_uint_quant_value));
          }

          quant_value1 = static_cast<float>(
              right_scale_vec1[i] * right_out_vec1[i] + right_zp_vec1[i]);
          quant_value2 = static_cast<float>(
              right_scale_vec2[i] * right_out_vec2[i] + right_zp_vec2[i]);
          if constexpr (RoundType == 0) {
            quant_value1 = roundWithTiesToEven(quant_value1);
            quant_value2 = roundWithTiesToEven(quant_value2);
          } else {
            quant_value1 = round(quant_value1);
            quant_value2 = round(quant_value2);
          }
          quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
          quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
          quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
          quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
          uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
          uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
          ano_uint_quant_value = 0;
          if (block_offset % 16 / 8 == 0) {
            right_cache_vec[i] |=
                ((ano_uint_quant_value) | (uint_quant_value1 & 0x0F));
            right_cache_vec[i + HALF_K_VEC_SIZE] |=
                ((ano_uint_quant_value) | (uint_quant_value2 & 0x0F));
          } else {
            right_cache_vec[i] |=
                ((uint_quant_value1 << 4) | (ano_uint_quant_value));
            right_cache_vec[i + HALF_K_VEC_SIZE] |=
                ((uint_quant_value2 << 4) | (ano_uint_quant_value));
          }
        }
        Store<uint8_t, K_VEC_SIZE>(left_cache_vec,
                                   &key_cache[left_tgt_cache_idx]);
        Store<uint8_t, K_VEC_SIZE>(right_cache_vec,
                                   &key_cache[right_tgt_cache_idx]);
      }
    } else {
      constexpr int K_VEC_SIZE = 4;
      constexpr int HALF_K_VEC_SIZE = 2;
      using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
      using LoadT = AlignedVector<T, HALF_K_VEC_SIZE>;
      using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
      using LoadScaleT = AlignedVector<T, HALF_K_VEC_SIZE>;
      LoadT src_vec1, src_vec2;
      LoadBiasT out_vec1, out_vec2;
      LoadScaleT scale_vec1, scale_vec2;
      LoadScaleT zp_vec1, zp_vec2;

      const T* qkv_now = quant_qkv + start_token_idx * hidden_size;
      const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
      const uint32_t cache_idx = kv_head_idx * HeadDim + head_bias;
      const int bias_idx = head_idx * HeadDim + head_bias;
      Load<T, HALF_K_VEC_SIZE>(&qkv_now[bias_idx], &src_vec1);
      Load<T, HALF_K_VEC_SIZE>(&qkv_now[bias_idx + 8], &src_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_scale[cache_idx], &scale_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_scale[cache_idx + 8], &scale_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_zero_points[cache_idx], &zp_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_zero_points[cache_idx + 8], &zp_vec2);

      out_vec1[0] = src_vec1[0];
      out_vec1[1] = src_vec1[1];
      out_vec2[0] = src_vec2[0];
      out_vec2[1] = src_vec2[1];

      const uint32_t base_tgt_cache_idx =
          block_idx * kv_num_heads * HeadDim * half_block_size +
          kv_head_idx * HeadDim * half_block_size +
          (lane_id / 4 * 16 + lane_id % 4 * 2) * half_block_size +
          block_offset / 16 % 4 / 2 * 8 * half_block_size +
          block_offset / 16 / 4 * 32 + block_offset / 16 % 2 * 16;
      const uint32_t tgt_cache_idx1 = base_tgt_cache_idx +
                                      block_offset % 8 / 2 * 4     // per 4
                                      + block_offset % 16 / 8 * 2  // per 2
                                      + block_offset % 2;          // per 1
      const uint32_t tgt_cache_idx2 = tgt_cache_idx1 + half_block_size;

      float quant_value1 =
          static_cast<float>(scale_vec1[0] * out_vec1[0] + zp_vec1[0]);
      float quant_value2 =
          static_cast<float>(scale_vec2[0] * out_vec2[0] + zp_vec2[0]);
      if constexpr (RoundType == 0) {
        quant_value1 = roundWithTiesToEven(quant_value1);
        quant_value2 = roundWithTiesToEven(quant_value2);
      } else {
        quant_value1 = round(quant_value1);
        quant_value2 = round(quant_value2);
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      uint8_t uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
      uint8_t uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
      value_cache[tgt_cache_idx1] =
          (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);

      quant_value1 =
          static_cast<float>(scale_vec1[1] * out_vec1[1] + zp_vec1[1]);
      quant_value2 =
          static_cast<float>(scale_vec2[1] * out_vec2[1] + zp_vec2[1]);
      if constexpr (RoundType == 0) {
        quant_value1 = roundWithTiesToEven(quant_value1);
        quant_value2 = roundWithTiesToEven(quant_value2);
      } else {
        quant_value1 = round(quant_value1);
        quant_value2 = round(quant_value2);
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
      uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
      value_cache[tgt_cache_idx2] =
          (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
    }
  }
}

template <typename T, int VecSize = 4, int RoundType = 0, int HeadDim = 128>
__global__ void append_decode_cache_int4_neox_rope_kernel(
    const int* __restrict__ quant_qkv,  // [bsz, num_heads + 2 * kv_num_heads,
                                        // head_size]
    uint8_t* __restrict__ key_cache,    // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    uint8_t* __restrict__ value_cache,  // [num_blocks, kv_num_heads,
                                        // block_size, head_size // 2]
    T* __restrict__ qkv_out,
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,  // [num_tokens]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const float* __restrict__ cos_emb,
    const float* __restrict__ sin_emb,
    const float* __restrict__ qkv_out_scales,  // [num_head + 2 *
                                               // kv_num_heads, dim_head]
    const T* __restrict__ qkv_biases,  // [num_head + 2 * kv_num_heads,
                                       // dim_head]
    const T* __restrict__ cache_k_scale,
    const T* __restrict__ cache_v_scale,
    const T* __restrict__ cache_k_zero_points,
    const T* __restrict__ cache_v_zero_points,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int block_size,
    const float max_bound,
    const float min_bound,
    const int kv_num_heads) {
  static_assert(HeadDim == 128, "just support HeadDim be 128 now!");
  static_assert(VecSize == 4, "just support VecSize be 4 now, 32 * 4!");
  constexpr int NUM_WARPS = 4;
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane_id = tid % 32;
  const int bid = blockIdx.x, head_idx = blockIdx.y * NUM_WARPS + wid;

  const int64_t hidden_size = (num_heads + 2 * kv_num_heads) * HeadDim;
  constexpr int half_head_size = HeadDim / 2;
  const int half_block_size = block_size / 2;
  const int start_token_idx = bid * max_seq_len - __ldg(&cum_offsets[bid]);
  if (seq_lens_encoder[bid] > 0) return;
  const int write_seq_id = seq_lens[bid];
  if (write_seq_id == 0) return;
  const int* block_table_now = nullptr;

  block_table_now = block_tables + bid * max_blocks_per_seq;

  const int block_idx = __ldg(&block_table_now[write_seq_id / block_size]);
  const int block_offset = write_seq_id % block_size;

  if (head_idx < num_heads) {
    // q
    using LoadT = AlignedVector<int, VecSize>;
    using LoadBiasT = AlignedVector<T, VecSize>;
    using LoadOutScaleT = AlignedVector<float, VecSize>;
    constexpr int HalfVecSize = VecSize / 2;
    using LoadEmbT = AlignedVector<float, VecSize>;

    LoadT left_vec;
    LoadT right_vec;
    LoadBiasT left_bias_vec;
    LoadBiasT right_bias_vec;
    LoadOutScaleT left_out_scale_vec;
    LoadOutScaleT right_out_scale_vec;
    LoadEmbT cos_emb_vec;
    LoadEmbT sin_emb_vec;
    const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
    T* qkv_out_now = qkv_out + start_token_idx * hidden_size;
#pragma unroll
    for (uint32_t head_bias = lane_id * VecSize; head_bias < half_head_size;
         head_bias += 32 * VecSize) {
      const int bias_idx_left = head_idx * HeadDim + head_bias;
      const int bias_idx_right = bias_idx_left + half_head_size;
      Load<int, VecSize>(&qkv_now[bias_idx_left], &left_vec);
      Load<int, VecSize>(&qkv_now[bias_idx_right], &right_vec);

      if (qkv_biases) {
        Load<T, VecSize>(&qkv_biases[bias_idx_left], &left_bias_vec);
        Load<T, VecSize>(&qkv_biases[bias_idx_right], &right_bias_vec);
      }
      Load<float, VecSize>(&qkv_out_scales[bias_idx_left], &left_out_scale_vec);
      Load<float, VecSize>(&qkv_out_scales[bias_idx_right],
                           &right_out_scale_vec);
      // q rope
      const uint32_t emb_idx = write_seq_id * HeadDim + head_bias;
      Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        // dequant + add_bias + rope
        float input_left = static_cast<float>(left_vec[i]);
        float input_right = static_cast<float>(right_vec[i]);
        input_left = qkv_biases ? input_left * left_out_scale_vec[i] +
                                      static_cast<float>(left_bias_vec[i])
                                : input_left * left_out_scale_vec[i];
        input_right = qkv_biases ? input_right * right_out_scale_vec[i] +
                                       static_cast<float>(right_bias_vec[i])
                                 : input_right * right_out_scale_vec[i];
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        left_bias_vec[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        right_bias_vec[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
      Store<T, VecSize>(left_bias_vec, &qkv_out_now[bias_idx_left]);
      Store<T, VecSize>(right_bias_vec, &qkv_out_now[bias_idx_right]);
    }
  } else if (head_idx < num_heads + 2 * kv_num_heads) {
    // k
    constexpr int KV_VEC_SIZE = 16 / sizeof(uint8_t);  // 16
    using LoadPadKVT = AlignedVector<uint8_t, KV_VEC_SIZE>;
    const uint32_t kv_head_idx = (head_idx - num_heads) % kv_num_heads;
    if (block_offset == 0) {
      // pad zero for this kv_head_idx for this block
      LoadPadKVT pad_cache_vec;
      *(reinterpret_cast<uint4*>(pad_cache_vec.val)) = make_uint4(0, 0, 0, 0);
      if (head_idx < num_heads + kv_num_heads) {
        constexpr int num_vecs_per_head_dim = half_head_size / KV_VEC_SIZE;
        constexpr int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx = (block_idx * kv_num_heads + kv_head_idx) *
                                     block_size * half_head_size +
                                 lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim;
             block_i < block_size;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &key_cache[tgt_idx + block_i * half_head_size]);
        }
      } else {
        const int num_vecs_per_head_dim = half_block_size / KV_VEC_SIZE;
        const int num_token_each_time = 32 / num_vecs_per_head_dim;
        const uint32_t tgt_idx = (block_idx * kv_num_heads + kv_head_idx) *
                                     HeadDim * half_block_size +
                                 lane_id % num_vecs_per_head_dim * KV_VEC_SIZE;
        for (int block_i = lane_id / num_vecs_per_head_dim; block_i < HeadDim;
             block_i += num_token_each_time) {
          Store<uint8_t, KV_VEC_SIZE>(
              pad_cache_vec, &value_cache[tgt_idx + block_i * half_block_size]);
        }
      }
    }
    if (head_idx < num_heads + kv_num_heads) {
      const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
      if (head_bias < half_head_size) {
        constexpr int K_VEC_SIZE = 4;
        constexpr int HALF_K_VEC_SIZE = 2;
        using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
        using LoadT = AlignedVector<int, HALF_K_VEC_SIZE>;
        using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
        using LoadOutScaleT = AlignedVector<float, HALF_K_VEC_SIZE>;
        using LoadScaleT = AlignedVector<T, HALF_K_VEC_SIZE>;
        using LoadEmbT = AlignedVector<float, HALF_K_VEC_SIZE>;
        LoadT left_src_vec1, left_src_vec2, right_src_vec1, right_src_vec2;
        LoadBiasT left_bias_vec1, left_bias_vec2, right_bias_vec1,
            right_bias_vec2;
        LoadOutScaleT left_out_scale_vec1, left_out_scale_vec2,
            right_out_scale_vec1, right_out_scale_vec2;
        LoadScaleT left_scale_vec1, left_scale_vec2, right_scale_vec1,
            right_scale_vec2;
        LoadScaleT left_zp_vec1, left_zp_vec2, right_zp_vec1, right_zp_vec2;
        LoadEmbT cos_emb_vec1, cos_emb_vec2;
        LoadEmbT sin_emb_vec1, sin_emb_vec2;

        const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
        const int left_bias_idx = head_idx * HeadDim + head_bias;
        const int right_bias_idx = left_bias_idx + half_head_size;

        const uint32_t left_cache_idx = kv_head_idx * HeadDim + head_bias;
        const uint32_t right_cache_idx = left_cache_idx + half_head_size;

        Load<int, HALF_K_VEC_SIZE>(&qkv_now[left_bias_idx], &left_src_vec1);
        Load<int, HALF_K_VEC_SIZE>(&qkv_now[left_bias_idx + 8], &left_src_vec2);
        Load<int, HALF_K_VEC_SIZE>(&qkv_now[right_bias_idx], &right_src_vec1);
        Load<int, HALF_K_VEC_SIZE>(&qkv_now[right_bias_idx + 8],
                                   &right_src_vec2);
        if (qkv_biases) {
          Load<T, HALF_K_VEC_SIZE>(&qkv_biases[left_bias_idx], &left_bias_vec1);
          Load<T, HALF_K_VEC_SIZE>(&qkv_biases[left_bias_idx + 8],
                                   &left_bias_vec2);
          Load<T, HALF_K_VEC_SIZE>(&qkv_biases[right_bias_idx],
                                   &right_bias_vec1);
          Load<T, HALF_K_VEC_SIZE>(&qkv_biases[right_bias_idx + 8],
                                   &right_bias_vec2);
        }
        Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[left_bias_idx],
                                     &left_out_scale_vec1);
        Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[left_bias_idx + 8],
                                     &left_out_scale_vec2);
        Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[right_bias_idx],
                                     &right_out_scale_vec1);
        Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[right_bias_idx + 8],
                                     &right_out_scale_vec2);

        const uint32_t emb_idx = write_seq_id * HeadDim + head_bias;
        Load<float, HALF_K_VEC_SIZE>(&cos_emb[emb_idx], &cos_emb_vec1);
        Load<float, HALF_K_VEC_SIZE>(&cos_emb[emb_idx + 8], &cos_emb_vec2);
        Load<float, HALF_K_VEC_SIZE>(&sin_emb[emb_idx], &sin_emb_vec1);
        Load<float, HALF_K_VEC_SIZE>(&sin_emb[emb_idx + 8], &sin_emb_vec2);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[left_cache_idx],
                                 &left_scale_vec1);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[left_cache_idx + 8],
                                 &left_scale_vec2);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[left_cache_idx],
                                 &left_zp_vec1);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[left_cache_idx + 8],
                                 &left_zp_vec2);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[right_cache_idx],
                                 &right_scale_vec1);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_scale[right_cache_idx + 8],
                                 &right_scale_vec2);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[right_cache_idx],
                                 &right_zp_vec1);
        Load<T, HALF_K_VEC_SIZE>(&cache_k_zero_points[right_cache_idx + 8],
                                 &right_zp_vec2);

        for (int i = 0; i < HALF_K_VEC_SIZE; i++) {
          float input_left = static_cast<float>(left_src_vec1[i]);
          float input_right = static_cast<float>(right_src_vec1[i]);
          input_left = qkv_biases ? input_left * left_out_scale_vec1[i] +
                                        static_cast<float>(left_bias_vec1[i])
                                  : input_left * left_out_scale_vec1[i];
          input_right = qkv_biases ? input_right * right_out_scale_vec1[i] +
                                         static_cast<float>(right_bias_vec1[i])
                                   : input_right * right_out_scale_vec1[i];
          float cos_tmp = cos_emb_vec1[0];
          float sin_tmp = sin_emb_vec1[0];
          left_bias_vec1[i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          right_bias_vec1[i] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);


          input_left = static_cast<float>(left_src_vec2[i]);
          input_right = static_cast<float>(right_src_vec2[i]);
          cos_tmp = cos_emb_vec2[i];
          sin_tmp = sin_emb_vec2[i];
          left_bias_vec2[i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          right_bias_vec2[i] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
          // quant + write k
        }

        LoadKVResT left_cache_vec, right_cache_vec;
        const int left_start_block_16 =
            block_offset / 16 * 16 + block_offset % 8 + lane_id / 4 % 4 / 2 * 8;
        const uint32_t left_tgt_cache_idx =
            block_idx * kv_num_heads * block_size * half_head_size +
            kv_head_idx * block_size * half_head_size +
            left_start_block_16 * half_head_size + lane_id / 4 / 4 * 32 +
            lane_id / 4 % 2 * 16 + lane_id % 4 * 4;

        const int right_lane_id = lane_id + 16;
        const int right_start_block_16 = block_offset / 16 * 16 +
                                         block_offset % 8 +
                                         right_lane_id / 4 % 4 / 2 * 8;
        const uint32_t right_tgt_cache_idx =
            block_idx * kv_num_heads * block_size * half_head_size +
            kv_head_idx * block_size * half_head_size +
            right_start_block_16 * half_head_size + right_lane_id / 4 / 4 * 32 +
            right_lane_id / 4 % 2 * 16 + right_lane_id % 4 * 4;
        Load<uint8_t, K_VEC_SIZE>(&key_cache[left_tgt_cache_idx],
                                  &left_cache_vec);
        Load<uint8_t, K_VEC_SIZE>(&key_cache[right_tgt_cache_idx],
                                  &right_cache_vec);

#pragma unroll
        for (uint32_t i = 0; i < HALF_K_VEC_SIZE; i++) {
          float quant_value1 = static_cast<float>(
              left_scale_vec1[i] * left_bias_vec1[i] + left_zp_vec1[i]);
          float quant_value2 = static_cast<float>(
              left_scale_vec2[i] * left_bias_vec2[i] + left_zp_vec2[i]);
          if constexpr (RoundType == 0) {
            quant_value1 = roundWithTiesToEven(quant_value1);
            quant_value2 = roundWithTiesToEven(quant_value2);
          } else {
            quant_value1 = round(quant_value1);
            quant_value2 = round(quant_value2);
          }
          quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
          quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
          quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
          quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
          uint8_t uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
          uint8_t uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
          uint8_t ano_uint_quant_value = 0;
          if (block_offset % 16 / 8 == 0) {
            left_cache_vec[i] |=
                ((ano_uint_quant_value) | (uint_quant_value1 & 0x0F));
            left_cache_vec[i + HALF_K_VEC_SIZE] |=
                ((ano_uint_quant_value) | (uint_quant_value2 & 0x0F));
          } else {
            left_cache_vec[i] |=
                ((uint_quant_value1 << 4) | (ano_uint_quant_value));
            left_cache_vec[i + HALF_K_VEC_SIZE] |=
                ((uint_quant_value2 << 4) | (ano_uint_quant_value));
          }

          quant_value1 = static_cast<float>(
              right_scale_vec1[i] * right_bias_vec1[i] + right_zp_vec1[i]);
          quant_value2 = static_cast<float>(
              right_scale_vec2[i] * right_bias_vec2[i] + right_zp_vec2[i]);
          if constexpr (RoundType == 0) {
            quant_value1 = roundWithTiesToEven(quant_value1);
            quant_value2 = roundWithTiesToEven(quant_value2);
          } else {
            quant_value1 = round(quant_value1);
            quant_value2 = round(quant_value2);
          }
          quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
          quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
          quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
          quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
          uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
          uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
          ano_uint_quant_value = 0;
          if (block_offset % 16 / 8 == 0) {
            right_cache_vec[i] |=
                ((ano_uint_quant_value) | (uint_quant_value1 & 0x0F));
            right_cache_vec[i + HALF_K_VEC_SIZE] |=
                ((ano_uint_quant_value) | (uint_quant_value2 & 0x0F));
          } else {
            right_cache_vec[i] |=
                ((uint_quant_value1 << 4) | (ano_uint_quant_value));
            right_cache_vec[i + HALF_K_VEC_SIZE] |=
                ((uint_quant_value2 << 4) | (ano_uint_quant_value));
          }
        }
        Store<uint8_t, K_VEC_SIZE>(left_cache_vec,
                                   &key_cache[left_tgt_cache_idx]);
        Store<uint8_t, K_VEC_SIZE>(right_cache_vec,
                                   &key_cache[right_tgt_cache_idx]);
      }
    } else {
      constexpr int K_VEC_SIZE = 4;
      constexpr int HALF_K_VEC_SIZE = 2;
      using LoadKVResT = AlignedVector<uint8_t, K_VEC_SIZE>;
      using LoadT = AlignedVector<int, HALF_K_VEC_SIZE>;
      using LoadBiasT = AlignedVector<T, HALF_K_VEC_SIZE>;
      using LoadOutScaleT = AlignedVector<float, HALF_K_VEC_SIZE>;
      using LoadScaleT = AlignedVector<T, HALF_K_VEC_SIZE>;
      LoadT src_vec1, src_vec2;
      LoadBiasT bias_vec1, bias_vec2;
      LoadOutScaleT out_scale_vec1, out_scale_vec2;
      LoadScaleT scale_vec1, scale_vec2;
      LoadScaleT zp_vec1, zp_vec2;

      const int* qkv_now = quant_qkv + start_token_idx * hidden_size;
      const int head_bias = lane_id / 4 * 16 + lane_id % 4 * 2;
      const uint32_t cache_idx = kv_head_idx * HeadDim + head_bias;
      const int bias_idx = head_idx * HeadDim + head_bias;
      Load<int, HALF_K_VEC_SIZE>(&qkv_now[bias_idx], &src_vec1);
      Load<int, HALF_K_VEC_SIZE>(&qkv_now[bias_idx + 8], &src_vec2);
      if (qkv_biases) {
        Load<T, HALF_K_VEC_SIZE>(&qkv_biases[bias_idx], &bias_vec1);
        Load<T, HALF_K_VEC_SIZE>(&qkv_biases[bias_idx + 8], &bias_vec2);
      }
      Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[bias_idx], &out_scale_vec1);
      Load<float, HALF_K_VEC_SIZE>(&qkv_out_scales[bias_idx + 8],
                                   &out_scale_vec2);

      Load<T, HALF_K_VEC_SIZE>(&cache_v_scale[cache_idx], &scale_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_scale[cache_idx + 8], &scale_vec2);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_zero_points[cache_idx], &zp_vec1);
      Load<T, HALF_K_VEC_SIZE>(&cache_v_zero_points[cache_idx + 8], &zp_vec2);

      float input_left = static_cast<float>(src_vec1[0]);
      float input_right = static_cast<float>(src_vec1[1]);
      input_left = qkv_biases ? input_left * out_scale_vec1[0] +
                                    static_cast<float>(bias_vec1[0])
                              : input_left * out_scale_vec1[0];
      input_right = qkv_biases ? input_right * out_scale_vec2[1] +
                                     static_cast<float>(bias_vec1[1])
                               : input_right * out_scale_vec2[1];

      bias_vec1[0] = static_cast<T>(input_left);
      bias_vec1[1] = static_cast<T>(input_right);

      input_left = static_cast<float>(src_vec2[0]);
      input_right = static_cast<float>(src_vec2[1]);
      input_left = qkv_biases ? input_left * out_scale_vec1[0] +
                                    static_cast<float>(bias_vec2[0])
                              : input_left * out_scale_vec1[0];
      input_right = qkv_biases ? input_right * out_scale_vec1[1] +
                                     static_cast<float>(bias_vec2[1])
                               : input_right * out_scale_vec1[1];

      bias_vec2[0] = static_cast<T>(input_left);
      bias_vec2[1] = static_cast<T>(input_right);

      const uint32_t base_tgt_cache_idx =
          block_idx * kv_num_heads * HeadDim * half_block_size +
          kv_head_idx * HeadDim * half_block_size +
          (lane_id / 4 * 16 + lane_id % 4 * 2) * half_block_size +
          block_offset / 16 % 4 / 2 * 8 * half_block_size +
          block_offset / 16 / 4 * 32 + block_offset / 16 % 2 * 16;
      const uint32_t tgt_cache_idx1 = base_tgt_cache_idx +
                                      block_offset % 8 / 2 * 4     // per 4
                                      + block_offset % 16 / 8 * 2  // per 2
                                      + block_offset % 2;          // per 1
      const uint32_t tgt_cache_idx2 = tgt_cache_idx1 + half_block_size;

      float quant_value1 =
          static_cast<float>(scale_vec1[0] * bias_vec1[0] + zp_vec1[0]);
      float quant_value2 =
          static_cast<float>(scale_vec2[0] * bias_vec2[0] + zp_vec2[0]);
      if constexpr (RoundType == 0) {
        quant_value1 = roundWithTiesToEven(quant_value1);
        quant_value2 = roundWithTiesToEven(quant_value2);
      } else {
        quant_value1 = round(quant_value1);
        quant_value2 = round(quant_value2);
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      uint8_t uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
      uint8_t uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
      value_cache[tgt_cache_idx1] =
          (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);

      quant_value1 =
          static_cast<float>(scale_vec1[1] * bias_vec1[1] + zp_vec1[1]);
      quant_value2 =
          static_cast<float>(scale_vec2[1] * bias_vec2[1] + zp_vec2[1]);
      if constexpr (RoundType == 0) {
        quant_value1 = roundWithTiesToEven(quant_value1);
        quant_value2 = roundWithTiesToEven(quant_value2);
      } else {
        quant_value1 = round(quant_value1);
        quant_value2 = round(quant_value2);
      }
      quant_value1 = quant_value1 > max_bound ? max_bound : quant_value1;
      quant_value1 = quant_value1 < min_bound ? min_bound : quant_value1;
      quant_value2 = quant_value2 > max_bound ? max_bound : quant_value2;
      quant_value2 = quant_value2 < min_bound ? min_bound : quant_value2;
      uint_quant_value1 = static_cast<uint8_t>(quant_value1 + 8.0f);
      uint_quant_value2 = static_cast<uint8_t>(quant_value2 + 8.0f);
      value_cache[tgt_cache_idx2] =
          (uint_quant_value2 << 4) | (uint_quant_value1 & 0x0F);
    }
  }
}