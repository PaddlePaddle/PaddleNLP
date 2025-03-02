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
#include "utils.cuh"

template <typename T, int VecSize = 1>
__global__ void decode_absorb_cache_kernel(
    const T* __restrict__ kv_nope,  // [bsz, kv_num_heads, pe_size] 512
    const T* __restrict__ kv_pe,  // [bsz, kv_num_heads, nope_size] 64
    T* __restrict__ kv_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // nope_size]                             
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_encoder,  // [bsz]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int nope_size,
    const int pe_size,
    const int block_size,
    const uint32_t elem_cnt) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  LoadT src_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t nope_hidden_size = kv_num_heads * nope_size;
  const uint32_t pe_hidden_size = kv_num_heads * pe_size;
  const uint32_t all_size = nope_size + pe_size;
  const int64_t hidden_size = nope_hidden_size + pe_hidden_size;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int ori_bi = linear_index / hidden_size;
    const int bias = linear_index % hidden_size;
    const int start_token_idx = ori_bi * max_seq_len - cum_offsets[ori_bi];
    if (seq_lens_encoder[ori_bi] > 0) return;
    const int write_seq_id = seq_lens[ori_bi];
    
    if (write_seq_id == 0) continue;

    const int* block_table_now = nullptr;

    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[write_seq_id / block_size];
    const int block_offset = write_seq_id % block_size;

    if (bias < nope_hidden_size) { // pe
      const uint32_t inner_bias = bias;
      const uint32_t hi = inner_bias / nope_size;
      const uint32_t h_bias = inner_bias % nope_size;
      const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                             hi * block_size * all_size +
                             block_offset * all_size + h_bias;
      const uint32_t ori_idx =
          start_token_idx * nope_hidden_size + inner_bias;
      Load<T, VecSize>(&kv_nope[ori_idx], &src_vec);
      Store<T, VecSize>(src_vec, &kv_cache[tgt_idx]);
    } else {
      const uint32_t inner_bias = bias - nope_hidden_size;
      const uint32_t hi = inner_bias / pe_size;
      const uint32_t h_bias = inner_bias % pe_size;
      const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                             hi * block_size * all_size +
                             block_offset * all_size + nope_size + h_bias;
      const uint32_t ori_idx =
          start_token_idx * pe_hidden_size + inner_bias;
      Load<T, VecSize>(&kv_pe[ori_idx], &src_vec);
      Store<T, VecSize>(src_vec, &kv_cache[tgt_idx]);
    }
  }
}

template <typename T, int VecSize = 1>
__global__ void prefill_absorb_cache_kernel(
    const T* __restrict__ kv_nope,  // [bsz, kv_num_heads, pe_size] 512
    const T* __restrict__ kv_pe,  // [bsz, kv_num_heads, nope_size] 64
    T* __restrict__ kv_cache,    // [num_blocks, kv_num_heads, block_size,
                                  // nope_size]                             
    const int* __restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int* __restrict__ padding_offsets,
    const int* __restrict__ cum_offsets,
    const int* __restrict__ seq_lens,          // [bsz]
    const int* __restrict__ seq_lens_decoder,  // [bsz]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int kv_num_heads,
    const int nope_size,
    const int pe_size,
    const int block_size,
    const uint32_t elem_cnt) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t nope_hidden_size = kv_num_heads * nope_size;
  const uint32_t pe_hidden_size = kv_num_heads * pe_size;
  const uint32_t all_size = nope_size + pe_size;
  const int64_t hidden_size = nope_hidden_size + pe_hidden_size;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const uint32_t token_idx = linear_index / hidden_size;
    const uint32_t bias = linear_index % hidden_size;
    const uint32_t ori_token_idx = token_idx + padding_offsets[token_idx];
    const uint32_t ori_bi = ori_token_idx / max_seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const uint32_t ori_seq_id =
        ori_token_idx % max_seq_len + seq_lens_decoder[ori_bi];  

    const int* block_table_now = nullptr;
    block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const uint32_t block_idx = block_table_now[ori_seq_id / block_size];
    const uint32_t block_offset = ori_seq_id % block_size;

    if (bias < nope_hidden_size) { // pe
      const uint32_t inner_bias = bias;
      const uint32_t hi = inner_bias / nope_size;
      const uint32_t h_bias = inner_bias % nope_size;
      const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                             hi * block_size * all_size +
                             block_offset * all_size + h_bias;
      const uint32_t ori_idx =
          token_idx * nope_hidden_size + inner_bias;
      Load<T, VecSize>(&kv_nope[ori_idx], &src_vec);
      Store<T, VecSize>(src_vec, &kv_cache[tgt_idx]);
    } else {
      const uint32_t inner_bias = bias - nope_hidden_size;
      const uint32_t hi = inner_bias / pe_size;
      const uint32_t h_bias = inner_bias % pe_size;
      const uint32_t tgt_idx = block_idx * kv_num_heads * block_size * all_size +
                             hi * block_size * all_size +
                             block_offset * all_size + nope_size + h_bias;
      const uint32_t ori_idx =
          token_idx * pe_hidden_size + inner_bias;
      Load<T, VecSize>(&kv_pe[ori_idx], &src_vec);
      Store<T, VecSize>(src_vec, &kv_cache[tgt_idx]);
    }
  }
}