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

#include "append_attention_func.cuh"
#include "append_attention_kernel.h"

template <typename T,
          typename CacheT,
          bool partition_kv,
          uint32_t GROUP_SIZE,
          bool CAUSAL,
          uint32_t NUM_WARPS,
          uint32_t NUM_WARP_Q,
          uint32_t NUM_WARP_KV,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t num_frags_x,
          uint32_t num_frags_z,
          uint32_t num_frags_y,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
__global__ void multi_query_append_attention_c4_kernel(
    T *__restrict__ q,             // [token_num, (num_heads + 2* kv_num_head) * head_dim]
    CacheT *__restrict__ cache_k,  // [max_block_num, num_heads, block_size,
                                   // head_dim]
    CacheT *__restrict__ cache_v,
    const T *__restrict__ cache_k_scale,       // [num_kv_heads, head_dim]
    const T *__restrict__ cache_k_zero_point,  // [num_kv_heads, head_dim]
    const T *__restrict__ cache_v_scale,       // [num_kv_heads, head_dim]
    const T *__restrict__ cache_v_zero_point,  // [num_kv_heads, head_dim]
    const T *__restrict__ shift_bias,          // [q_num_heads * HEAD_DIM]
    const T *__restrict__ smooth_weight,       // [q_num_heads * HEAD_DIM]
    const int *__restrict__ seq_lens,
    const int *__restrict__ seq_lens_kv,
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids_per_batch,
    const int *__restrict__ cum_offsets,
    const int *__restrict__ block_table,  // [bsz, block_num_per_seq]
    const int max_seq_len,
    const int max_dec_len,
    const int max_block_num_per_seq,
    const float scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const uint32_t chunk_size,
    T *__restrict__ tmp_workspace,  // split kv [token_num, num_chunks,
                                    // num_heads, head_dim]
    float *__restrict__ tmp_m,      // [token_num, num_chunks, num_heads]
    float *__restrict__ tmp_d,      // [token_num, num_chunks, num_heads]
    OutT *__restrict__ out,
    const int speculate_max_draft_token_num = 5) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  constexpr uint32_t num_vecs_per_head_k =
      HEAD_DIM / 2 / num_elems_per_128b<CacheT>();
  constexpr uint32_t num_vecs_per_blocksize =
      BLOCK_SIZE / 2 / num_elems_per_128b<CacheT>();
  constexpr uint32_t inv_k_stride = 8 / num_vecs_per_head_k;
  constexpr uint32_t inv_v_stride = 8 / num_vecs_per_blocksize;
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t kv_num_heads = gridDim.z;
  const uint32_t q_num_heads = kv_num_heads * GROUP_SIZE;
  const uint32_t q_head_idx = kv_head_idx * GROUP_SIZE;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_idx = blockIdx.y;

  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids_per_batch[btid];
  const uint32_t num_rows_per_block = NUM_WARPS * num_frags_x * 16;
  const int *block_table_now = nullptr;

  block_table_now = block_table + batch_id * max_block_num_per_seq;

  const uint32_t q_len = seq_lens[batch_id];
  if (q_len <= 0) {
    return;
  }
  const uint32_t q_end =
      min(q_len, div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE));
  uint32_t kv_len = seq_lens_kv[batch_id];
  if (ENABLE_PREFILL) {
    kv_len += q_len;
    if (kv_len <= 0) {
      return;
    }
  } else {
    if (kv_len <= 0) {
      return;
    }
    kv_len += q_len;
  }
  const uint32_t num_chunks_this_seq = div_up(kv_len, chunk_size);
  if (chunk_idx >= num_chunks_this_seq) {
    return;
  }

  const uint32_t chunk_start = partition_kv ? chunk_idx * chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min(kv_len, chunk_start + chunk_size) : kv_len;
  const uint32_t chunk_len = chunk_end - chunk_start;

  extern __shared__ uint8_t smem[];
  float s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  float m_frag[num_frags_x][2];
  float d_frag[num_frags_x][2];

  const T *cache_k_scale_now = cache_k_scale + kv_head_idx * HEAD_DIM;
  const T *cache_k_zp_now = cache_k_zero_point + kv_head_idx * HEAD_DIM;
  const T *cache_v_scale_now = cache_v_scale + kv_head_idx * HEAD_DIM;
  const T *cache_v_zp_now = cache_v_zero_point + kv_head_idx * HEAD_DIM;
  T *cache_k_scale_smem = reinterpret_cast<T *>(
      smem + NUM_WARPS * num_frags_x * 16 * HEAD_DIM * sizeof(T) +
      num_frags_z * 16 * HEAD_DIM / 2 * sizeof(CacheT) * 2);
  T *cache_k_zero_point_smem = cache_k_scale_smem + HEAD_DIM;
  T *cache_v_scale_smem = cache_k_zero_point_smem + HEAD_DIM;
  T *cache_v_zero_point_smem = cache_v_scale_smem + HEAD_DIM;
#pragma unroll
  for (uint32_t i = wid * 32 + tid; i < HEAD_DIM; i += 128) {
    cache_k_scale_smem[i] = cache_k_scale_now[i];
    cache_k_zero_point_smem[i] = cache_k_zp_now[i];
    cache_v_scale_smem[i] = cache_v_scale_now[i];
    cache_v_zero_point_smem[i] = cache_v_zp_now[i];
  }

  init_states<T, num_frags_x, num_frags_y>(o_frag, m_frag, d_frag);

  const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
  const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
  const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM / 2;
  const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM / 2;
  const uint32_t kv_b_stride = HEAD_DIM / 2;
  const uint32_t kv_d_stride = BLOCK_SIZE / 2;
  const uint32_t q_start_seq_id =
      batch_id * max_seq_len - __ldg(&cum_offsets[batch_id]);
  const uint32_t q_base_seq_id_this_block =
      (tile_id * NUM_WARPS + wid) * num_frags_x * 16;
  const uint32_t q_offset = q_start_seq_id * q_ori_n_stride +
                            q_head_idx * HEAD_DIM +
                            tid % 8 * num_elems_per_128b<T>();
  const uint32_t o_offset = q_start_seq_id * q_n_stride +
                            q_head_idx * HEAD_DIM +
                            tid % 8 * num_elems_per_128b<T>();
  T *q_base_ptr = q + q_offset;

  T *o_base_ptr_T = nullptr;
  OutT *o_base_ptr_int8 = nullptr;
  if constexpr (partition_kv) {
    if (ENABLE_PREFILL) {
      o_base_ptr_T = tmp_workspace + q_start_seq_id * num_chunks * q_n_stride +
                     chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
                     tid % 8 * num_elems_per_128b<T>();
    } else {
      o_base_ptr_T =
          tmp_workspace +
          batch_id * speculate_max_draft_token_num * num_chunks * q_n_stride +
          chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
          tid % 8 * num_elems_per_128b<T>();
    }
  } else {
    o_base_ptr_int8 = out + o_offset;
  }
  smem_t qo_smem(smem);

  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_x * 16 + tid % 16, tid / 16);
  load_q_global_smem<GROUP_SIZE, num_frags_x, num_frags_y, HEAD_DIM, T>(
      q_base_ptr,
      &qo_smem,
      q_base_seq_id_this_block,
      q_end,
      q_ori_n_stride,
      HEAD_DIM);
  commit_group();
  wait_group<0>();
  __syncthreads();

  q_smem_inplace_multiply_sm_scale<num_frags_x, num_frags_y, T>(&qo_smem,
                                                                scale);

  T cache_k_scale_frag[num_frags_y][4];
  T cache_k_zp_frag[num_frags_y][4];
  T magic_number;
  if constexpr (std::is_same<T, half>::value) {
    magic_number = static_cast<T>(1032.f);
  } else {
    magic_number = static_cast<T>(136.f);
  }
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    *(reinterpret_cast<uint32_t *>(&cache_k_scale_frag[fy][0])) =
        *(reinterpret_cast<uint32_t *>(&cache_k_scale_smem[fy * 16]) + tid % 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_scale_frag[fy][2])) =
        *(reinterpret_cast<uint32_t *>(&cache_k_scale_smem[fy * 16]) + tid % 4 +
          4);
    *(reinterpret_cast<uint32_t *>(&cache_k_zp_frag[fy][0])) =
        *(reinterpret_cast<uint32_t *>(&cache_k_zero_point_smem[fy * 16]) +
          tid % 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_zp_frag[fy][2])) =
        *(reinterpret_cast<uint32_t *>(&cache_k_zero_point_smem[fy * 16]) +
          tid % 4 + 4);
#pragma unroll
    for (uint32_t zp_i = 0; zp_i < 4; ++zp_i) {
      cache_k_zp_frag[fy][zp_i] += magic_number;  // 128 + 8 
    }
  }
  T cache_v_scale_frag[num_frags_y][2];
  T cache_v_zp_frag[num_frags_y][2];
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    cache_v_scale_frag[fy][0] = cache_v_scale_smem[fy * 16 + tid / 4];
    cache_v_scale_frag[fy][1] = cache_v_scale_smem[fy * 16 + tid / 4 + 8];
    cache_v_zp_frag[fy][0] =
        cache_v_zero_point_smem[fy * 16 + tid / 4] + magic_number;
    cache_v_zp_frag[fy][1] =
        cache_v_zero_point_smem[fy * 16 + tid / 4 + 8] + magic_number;
  }

  smem_t k_smem(smem + NUM_WARPS * num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + NUM_WARPS * num_frags_x * 16 * HEAD_DIM * sizeof(T) +
             num_frags_z * 16 * HEAD_DIM / 2 * sizeof(CacheT));


  const uint32_t num_iterations = div_up(
      CAUSAL
          ? (min(chunk_len,
                 sub_if_greater_or_zero(
                     kv_len - q_len +
                         div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE),
                     chunk_start)))
          : chunk_len,
      num_frags_z * 16);
  const uint32_t mask_check_iteration =
      (CAUSAL ? (min(chunk_len,
                     sub_if_greater_or_zero(
                         kv_len - q_len +
                             tile_id * num_rows_per_block / GROUP_SIZE,
                         chunk_start)))
              : chunk_len) /
      (num_frags_z * 16);

  uint32_t k_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t k_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 8 + tid / 4,
          tid %
              4);
  uint32_t v_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 16 + tid / 2, tid % 2);  // 2 * 128 / 8 = 32B, 64 nums

  uint32_t kv_idx_base = chunk_start;
  const uint32_t const_k_offset = kv_head_idx * kv_h_stride +
                                  (wid * 8 + tid / 4) * kv_b_stride +
                                  tid % 4 * num_elems_per_128b<CacheT>();
  const uint32_t const_v_offset = kv_head_idx * kv_h_stride +
                                  (wid * 16 + tid / 2) * kv_d_stride +
                                  tid % 2 * num_elems_per_128b<CacheT>();

  produce_k_blockwise_c4<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(k_smem,
                                     &k_smem_offset_w,
                                     cache_k,
                                     block_table_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end,
                                     const_k_offset);
  commit_group();
  produce_v_blockwise_c4<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(v_smem,
                                     &v_smem_offset_w,
                                     cache_v,
                                     block_table_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_d_stride,
                                     kv_idx_base,
                                     chunk_end,
                                     const_v_offset);
  commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    wait_group<1>();
    __syncthreads();

    compute_qk_c4<num_frags_x, num_frags_y, num_frags_z, T, CacheT>(
        &qo_smem,
        &q_smem_offset_r,
        &k_smem,
        &k_smem_offset_r,
        s_frag,
        cache_k_scale_frag,
        cache_k_zp_frag);

    if (iter >= mask_check_iteration) {
      mask_s<T,
             partition_kv,
             CAUSAL,
             GROUP_SIZE,
             NUM_WARPS,
             num_frags_x,
             num_frags_y,
             num_frags_z>(q_base_seq_id_this_block,
                          kv_idx_base,
                          q_len,
                          kv_len,
                          chunk_end,
                          s_frag);
    }

    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();

    kv_idx_base += num_frags_z * 16;
    produce_k_blockwise_c4<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(k_smem,
                                       &k_smem_offset_w,
                                       cache_k,
                                       block_table_now,
                                       kv_head_idx,
                                       kv_n_stride,
                                       kv_h_stride,
                                       kv_b_stride,
                                       kv_idx_base,
                                       chunk_end,
                                       const_k_offset);
    commit_group();
    wait_group<1>();
    __syncthreads();

    compute_sfm_v_c4<num_frags_x,
                     num_frags_y,
                     num_frags_z,
                     BLOCK_SIZE,
                     T,
                     CacheT>(&v_smem,
                             &v_smem_offset_r,
                             s_frag,
                             o_frag,
                             d_frag,
                             cache_v_scale_frag,
                             cache_v_zp_frag);
    __syncthreads();

    produce_v_blockwise_c4<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(v_smem,
                                       &v_smem_offset_w,
                                       cache_v,
                                       block_table_now,
                                       kv_head_idx,
                                       kv_n_stride,
                                       kv_h_stride,
                                       kv_d_stride,
                                       kv_idx_base,
                                       chunk_end,
                                       const_v_offset);
    commit_group();
  }
  wait_group<0>();
  __syncthreads();

  if constexpr (!partition_kv) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }

  if constexpr (partition_kv) {
    write_o_reg_gmem_shift_smooth_quant<GROUP_SIZE,
                                        num_frags_x,
                                        num_frags_y,
                                        partition_kv>(
        o_frag,
        &qo_smem,
        o_base_ptr_T,
        shift_bias,
        smooth_weight,
        q_base_seq_id_this_block,
        q_head_idx,
        quant_max_bound,
        quant_min_bound,
        in_scale,
        q_len,
        partition_kv ? q_n_stride * num_chunks : q_n_stride,
        HEAD_DIM);
  } else {
    write_o_reg_gmem_shift_smooth_quant<GROUP_SIZE,
                                        num_frags_x,
                                        num_frags_y,
                                        partition_kv>(
        o_frag,
        &qo_smem,
        o_base_ptr_int8,
        shift_bias,
        smooth_weight,
        q_base_seq_id_this_block,
        q_head_idx,
        quant_max_bound,
        quant_min_bound,
        in_scale,
        q_len,
        partition_kv ? q_n_stride * num_chunks : q_n_stride,
        HEAD_DIM);
  }

  if constexpr (partition_kv) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_idx_now =
            q_base_seq_id_this_block + tid / 4 + j * 8 + fx * 16;
        const uint32_t qo_head_idx = q_head_idx + qo_idx_now % GROUP_SIZE;
        const uint32_t qo_idx = q_start_seq_id + qo_idx_now / GROUP_SIZE;
        if (qo_idx - q_start_seq_id < q_len) {
          uint32_t offset;
          if (ENABLE_PREFILL) {
            offset =
                (qo_idx * num_chunks + chunk_idx) * q_num_heads + qo_head_idx;
          } else {
            offset = ((batch_id * speculate_max_draft_token_num +
                       qo_idx_now / GROUP_SIZE) *
                          num_chunks +
                      chunk_idx) *
                         q_num_heads +
                     qo_head_idx;
          }
          tmp_m[offset] = m_frag[fx][j];
          tmp_d[offset] = d_frag[fx][j];
        }
      }
    }
  }
}

template <typename T,
          typename CacheT,
          bool partition_kv,
          uint32_t GROUP_SIZE,
          bool CAUSAL,
          uint32_t NUM_WARPS,
          uint32_t NUM_WARP_Q,
          uint32_t NUM_WARP_KV,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          uint32_t num_frags_x,
          uint32_t num_frags_z,
          uint32_t num_frags_y,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
__global__ void multi_query_append_attention_c4_warp1_4_kernel(
    T *__restrict__ q,             // [token_num, (num_heads + 2* kv_num_head) * head_dim]
    CacheT *__restrict__ cache_k,  // [max_block_num, num_heads, block_size,
                                   // head_dim]
    CacheT *__restrict__ cache_v,
    const T *__restrict__ cache_k_scale,       // [num_kv_heads, head_dim]
    const T *__restrict__ cache_k_zero_point,  // [num_kv_heads, head_dim]
    const T *__restrict__ cache_v_scale,       // [num_kv_heads, head_dim]
    const T *__restrict__ cache_v_zero_point,  // [num_kv_heads, head_dim]
    const T *__restrict__ shift_bias,          // [q_num_heads * HEAD_DIM]
    const T *__restrict__ smooth_weight,       // [q_num_heads * HEAD_DIM]
    const int *__restrict__ seq_lens,
    const int *__restrict__ seq_lens_kv,
    const int *__restrict__ batch_ids,
    const int *__restrict__ tile_ids_per_batch,
    const int *__restrict__ cum_offsets,
    const int *__restrict__ block_table,  // [bsz, block_num_per_seq]
    const int max_seq_len,
    const int max_dec_len,
    const int max_block_num_per_seq,
    const float scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const uint32_t chunk_size,
    T *__restrict__ tmp_workspace,  // split kv [token_num, num_chunks,
                                    // num_heads, head_dim]
    float *__restrict__ tmp_m,      // [token_num, num_chunks, num_heads]
    float *__restrict__ tmp_d,      // [token_num, num_chunks, num_heads]
    OutT *__restrict__ out,
    const int speculate_max_draft_token_num = 5) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
  constexpr uint32_t num_vecs_per_head_k =
      HEAD_DIM / 2 / num_elems_per_128b<CacheT>();
  constexpr uint32_t num_vecs_per_blocksize =
      BLOCK_SIZE / 2 / num_elems_per_128b<CacheT>();
  constexpr uint32_t inv_k_stride = 8 / num_vecs_per_head_k;
  constexpr uint32_t inv_v_stride = 8 / num_vecs_per_blocksize;
  static_assert(NUM_WARP_Q == 1, "NUM_WARP_Q must be 1");
  static_assert(NUM_WARP_KV == 4, "NUM_WARP_KV must be 4");
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t kv_num_heads = gridDim.z;
  const uint32_t q_num_heads = kv_num_heads * GROUP_SIZE;
  const uint32_t q_head_idx = kv_head_idx * GROUP_SIZE;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_idx = blockIdx.y;

  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids_per_batch[btid];
  const uint32_t num_rows_per_block = num_frags_x * 16;
  const int *block_table_now = block_table + batch_id * max_block_num_per_seq;

  const uint32_t q_len = seq_lens[batch_id];
  if (q_len <= 0) {
    return;
  }
  const uint32_t q_end =
      min(q_len, div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE));
  uint32_t kv_len = seq_lens_kv[batch_id];
  if (ENABLE_PREFILL) {
    kv_len += q_len;
    if (kv_len <= 0) {
      return;
    }
  } else {
    if (kv_len <= 0) {
      return;
    }
    kv_len += q_len;
  }
  const uint32_t num_chunks_this_seq = div_up(kv_len, chunk_size);
  if (chunk_idx >= num_chunks_this_seq) {
    return;
  }

  const uint32_t chunk_start = partition_kv ? chunk_idx * chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min(kv_len, chunk_start + chunk_size) : kv_len;
  const uint32_t chunk_len = chunk_end - chunk_start;

  extern __shared__ uint8_t smem[];
  float s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  float m_frag[num_frags_x][2];
  float d_frag[num_frags_x][2];
  init_states<T, num_frags_x, num_frags_y>(o_frag, m_frag, d_frag);

  const T *cache_k_scale_now = cache_k_scale + kv_head_idx * HEAD_DIM;
  const T *cache_k_zp_now = cache_k_zero_point + kv_head_idx * HEAD_DIM;
  const T *cache_v_scale_now = cache_v_scale + kv_head_idx * HEAD_DIM;
  const T *cache_v_zp_now = cache_v_zero_point + kv_head_idx * HEAD_DIM;
  T *cache_k_scale_smem = reinterpret_cast<T *>(
      smem + NUM_WARP_Q * num_frags_x * 16 * HEAD_DIM * sizeof(T) +
      NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM / 2 * sizeof(CacheT) * 2);
  T *cache_k_zero_point_smem = cache_k_scale_smem + HEAD_DIM;
  T *cache_v_scale_smem = cache_k_zero_point_smem + HEAD_DIM;
  T *cache_v_zero_point_smem = cache_v_scale_smem + HEAD_DIM;
#pragma unroll
  for (uint32_t i = wid * 32 + tid; i < HEAD_DIM; i += 128) {
    cache_k_scale_smem[i] = cache_k_scale_now[i];
    cache_k_zero_point_smem[i] = cache_k_zp_now[i];
    cache_v_scale_smem[i] = cache_v_scale_now[i];
    cache_v_zero_point_smem[i] = cache_v_zp_now[i];
  }

  const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
  const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
  const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM / 2;
  const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM / 2;
  const uint32_t kv_b_stride = HEAD_DIM / 2;
  const uint32_t kv_d_stride = BLOCK_SIZE / 2;
  const uint32_t q_start_seq_id =
      batch_id * max_seq_len - __ldg(&cum_offsets[batch_id]);
  const uint32_t q_base_seq_id_this_block = tile_id * num_frags_x * 16;
  const uint32_t q_offset = q_start_seq_id * q_ori_n_stride +
                            q_head_idx * HEAD_DIM +
                            tid % 8 * num_elems_per_128b<T>();
  const uint32_t o_offset = q_start_seq_id * q_n_stride +
                            q_head_idx * HEAD_DIM +
                            tid % 8 * num_elems_per_128b<T>();
  T *q_base_ptr = q + q_offset;

  T *o_base_ptr_T = nullptr;
  OutT *o_base_ptr_int8 = nullptr;
  if (num_chunks_this_seq <= 1) {
    o_base_ptr_int8 = out + o_offset;
  } else {
    if (ENABLE_PREFILL) {
      o_base_ptr_T = tmp_workspace + batch_id * num_chunks * q_n_stride +
                     chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
                     tid % 8 * num_elems_per_128b<T>();
    } else {
      o_base_ptr_T =
          tmp_workspace +
          batch_id * speculate_max_draft_token_num * num_chunks * q_n_stride +
          chunk_idx * q_n_stride + q_head_idx * HEAD_DIM +
          tid % 8 * num_elems_per_128b<T>();
    }
  }

  smem_t qo_smem(smem);

  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16, tid / 16);
  load_q_global_smem_multi_warps<GROUP_SIZE,
                                 num_frags_x,
                                 num_frags_y,
                                 HEAD_DIM,
                                 T>(q_base_ptr,
                                    &qo_smem,
                                    q_base_seq_id_this_block,
                                    q_end,
                                    q_ori_n_stride,
                                    HEAD_DIM);
  commit_group();
  wait_group<0>();
  __syncthreads();

  q_smem_inplace_multiply_sm_scale_multi_warps<num_frags_x, num_frags_y, T>(
      &qo_smem, scale);

  T cache_k_scale_frag[num_frags_y][4];
  T cache_k_zp_frag[num_frags_y][4];
  T magic_number;
  if constexpr (std::is_same<T, half>::value) {
    magic_number = static_cast<T>(1032.f);
  } else {
    magic_number = static_cast<T>(136.f);
  }
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    *(reinterpret_cast<uint32_t *>(&cache_k_scale_frag[fy][0])) =
        *(reinterpret_cast<uint32_t *>(&cache_k_scale_smem[fy * 16]) + tid % 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_scale_frag[fy][2])) =
        *(reinterpret_cast<uint32_t *>(&cache_k_scale_smem[fy * 16]) + tid % 4 +
          4);
    *(reinterpret_cast<uint32_t *>(&cache_k_zp_frag[fy][0])) =
        *(reinterpret_cast<uint32_t *>(&cache_k_zero_point_smem[fy * 16]) +
          tid % 4);
    *(reinterpret_cast<uint32_t *>(&cache_k_zp_frag[fy][2])) =
        *(reinterpret_cast<uint32_t *>(&cache_k_zero_point_smem[fy * 16]) +
          tid % 4 + 4);
#pragma unroll
    for (uint32_t zp_i = 0; zp_i < 4; ++zp_i) {
      cache_k_zp_frag[fy][zp_i] += magic_number;
    }
  }
  T cache_v_scale_frag[num_frags_y][2];
  T cache_v_zp_frag[num_frags_y][2];
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
    cache_v_scale_frag[fy][0] = cache_v_scale_smem[fy * 16 + tid / 4];
    cache_v_scale_frag[fy][1] = cache_v_scale_smem[fy * 16 + tid / 4 + 8];
    cache_v_zp_frag[fy][0] =
        cache_v_zero_point_smem[fy * 16 + tid / 4] + magic_number;
    cache_v_zp_frag[fy][1] =
        cache_v_zero_point_smem[fy * 16 + tid / 4 + 8] + magic_number;
  }

  smem_t k_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
             NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM / 2 * sizeof(CacheT));

  const uint32_t num_iterations = div_up(
      CAUSAL
          ? (min(chunk_len,
                 sub_if_greater_or_zero(
                     kv_len - q_len +
                         div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE),
                     chunk_start)))
          : chunk_len,
      NUM_WARP_KV * num_frags_z * 16);
  const uint32_t mask_check_iteration =
      (CAUSAL ? (min(chunk_len,
                     sub_if_greater_or_zero(
                         kv_len - q_len +
                             tile_id * num_rows_per_block / GROUP_SIZE,
                         chunk_start)))
              : chunk_len) /
      (NUM_WARP_KV * num_frags_z * 16);

  uint32_t k_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * num_frags_y * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t k_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 8 + tid / 4,
          tid %
              4);
  uint32_t v_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 16 + tid / 2, tid % 2);

  uint32_t kv_idx_base = chunk_start;
  const uint32_t const_k_offset = kv_head_idx * kv_h_stride +
                                  (wid * 8 + tid / 4) * kv_b_stride +
                                  tid % 4 * num_elems_per_128b<CacheT>();
  const uint32_t const_v_offset = kv_head_idx * kv_h_stride +
                                  (wid * 16 + tid / 2) * kv_d_stride +
                                  tid % 2 * num_elems_per_128b<CacheT>();

  produce_k_blockwise_c4<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(k_smem,
                                     &k_smem_offset_w,
                                     cache_k,
                                     block_table_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end,
                                     const_k_offset);
  commit_group();
  produce_v_blockwise_c4<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(v_smem,
                                     &v_smem_offset_w,
                                     cache_v,
                                     block_table_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_d_stride,
                                     kv_idx_base,
                                     chunk_end,
                                     const_v_offset);
  commit_group();
#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    wait_group<1>();
    __syncthreads();
    compute_qk_c4<num_frags_x, num_frags_y, num_frags_z, T, CacheT>(
        &qo_smem,
        &q_smem_offset_r,
        &k_smem,
        &k_smem_offset_r,
        s_frag,
        cache_k_scale_frag,
        cache_k_zp_frag);
    if (iter >= mask_check_iteration) {
      mask_s<T,
             partition_kv,
             CAUSAL,
             GROUP_SIZE,
             NUM_WARPS,
             num_frags_x,
             num_frags_y,
             num_frags_z>(q_base_seq_id_this_block,
                          kv_idx_base + wid * num_frags_z * 16,
                          q_len,
                          kv_len,
                          chunk_end,
                          s_frag);
    }

    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();

    kv_idx_base += NUM_WARP_KV * num_frags_z * 16;
    produce_k_blockwise_c4<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(k_smem,
                                       &k_smem_offset_w,
                                       cache_k,
                                       block_table_now,
                                       kv_head_idx,
                                       kv_n_stride,
                                       kv_h_stride,
                                       kv_b_stride,
                                       kv_idx_base,
                                       chunk_end,
                                       const_k_offset);
    commit_group();
    wait_group<1>();
    __syncthreads();

    // compute sfm*v
    compute_sfm_v_c4<num_frags_x,
                     num_frags_y,
                     num_frags_z,
                     BLOCK_SIZE,
                     T,
                     CacheT>(&v_smem,
                             &v_smem_offset_r,
                             s_frag,
                             o_frag,
                             d_frag,
                             cache_v_scale_frag,
                             cache_v_zp_frag);
    __syncthreads();

    produce_v_blockwise_c4<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(v_smem,
                                       &v_smem_offset_w,
                                       cache_v,
                                       block_table_now,
                                       kv_head_idx,
                                       kv_n_stride,
                                       kv_h_stride,
                                       kv_d_stride,
                                       kv_idx_base,
                                       chunk_end,
                                       const_v_offset);
    commit_group();
  }
  wait_group<0>();
  __syncthreads();

  merge_block_res_v2<num_frags_x, num_frags_y, T>(
      o_frag, reinterpret_cast<float *>(smem), m_frag, d_frag, wid, tid);

  if (num_chunks_this_seq <= 1) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }

  // write o
  // [num_frags_x, 16, num_frags_y, 16]
  if (num_chunks_this_seq <= 1) {
    write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE,
                                                    num_frags_x,
                                                    num_frags_y,
                                                    false>(
        o_frag,
        &qo_smem,
        o_base_ptr_int8,
        shift_bias,
        smooth_weight,
        q_base_seq_id_this_block,
        q_head_idx,
        quant_max_bound,
        quant_min_bound,
        in_scale,
        q_len,
        q_n_stride,
        HEAD_DIM);
  } else {
    write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE,
                                                    num_frags_x,
                                                    num_frags_y,
                                                    partition_kv>(
        o_frag,
        &qo_smem,
        o_base_ptr_T,
        shift_bias,
        smooth_weight,
        q_base_seq_id_this_block,
        q_head_idx,
        quant_max_bound,
        quant_min_bound,
        in_scale,
        q_len,
        q_n_stride * num_chunks,
        HEAD_DIM);
  }

  if (num_chunks_this_seq > 1) {
    if (wid == 0) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          const uint32_t qo_idx_now =
              q_base_seq_id_this_block + tid / 4 + j * 8 + fx * 16;
          const uint32_t qo_head_idx = q_head_idx + qo_idx_now % GROUP_SIZE;
          const uint32_t qo_idx = q_start_seq_id + qo_idx_now / GROUP_SIZE;
          if (qo_idx - q_start_seq_id < q_len) {
            uint32_t offset;
            if (ENABLE_PREFILL) {
              offset = (batch_id * num_chunks + chunk_idx) * q_num_heads +
                       qo_head_idx;
            } else {
              offset = ((batch_id * speculate_max_draft_token_num +
                         qo_idx_now / GROUP_SIZE) *
                            num_chunks +
                        chunk_idx) *
                           q_num_heads +
                       qo_head_idx;
            }
            tmp_m[offset] = m_frag[fx][j];
            tmp_d[offset] = d_frag[fx][j];
          }
        }
      }
    }
  }
}

template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
void MultiQueryAppendC4Attention(
    const AppendAttnMetaData &meta_data,
    const paddle::Tensor &qkv,
    const paddle::Tensor &cache_k,
    const paddle::Tensor &cache_v,
    const paddle::optional<paddle::Tensor> &attn_mask,
    const paddle::Tensor &cache_k_scale,
    const paddle::Tensor &cache_v_scale,
    const paddle::optional<paddle::Tensor> &cache_k_zp,
    const paddle::optional<paddle::Tensor> &cache_v_zp,
    const paddle::optional<paddle::Tensor> &shift_bias,
    const paddle::optional<paddle::Tensor> &smooth_weight,
    const paddle::Tensor &seq_lens_q,
    const paddle::Tensor &seq_lens_kv,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &padding_offsets,
    const paddle::Tensor &cum_offsets,
    const paddle::Tensor &block_table,
    const paddle::Tensor &batch_ids,
    const paddle::Tensor &tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t &stream,
    paddle::Tensor *out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;

  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_nums;
  auto bsz = meta_data.batch_size;
  auto max_block_num_per_seq = meta_data.max_blocks_per_seq;

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q);
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  auto *allocator = paddle::GetAllocator(qkv.place());

  const float scale = 1.f / sqrt(HEAD_DIM);

  if constexpr (NUM_WARP_Q == 4) {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16;
    constexpr uint32_t smem_size =
        num_warps * num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        num_frags_z * 16 * HEAD_DIM / 2 * sizeof(uint8_t) * 2 +
        HEAD_DIM * 4 * sizeof(T);
    auto split_kv_kernel =
        multi_query_append_attention_c4_kernel<NV_TYPE,
                                               uint8_t,
                                               true,
                                               GROUP_SIZE,
                                               CAUSAL,
                                               num_warps,
                                               NUM_WARP_Q,
                                               NUM_WARP_KV,
                                               HEAD_DIM,
                                               BLOCK_SIZE,
                                               num_frags_x,
                                               num_frags_z,
                                               num_frags_y,
                                               OUT_NV_TYPE,
                                               ENABLE_PREFILL>;
    cudaFuncSetAttribute(split_kv_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    const int dev_id = 0;
    int sm_count;
    int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
    assert(act_blocks_per_sm > 1);
    const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    const float ratio = static_cast<float>(num_blocks_need) /
                        static_cast<float>(num_blocks_per_wave);

    uint32_t chunk_size = get_max_partition_size(bsz);
    if (!is_decoder) {
      chunk_size = max_seq_len;
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);

    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_c4_kernel<NV_TYPE,
                                                 uint8_t,
                                                 false,
                                                 GROUP_SIZE,
                                                 CAUSAL,
                                                 num_warps,
                                                 NUM_WARP_Q,
                                                 NUM_WARP_KV,
                                                 HEAD_DIM,
                                                 BLOCK_SIZE,
                                                 num_frags_x,
                                                 num_frags_z,
                                                 num_frags_y,
                                                 OUT_NV_TYPE,
                                                 ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }
      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(qkv.data<T>())),
          const_cast<uint8_t *>(cache_k.data<uint8_t>()),
          const_cast<uint8_t *>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(cache_k_zp.get().data<T>()))
                     : nullptr,
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(cache_v_zp.get().data<T>()))
                     : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE *>(
                              const_cast<T *>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          quant_max_bound,
          quant_min_bound,
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
          speculate_max_draft_token_num);
    } else {
      phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
      if (ENABLE_PREFILL) {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(token_num * num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(token_num * num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(token_num * num_chunks * num_heads));
      } else {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads));
      }
      split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(qkv.data<T>())),
          const_cast<uint8_t *>(cache_k.data<uint8_t>()),
          const_cast<uint8_t *>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(cache_k_zp.get().data<T>()))
                     : nullptr,
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(cache_v_zp.get().data<T>()))
                     : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE *>(
                              const_cast<T *>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          quant_max_bound,
          quant_min_bound,
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr()),
          static_cast<float *>(tmp_m->ptr()),
          static_cast<float *>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr()),
                static_cast<float *>(tmp_m->ptr()),
                static_cast<float *>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE *>(
                                 const_cast<T *>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE *>(const_cast<T *>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
                quant_max_bound,
                quant_min_bound,
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr()),
                static_cast<float *>(tmp_m->ptr()),
                static_cast<float *>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE *>(
                                 const_cast<T *>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE *>(const_cast<T *>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
                quant_max_bound,
                quant_min_bound,
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                speculate_max_draft_token_num);
      }
    }
  } else {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV * 4;
    constexpr uint32_t smem_size =
        num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM / 2 * sizeof(uint8_t) * 2 +
        HEAD_DIM * 4 * sizeof(T);
    auto split_kv_kernel =
        multi_query_append_attention_c4_warp1_4_kernel<NV_TYPE,
                                                       uint8_t,
                                                       true,
                                                       GROUP_SIZE,
                                                       CAUSAL,
                                                       num_warps,
                                                       NUM_WARP_Q,
                                                       NUM_WARP_KV,
                                                       HEAD_DIM,
                                                       BLOCK_SIZE,
                                                       num_frags_x,
                                                       num_frags_z,
                                                       num_frags_y,
                                                       OUT_NV_TYPE,
                                                       ENABLE_PREFILL>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
    assert(act_blocks_per_sm > 1);
    const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    const float ratio = static_cast<float>(num_blocks_need) /
                        static_cast<float>(num_blocks_per_wave);


    uint32_t chunk_size = get_max_partition_size(bsz);
    if (!is_decoder) {
      chunk_size = max_seq_len;
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_c4_warp1_4_kernel<NV_TYPE,
                                                         uint8_t,
                                                         false,
                                                         GROUP_SIZE,
                                                         CAUSAL,
                                                         num_warps,
                                                         NUM_WARP_Q,
                                                         NUM_WARP_KV,
                                                         HEAD_DIM,
                                                         BLOCK_SIZE,
                                                         num_frags_x,
                                                         num_frags_z,
                                                         num_frags_y,
                                                         OUT_NV_TYPE,
                                                         ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }
      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(qkv.data<T>())),
          const_cast<uint8_t *>(cache_k.data<uint8_t>()),
          const_cast<uint8_t *>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(cache_k_zp.get().data<T>()))
                     : nullptr,
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(cache_v_zp.get().data<T>()))
                     : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE *>(
                              const_cast<T *>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          quant_max_bound,
          quant_min_bound,
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
          speculate_max_draft_token_num);
    } else {
      phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
      if (is_decoder) {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(bsz * num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(bsz * num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(bsz * num_chunks * num_heads));
      } else {
        if (ENABLE_PREFILL) {
          tmp_workspace =
              allocator->Allocate(phi::SizeOf(qkv.dtype()) *
                                  static_cast<size_t>(token_num * num_chunks *
                                                      num_heads * HEAD_DIM));
          tmp_m = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(token_num * num_chunks * num_heads));
          tmp_d = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(token_num * num_chunks * num_heads));
        } else {
          tmp_workspace = allocator->Allocate(
              phi::SizeOf(qkv.dtype()) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads * HEAD_DIM));
          tmp_m = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads));
          tmp_d = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads));
        }
      }
      split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(qkv.data<T>())),
          const_cast<uint8_t *>(cache_k.data<uint8_t>()),
          const_cast<uint8_t *>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(cache_k_zp.get().data<T>()))
                     : nullptr,
          reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(cache_v_zp.get().data<T>()))
                     : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE *>(
                           const_cast<T *>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE *>(
                              const_cast<T *>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          quant_max_bound,
          quant_min_bound,
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr()),
          static_cast<float *>(tmp_m->ptr()),
          static_cast<float *>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr()),
                static_cast<float *>(tmp_m->ptr()),
                static_cast<float *>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE *>(
                                 const_cast<T *>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE *>(const_cast<T *>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
                quant_max_bound,
                quant_min_bound,
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE *>(tmp_workspace->ptr()),
                static_cast<float *>(tmp_m->ptr()),
                static_cast<float *>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE *>(
                                 const_cast<T *>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE *>(const_cast<T *>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE *>(out->data<OutT>()),
                quant_max_bound,
                quant_min_bound,
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                speculate_max_draft_token_num);
      }
    }
  }
}

template <typename T, typename OutT>
void CascadeAppendAttentionC4Kernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,  // [token_num, (num_heads + 2* kv_num_head) * head_dim]
    const paddle::Tensor&
        cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const paddle::Tensor&
        cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>&
        cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const int num_blocks,
    const int block_shape_q,
    const int max_seq_len,
    const int max_dec_len,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  const auto token_num = meta_data.token_nums;
  const auto block_size = meta_data.block_size;
  const auto bsz = meta_data.batch_size;
  const auto num_heads = meta_data.q_num_heads;
  const auto group_size = meta_data.q_num_heads / meta_data.kv_num_heads;
  const auto head_dim = meta_data.head_dims;

  DISPATCH_CAUSAL(
      causal,
      CAUSAL,
      {DISPATCH_ENABLE_PREFILL(
          enable_prefill,
          ENABLE_PREFILL,
          {DISPATCH_GQA_GROUP_SIZE(
              group_size,
              GROUP_SIZE,
              {DISPATCH_HEAD_DIM(
                  head_dim,
                  HEAD_DIM,
                  {DISPATCH_BLOCK_SIZE(
                      block_size,
                      BLOCK_SIZE,
                      {DISPATCH_BLOCKSHAPE_Q(
                          block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, {
                            MultiQueryAppendC4Attention<T,
                                                        GROUP_SIZE,
                                                        HEAD_DIM,
                                                        BLOCK_SIZE,
                                                        CAUSAL,
                                                        BLOCK_SHAPE_Q,
                                                        NUM_WARP_Q,
                                                        OutT,
                                                        ENABLE_PREFILL>(
                                meta_data,
                                qkv,
                                cache_k,
                                cache_v,
                                attn_mask,
                                cache_k_scale.get(),
                                cache_v_scale.get(),
                                cache_k_zp,
                                cache_v_zp,
                                shift_bias,
                                smooth_weight,
                                seq_lens_q,
                                seq_lens_kv,
                                seq_lens_encoder,
                                padding_offsets,
                                cum_offsets,
                                block_table,
                                batch_ids,
                                tile_ids_per_batch,
                                num_blocks,
                                max_seq_len,
                                max_dec_len,
                                quant_max_bound,
                                quant_min_bound,
                                in_scale,
                                speculate_max_draft_token_num,
                                is_decoder,
                                stream,
                                out);
                          })})})})})})
}
