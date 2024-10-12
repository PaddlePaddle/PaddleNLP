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

// #define DEBUG_ATTN_C8
template <typename T,
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
__global__ void multi_query_append_attention_kernel(
    T *__restrict__ q,        // [token_num. num_heads, head_dim]
    T *__restrict__ cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    T *__restrict__ cache_v,
    const T *__restrict__ shift_bias,     // [q_num_heads * HEAD_DIM]
    const T *__restrict__ smooth_weight,  // [q_num_heads * HEAD_DIM]
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
    const float in_scale,
    const uint32_t chunk_size,
    T *__restrict__ tmp_workspace,  // split kv [token_num, num_chunks, num_heads, head_dim]
    float *__restrict__ tmp_m,      // [token_num, num_chunks, num_heads]
    float *__restrict__ tmp_d,      // [token_num, num_chunks, num_heads]
    OutT *__restrict__ out,
    const int speculate_max_draft_token_num = 5) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
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
    kv_len += q_len;  // !!!
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

  const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
  const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
  const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_b_stride = HEAD_DIM;
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

  /*
    1 ｜ 3
    ——————
    2 ｜ 4
  */
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_x * 16 + tid % 16, tid / 16);  // 16 * 16
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

  smem_t k_smem(smem + NUM_WARPS * num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + (NUM_WARPS * num_frags_x + num_frags_z) * 16 * HEAD_DIM *
                        sizeof(T));


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
#ifdef DEBUG_ATTN
  if (tid == 0 && wid == 0 && kv_head_idx == 0) {
    printf(
        "batch_id: %d, tile_id: %d, chunk_size: %d, q_len: %d, kv_len: %d, "
        "chunk_start: %d, chunk_end: %d, num_iterations: %d, "
        "mask_check_iteration: %d\n",
        (int)batch_id,
        (int)tile_id,
        (int)chunk_size,
        (int)q_len,
        (int)kv_len,
        (int)chunk_start,
        (int)chunk_end,
        (int)num_iterations,
        (int)mask_check_iteration);
  }
  __syncthreads();
#endif

  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  /*
    1 ｜ 3
    ——————
    2 ｜ 4   transpose
  */
  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head>(tid % 16, tid / 16);

  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * 4 + tid / 8, tid % 8);

  uint32_t kv_idx_base = chunk_start;
  int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  const uint32_t const_offset = kv_head_idx * kv_h_stride +
                                (wid * 4 + tid / 8) * kv_b_stride +
                                tid % 8 * num_elems_per_128b<T>();
  T *cache_k_now = cache_k + block_id * kv_n_stride + const_offset;
  T *cache_v_now = cache_v + block_id * kv_n_stride + const_offset;

  produce_kv_blockwise<SharedMemFillMode::kNoFill,
                       NUM_WARPS,
                       BLOCK_SIZE,
                       num_frags_y,
                       num_frags_z,
                       NUM_WARP_Q>(k_smem,
                                   &kv_smem_offset_w,
                                   &cache_k_now,
                                   kv_head_idx,
                                   kv_n_stride,
                                   kv_h_stride,
                                   kv_b_stride,
                                   kv_idx_base,
                                   chunk_end);
  commit_group();
  produce_kv_blockwise<SharedMemFillMode::kFillZero,
                       NUM_WARPS,
                       BLOCK_SIZE,
                       num_frags_y,
                       num_frags_z,
                       NUM_WARP_Q>(v_smem,
                                   &kv_smem_offset_w,
                                   &cache_v_now,
                                   kv_head_idx,
                                   kv_n_stride,
                                   kv_h_stride,
                                   kv_b_stride,
                                   kv_idx_base,
                                   chunk_end);
  commit_group();
#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 && blockIdx.x == 0) {
      printf("cache_k_smem\n");
      T *k_smem_t = reinterpret_cast<T *>(k_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
          printf("k_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)j,
                 (float)k_smem_t[i * num_frags_y * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // s = qk
    compute_qk<num_frags_x, num_frags_y, num_frags_z, T>(
        &qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
    // mask according to kv_idx and q_idx
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

    // update m,d
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();
#ifdef DEBUG_ATTN
    if (threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 0 &&
        blockIdx.y == 0 && blockIdx.z == 0) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf(
                "after_update_mdo_states_tid:%d_mask_s_s_frag[%d][%d][%d]:%f  ",
                (int)threadIdx.x,
                (int)fx,
                (int)fz,
                (int)k,
                s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    kv_idx_base += num_frags_z * 16;
    block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
    if (block_id < 0) {
      block_id = 0;  // 搬但不算
    }
    cache_k_now = cache_k + block_id * kv_n_stride + const_offset;
    produce_kv_blockwise<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(k_smem,
                                     &kv_smem_offset_w,
                                     &cache_k_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
    commit_group();
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 && blockIdx.x == 0) {
      printf("cache_v_smem\n");
      T *v_smem_t = reinterpret_cast<T *>(v_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
          printf("v_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)j,
                 (float)v_smem_t[i * num_frags_y * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, T>(
        &v_smem, &v_smem_offset_r, s_frag, o_frag, d_frag);

    __syncthreads();
    cache_v_now = cache_v + block_id * kv_n_stride + const_offset;
    produce_kv_blockwise<SharedMemFillMode::kFillZero,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(v_smem,
                                     &kv_smem_offset_w,
                                     &cache_v_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
    commit_group();
  }
  wait_group<0>();
  __syncthreads();
#ifdef DEBUG_ATTN
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
    printf("o_res\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  if constexpr (!partition_kv) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }

  // write o
  // [num_frags_x, 16, num_frags_y, 16]
  // if (in_scale > 0.0) {
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
__global__ void multi_query_append_attention_warp1_4_kernel(
    T *__restrict__ q,        // [token_num. num_heads, head_dim]
    T *__restrict__ cache_k,  // [max_block_num, num_heads, block_size,
                              // head_dim]
    T *__restrict__ cache_v,
    const T *__restrict__ shift_bias,     // [q_num_heads * HEAD_DIM]
    const T *__restrict__ smooth_weight,  // [q_num_heads * HEAD_DIM]
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
    const float in_scale,
    const uint32_t chunk_size,
    T *__restrict__ tmp_workspace,  // split kv [token_num, num_chunks,
                                    // num_heads, head_dim]
    float *__restrict__ tmp_m,      // [token_num, num_chunks, num_heads]
    float *__restrict__ tmp_d,      // [token_num, num_chunks, num_heads]
    OutT *__restrict__ out,
    const int speculate_max_draft_token_num = 5) {
  // q_len <= 32, num_frags_x = 1/2, num_frags_z = 4 / 4 * 1/2/4, num_frags_y =
  // HEAD_DIM / 16
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();
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
    kv_len += q_len;  // !!!
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

  const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
  const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
  const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_b_stride = HEAD_DIM;
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
  // } else {
  //   o_base_ptr_int8 = out + o_offset;
  }

  smem_t qo_smem(smem);

  /*
    1 ｜ 3
    ——————
    2 ｜ 4
  */
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16, tid / 16);  // 16 * 16
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

  smem_t k_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + (num_frags_x + NUM_WARP_KV * num_frags_z) * 16 * HEAD_DIM *
                        sizeof(T));

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

  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  /*
    1 ｜ 3
    ——————
    2 ｜ 4   transpose
  */
  uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_z * 16 + tid % 16, tid / 16);
  uint32_t kv_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * 4 + tid / 8, tid % 8);  // 注意内存访问事务，8 * 128 / 8 = 128B
  // uint32_t kv_smem_offset_w =
  // smem_t::get_permuted_offset<num_vecs_per_head>(wid * num_frags_z * 16 + tid
  // / 8, tid % 8); // 注意内存访问事务，8 * 128 / 8 = 128B

  uint32_t kv_idx_base = chunk_start;
  int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  const uint32_t const_offset = kv_head_idx * kv_h_stride +
                                (wid * 4 + tid / 8) * kv_b_stride +
                                tid % 8 * num_elems_per_128b<T>();
  // uint32_t kv_idx_base = chunk_start + wid * num_frags_z * 16;
  // int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  // const uint32_t const_offset = kv_head_idx * kv_h_stride + (wid *
  // num_frags_z * 16 % BLOCK_SIZE + tid / 8) * kv_b_stride + tid % 8 *
  // num_elems_per_128b<T>();
  T *cache_k_now = cache_k + block_id * kv_n_stride + const_offset;
  T *cache_v_now = cache_v + block_id * kv_n_stride + const_offset;

  // load BLOCK_SIZE * HEAD_DIM each time
  produce_kv_blockwise<SharedMemFillMode::kNoFill,
                       NUM_WARPS,
                       BLOCK_SIZE,
                       num_frags_y,
                       num_frags_z,
                       NUM_WARP_Q>(k_smem,
                                   &kv_smem_offset_w,
                                   &cache_k_now,
                                   kv_head_idx,
                                   kv_n_stride,
                                   kv_h_stride,
                                   kv_b_stride,
                                   kv_idx_base,
                                   chunk_end);
  commit_group();

  produce_kv_blockwise<SharedMemFillMode::kFillZero,
                       NUM_WARPS,
                       BLOCK_SIZE,
                       num_frags_y,
                       num_frags_z,
                       NUM_WARP_Q>(v_smem,
                                   &kv_smem_offset_w,
                                   &cache_v_now,
                                   kv_head_idx,
                                   kv_n_stride,
                                   kv_h_stride,
                                   kv_b_stride,
                                   kv_idx_base,
                                   chunk_end);
  commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    wait_group<1>();
    __syncthreads();

    // s = qk
    compute_qk<num_frags_x, num_frags_y, num_frags_z, T>(
        &qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
    // mask according to kv_idx and q_idx
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

    // update m,d
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();

    kv_idx_base += NUM_WARP_KV * num_frags_z * 16;
    block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
    if (block_id < 0) {
      block_id = 0;  // 搬但不算
    }
    cache_k_now = cache_k + block_id * kv_n_stride + const_offset;
    produce_kv_blockwise<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(k_smem,
                                     &kv_smem_offset_w,
                                     &cache_k_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
    commit_group();
    wait_group<1>();
    __syncthreads();

    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, T>(
        &v_smem, &v_smem_offset_r, s_frag, o_frag, d_frag);
    __syncthreads();

    cache_v_now = cache_v + block_id * kv_n_stride + const_offset;
    produce_kv_blockwise<SharedMemFillMode::kFillZero,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(v_smem,
                                     &kv_smem_offset_w,
                                     &cache_v_now,
                                     kv_head_idx,
                                     kv_n_stride,
                                     kv_h_stride,
                                     kv_b_stride,
                                     kv_idx_base,
                                     chunk_end);
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
    write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE, num_frags_x, num_frags_y, false>(
      o_frag,
      &qo_smem,
      o_base_ptr_int8,
      shift_bias,
      smooth_weight,
      q_base_seq_id_this_block,
      q_head_idx,
      in_scale,
      q_len,
      q_n_stride, 
      HEAD_DIM
    );
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
        in_scale,
        q_len,
        q_n_stride * num_chunks,
        HEAD_DIM);
  // } else {
  //   write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE,
  //                                                   num_frags_x,
  //                                                   num_frags_y,
  //                                                   partition_kv>(
  //       o_frag,
  //       &qo_smem,
  //       o_base_ptr_int8,
  //       shift_bias,
  //       smooth_weight,
  //       q_base_seq_id_this_block,
  //       q_head_idx,
  //       in_scale,
  //       q_len,
  //       partition_kv ? q_n_stride * num_chunks : q_n_stride,
  //       HEAD_DIM);
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
__global__ void multi_query_append_attention_c8_kernel(
    T *__restrict__ q,             // [token_num. num_heads, head_dim]
    CacheT *__restrict__ cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    CacheT *__restrict__ cache_v,
    const T *__restrict__ cache_k_scale,  // [num_kv_heads]
    const T *__restrict__ cache_v_scale,  // [num_kv_heads]
    const T *__restrict__ shift_bias,     // [q_num_heads * HEAD_DIM]
    const T *__restrict__ smooth_weight,  // [q_num_heads * HEAD_DIM]
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
    const float in_scale,
    const uint32_t chunk_size,
    T *__restrict__ tmp_workspace,  // split kv [token_num, num_chunks,
                                    // num_heads, head_dim]
    float *__restrict__ tmp_m,      // [token_num, num_chunks, num_heads]
    float *__restrict__ tmp_d,      // [token_num, num_chunks, num_heads]
    OutT *__restrict__ out,
    const int speculate_max_draft_token_num = 5) {
#ifdef DEBUG_ATTN_C8
  __syncthreads();
  printf("launched multi_query_append_attention_c8_kernel");
  __syncthreads();
#endif
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>(); // 128 / 8 = 16
  constexpr uint32_t num_vecs_per_head_k = HEAD_DIM / num_elems_per_128b<CacheT>(); // 128 / 16 = 8
  constexpr uint32_t num_vecs_per_blocksize = BLOCK_SIZE / num_elems_per_128b<CacheT>(); //  64 / 16 = 4
  constexpr uint32_t inv_k_stride = 8 / num_vecs_per_head_k;
  constexpr uint32_t inv_v_stride = 8 / num_vecs_per_blocksize;
  const uint32_t btid = blockIdx.x, kv_head_idx = blockIdx.z;
  const uint32_t kv_num_heads = gridDim.z;
  const uint32_t q_num_heads = kv_num_heads * GROUP_SIZE;
  const uint32_t q_head_idx = kv_head_idx * GROUP_SIZE;
  const uint32_t tid = threadIdx.x, wid = threadIdx.y;
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_idx = blockIdx.y;
#ifdef DEBUG_ATTN_C8
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf("num_vecs_per_head: %d, num_vecs_per_head_k: %d, num_vecs_per_blocksize: %d, inv_k_stride: %d, inv_v_stride: %d\n",
            (int)num_vecs_per_head, (int)num_vecs_per_head_k, (int)num_vecs_per_blocksize, (int)inv_k_stride, (int)inv_v_stride);
  }
  __syncthreads();
#endif

  const uint32_t batch_id = batch_ids[btid];
  const uint32_t tile_id = tile_ids_per_batch[btid];
  const uint32_t num_rows_per_block = NUM_WARPS * num_frags_x * 16;
  const int *block_table_now = nullptr;

  block_table_now = block_table + batch_id * max_block_num_per_seq;

  const uint32_t q_len = seq_lens[batch_id];
  if (q_len <= 0) {
    return;
  }
  const T cache_k_scale_reg = cache_k_scale[kv_head_idx];
  const T cache_v_scale_reg = cache_v_scale[kv_head_idx];

  const uint32_t q_end = min(q_len, div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE));
  uint32_t kv_len = seq_lens_kv[batch_id];
  if (ENABLE_PREFILL) {
    kv_len += q_len; // !!!
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
  const uint32_t chunk_end = partition_kv ? min(kv_len, chunk_start + chunk_size) : kv_len;
  const uint32_t chunk_len = chunk_end - chunk_start;

  extern __shared__ uint8_t smem[];
  float s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  float m_frag[num_frags_x][2];
  float d_frag[num_frags_x][2];
  init_states<T, num_frags_x, num_frags_y>(o_frag, m_frag, d_frag);

  const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
  const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
  const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_b_stride = HEAD_DIM;
  const uint32_t kv_d_stride = BLOCK_SIZE;
  const uint32_t q_start_seq_id = batch_id * max_seq_len - __ldg(&cum_offsets[batch_id]);
  const uint32_t q_base_seq_id_this_block = (tile_id * NUM_WARPS + wid) * num_frags_x * 16;
  const uint32_t q_offset = q_start_seq_id * q_ori_n_stride + q_head_idx * HEAD_DIM + tid % 8 * num_elems_per_128b<T>();
  const uint32_t o_offset = q_start_seq_id * q_n_stride + q_head_idx * HEAD_DIM + tid % 8 * num_elems_per_128b<T>();
  T *q_base_ptr = q + q_offset;
#ifdef DEBUG_ATTN_C8
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf("q_start_seq_id: %d, q_offset: %d, q_ori_n_stride: %d, q_base: %f\n",
            (int)q_start_seq_id, (int)q_offset, (int)q_ori_n_stride, (float)*q_base_ptr);
  }
  __syncthreads();
#endif

  T *o_base_ptr_T = nullptr;
  OutT *o_base_ptr_int8 = nullptr;
  if constexpr (partition_kv) {
    if (ENABLE_PREFILL) {
      o_base_ptr_T = tmp_workspace + q_start_seq_id * num_chunks * q_n_stride + chunk_idx * q_n_stride + q_head_idx * HEAD_DIM + tid % 8 * num_elems_per_128b<T>();
    } else {
      o_base_ptr_T = tmp_workspace + batch_id * speculate_max_draft_token_num * num_chunks * q_n_stride + chunk_idx * q_n_stride + q_head_idx * HEAD_DIM + tid % 8 * num_elems_per_128b<T>();
    }
  } else {
    o_base_ptr_int8 = out + o_offset;
  }
  smem_t qo_smem(smem);

  /*
    1 ｜ 3
    ——————
    2 ｜ 4
  */
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(wid * num_frags_x * 16 + tid % 16, tid / 16); // 16 * 16
  load_q_global_smem<GROUP_SIZE, num_frags_x, num_frags_y, HEAD_DIM, T>(
    q_base_ptr,
    &qo_smem,
    q_base_seq_id_this_block,
    q_end,
    q_ori_n_stride,
    HEAD_DIM
  );
  commit_group();
  wait_group<0>();
  __syncthreads();

  q_smem_inplace_multiply_sm_scale<num_frags_x, num_frags_y, T>(&qo_smem, scale);
  smem_t k_smem(smem + NUM_WARPS * num_frags_x * 16 * HEAD_DIM * sizeof(T)),
         v_smem(smem + NUM_WARPS * num_frags_x * 16 * HEAD_DIM * sizeof(T) + num_frags_z * 16 * HEAD_DIM * sizeof(CacheT));
  

  const uint32_t num_iterations = div_up(CAUSAL ? 
                                         (min(chunk_len, 
                                              sub_if_greater_or_zero(kv_len - q_len + div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE), chunk_start)))
                                              : chunk_len, num_frags_z * 16);
  const uint32_t mask_check_iteration = (CAUSAL ? 
                                          (min(chunk_len, 
                                            sub_if_greater_or_zero(kv_len - q_len + tile_id * num_rows_per_block / GROUP_SIZE, chunk_start)))
                                            : chunk_len) / (num_frags_z * 16);
#ifdef DEBUG_ATTN_C8
  if (tid == 0 && wid == 0) {
    printf("batch_id: %d, tile_id: %d, chunk_size: %d, q_len: %d, kv_len: %d, chunk_start: %d, chunk_end: %d, num_iterations: %d, mask_check_iteration: %d\n",
           (int)batch_id, (int)tile_id, (int)chunk_size, (int)q_len, (int)kv_len, (int)chunk_start, (int)chunk_end, (int)num_iterations, (int)mask_check_iteration);
  }
  __syncthreads();
#endif

  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */       
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */ 
  uint32_t v_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  
  uint32_t k_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(wid * 4 + tid / 8, tid % 8); // 8 * 128 / 8 = 128 !!! just for HEAD_DIM >= 128
  uint32_t v_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(wid * 8 + tid / 4, tid % 4); // 4 * 128 / 8 = 64
  
  uint32_t kv_idx_base = chunk_start;
  // int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  const uint32_t const_k_offset = kv_head_idx * kv_h_stride + (wid * 4 + tid / 8) * kv_b_stride + tid % 8 * num_elems_per_128b<CacheT>();
  const uint32_t const_v_offset = kv_head_idx * kv_h_stride + (wid * 8 + tid / 4) * kv_d_stride + tid % 4 * num_elems_per_128b<CacheT>();
  // CacheT *cache_k_now = cache_k + block_id * kv_n_stride + const_k_offset;
  // CacheT *cache_v_now = cache_v + block_id * kv_n_stride + const_v_offset;
#ifdef DEBUG_ATTN_C8
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
    printf("387 ori q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: %d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
           (int)q_smem_offset_r, (int)k_smem_offset_r, (int)v_smem_offset_r, (int)k_smem_offset_w, (int)v_smem_offset_w);
  }
  __syncthreads();
#endif

  produce_k_blockwise_c8<SharedMemFillMode::kNoFill, NUM_WARPS, BLOCK_SIZE, num_frags_y, num_frags_z, NUM_WARP_Q>(
    k_smem,
    &k_smem_offset_w,
    cache_k,
    block_table_now,
    kv_head_idx,
    kv_n_stride,
    kv_h_stride,
    kv_b_stride,
    kv_idx_base,
    chunk_end,
    const_k_offset
  );
  commit_group();
  produce_v_blockwise_c8<SharedMemFillMode::kNoFill, NUM_WARPS, BLOCK_SIZE, num_frags_y, num_frags_z, NUM_WARP_Q>(
    v_smem,
    &v_smem_offset_w,
    cache_v,
    block_table_now,
    kv_head_idx,
    kv_n_stride,
    kv_h_stride,
    kv_d_stride,
    kv_idx_base,
    chunk_end,
    const_v_offset
  );
  commit_group();
#ifdef DEBUG_ATTN_C8
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
    printf("418 ori q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: %d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
           (int)q_smem_offset_r, (int)k_smem_offset_r, (int)v_smem_offset_r, (int)k_smem_offset_w, (int)v_smem_offset_w);
  }
  __syncthreads();
#endif

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN_C8
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
      printf("cache_k_smem\n");
      uint8_t *k_smem_t = reinterpret_cast<uint8_t*>(k_smem.base);
      for (uint32_t i = 0; i < num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
          printf("k_smem[%d][%d] = %d  ", (int)i, (int)j, (int)k_smem_t[i * num_frags_y * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    // s = qk
    compute_qk_c8<num_frags_x, num_frags_y, num_frags_z, T, CacheT>(
      &qo_smem, 
      &q_smem_offset_r, 
      &k_smem,
      &k_smem_offset_r, 
      cache_k_scale_reg, 
      s_frag);
#ifdef DEBUG_ATTN_C8
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
      printf("111 iter: %d, q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: %d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
            (int)iter, (int)q_smem_offset_r, (int)k_smem_offset_r, (int)v_smem_offset_r, (int)k_smem_offset_w, (int)v_smem_offset_w);
    }
    __syncthreads();
#endif

    // mask according to kv_idx and q_idx
    if (iter >= mask_check_iteration) {
      mask_s<T, partition_kv, CAUSAL, GROUP_SIZE, NUM_WARPS, num_frags_x, num_frags_y, num_frags_z>(
             q_base_seq_id_this_block, kv_idx_base, q_len, kv_len, chunk_end, s_frag);
    }
#ifdef DEBUG_ATTN_C8
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("mask_s s_frag[%d][%d][%d]: %f  ", (int)fx, (int)fz, (int)k, s_frag[fx][fz][k]);
          }
          printf("\n"); 
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

     // update m,d
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
      s_frag, o_frag, m_frag, d_frag);
    __syncthreads();

    kv_idx_base += num_frags_z * 16;
    produce_k_blockwise_c8<SharedMemFillMode::kNoFill, NUM_WARPS, BLOCK_SIZE, num_frags_y, num_frags_z, NUM_WARP_Q>(
      k_smem,
      &k_smem_offset_w,
      cache_k,
      block_table_now,
      kv_head_idx,
      kv_n_stride,
      kv_h_stride,
      kv_b_stride,
      kv_idx_base,
      chunk_end,
      const_k_offset
    );
    commit_group();
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN_C8
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
      printf("222 iter: %d, q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: %d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
            (int)iter, (int)q_smem_offset_r, (int)k_smem_offset_r, (int)v_smem_offset_r, (int)k_smem_offset_w, (int)v_smem_offset_w);
    }
    __syncthreads();
#endif

    // compute sfm*v
    compute_sfm_v_c8<num_frags_x, num_frags_y, num_frags_z, BLOCK_SIZE, T, CacheT>(
      &v_smem, 
      &v_smem_offset_r, 
      s_frag,
      o_frag, 
      d_frag, 
      cache_v_scale_reg);
    __syncthreads();
#ifdef DEBUG_ATTN_C8
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
      printf("333 iter: %d, q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: %d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
            (int)iter, (int)q_smem_offset_r, (int)k_smem_offset_r, (int)v_smem_offset_r, (int)k_smem_offset_w, (int)v_smem_offset_w);
    }
    __syncthreads();
#endif
    produce_v_blockwise_c8<SharedMemFillMode::kNoFill, NUM_WARPS, BLOCK_SIZE, num_frags_y, num_frags_z, NUM_WARP_Q>(
      v_smem,
      &v_smem_offset_w,
      cache_v,
      block_table_now,
      kv_head_idx,
      kv_n_stride,
      kv_h_stride,
      kv_d_stride,
      kv_idx_base,
      chunk_end,
      const_v_offset
    );
    commit_group();
#ifdef DEBUG_ATTN_C8
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
      printf("444 iter: %d, q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: %d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
            (int)iter, (int)q_smem_offset_r, (int)k_smem_offset_r, (int)v_smem_offset_r, (int)k_smem_offset_w, (int)v_smem_offset_w);
    }
    __syncthreads();
#endif
  }
  wait_group<0>();
  __syncthreads();

  if constexpr (!partition_kv) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }

  // write o
  // [num_frags_x, 16, num_frags_y, 16]
  if constexpr (partition_kv) {
    write_o_reg_gmem_shift_smooth_quant<GROUP_SIZE, num_frags_x, num_frags_y, partition_kv>(
      o_frag,
      &qo_smem,
      o_base_ptr_T,
      shift_bias,
      smooth_weight,
      q_base_seq_id_this_block,
      q_head_idx,
      in_scale,
      q_len,
      partition_kv ? q_n_stride * num_chunks : q_n_stride, 
      HEAD_DIM
    );
  } else {
    write_o_reg_gmem_shift_smooth_quant<GROUP_SIZE, num_frags_x, num_frags_y, partition_kv>(
      o_frag,
      &qo_smem,
      o_base_ptr_int8,
      shift_bias,
      smooth_weight,
      q_base_seq_id_this_block,
      q_head_idx,
      in_scale,
      q_len,
      partition_kv ? q_n_stride * num_chunks : q_n_stride, 
      HEAD_DIM
    );
  }


  if constexpr (partition_kv) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_idx_now = q_base_seq_id_this_block + tid / 4 + j * 8 + fx * 16;
        const uint32_t qo_head_idx = q_head_idx + qo_idx_now % GROUP_SIZE;
        const uint32_t qo_idx = q_start_seq_id + qo_idx_now / GROUP_SIZE;
        if (qo_idx - q_start_seq_id < q_len) {
          uint32_t offset;
          if (ENABLE_PREFILL) {
            offset = (qo_idx * num_chunks + chunk_idx) * q_num_heads + qo_head_idx;
          } else {
            offset = ((batch_id * speculate_max_draft_token_num + qo_idx_now / GROUP_SIZE) * num_chunks + chunk_idx) * q_num_heads + qo_head_idx;
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
__global__ void multi_query_append_attention_c8_warp1_4_kernel(
    T *__restrict__ q,             // [token_num. num_heads, head_dim]
    CacheT *__restrict__ cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    CacheT *__restrict__ cache_v,
    const T *__restrict__ cache_k_scale,  // [num_kv_heads, head_dim]
    const T *__restrict__ cache_v_scale,  // [num_kv_heads, head_dim]
    const T *__restrict__ shift_bias,     // [q_num_heads * HEAD_DIM]
    const T *__restrict__ smooth_weight,  // [q_num_heads * HEAD_DIM]
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
      HEAD_DIM / num_elems_per_128b<CacheT>();
  constexpr uint32_t num_vecs_per_blocksize =
      BLOCK_SIZE / num_elems_per_128b<CacheT>();
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
  const T cache_k_scale_reg = cache_k_scale[kv_head_idx];
  const T cache_v_scale_reg = cache_v_scale[kv_head_idx];
  const uint32_t q_end =
      min(q_len, div_up((tile_id + 1) * num_rows_per_block, GROUP_SIZE));
  uint32_t kv_len = seq_lens_kv[batch_id];
  if (ENABLE_PREFILL) {
    kv_len += q_len;  // !!!
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

  const uint32_t q_n_stride = q_num_heads * HEAD_DIM;
  const uint32_t q_ori_n_stride = (q_num_heads + kv_num_heads * 2) * HEAD_DIM;
  const uint32_t kv_n_stride = kv_num_heads * BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
  const uint32_t kv_b_stride = HEAD_DIM;
  const uint32_t kv_d_stride = BLOCK_SIZE;
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
  // if constexpr (partition_kv) {
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
  // } else {
  //   o_base_ptr_int8 = out + o_offset;
  }
#ifdef DEBUG_ATTN_C8
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf(
        "q_base_seq_id_this_block: %d, q_base_seq_id_this_block: %d, q_offset: "
        "%d, o_offset: %d\n",
        (int)q_base_seq_id_this_block,
        (int)q_base_seq_id_this_block,
        (int)q_offset,
        (int)o_offset);
  }
  __syncthreads();
#endif

  smem_t qo_smem(smem);

  /*
    1 ｜ 3
    ——————
    2 ｜ 4
  */
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16, tid / 16);  // 16 * 16
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
#ifdef DEBUG_ATTN_C8
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 &&
      blockIdx.x == gridDim.x - 1) {
    printf("after scale\n");
    T *q_smem_t = reinterpret_cast<T *>(qo_smem.base);
    for (uint32_t i = 0; i < num_frags_x * 16; ++i) {
      for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
        if (blockIdx.z == 0) {
          printf("q_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)(j),
                 (float)q_smem_t[i * num_frags_y * 16 + j]);
        } else {
          int res = q_smem_t[i * num_frags_y * 16 + j] + static_cast<T>(1.f);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  smem_t k_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T)),
      v_smem(smem + num_frags_x * 16 * HEAD_DIM * sizeof(T) +
             NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(CacheT));

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
#ifdef DEBUG_ATTN_C8
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
      blockIdx.x == gridDim.x - 1) {
    printf(
        "cid: %d, batch_id: %d, tile_id: %d, chunk_size: %d, q_len: %d, "
        "kv_len: %d, chunk_start: %d, chunk_end: %d, num_iterations: %d, "
        "mask_check_iteration: %d\n",
        (int)blockIdx.y,
        (int)batch_id,
        (int)tile_id,
        (int)chunk_size,
        (int)q_len,
        (int)kv_len,
        (int)chunk_start,
        (int)chunk_end,
        (int)num_iterations,
        (int)mask_check_iteration);
  }
  __syncthreads();
#endif
  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  /*
    1 ｜ 2
    ——————
    3 ｜ 4   transpose
  */
  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          (wid / 2) * num_frags_y * 16 + 8 * (tid / 16) + tid % 8,
          (wid % 2) * num_frags_z + (tid % 16) / 8);

  uint32_t k_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 4 + tid / 8,
          tid %
              8);  // 4 * 128 / 8 = 64B, 128 nums, just fot head_dim >= 128 !!!
  uint32_t v_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 8 + tid / 4, tid % 4);  // 2 * 128 / 8 = 32B, 64 nums

  // uint32_t kv_idx_base = chunk_start;
  // int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  // const uint32_t const_offset = kv_head_idx * kv_h_stride + (wid * 4 + tid /
  // 8) * kv_b_stride + tid % 8 * num_elems_per_128b<T>();
  uint32_t kv_idx_base = chunk_start;
  // int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  const uint32_t const_k_offset = kv_head_idx * kv_h_stride +
                                  (wid * 4 + tid / 8) * kv_b_stride +
                                  tid % 8 * num_elems_per_128b<CacheT>();
  const uint32_t const_v_offset = kv_head_idx * kv_h_stride +
                                  (wid * 8 + tid / 4) * kv_d_stride +
                                  tid % 4 * num_elems_per_128b<CacheT>();
  // T *cache_k_now = cache_k + block_id * kv_n_stride + const_k_offset;
  // T *cache_v_now = cache_v + block_id * kv_n_stride + const_v_offset;

#ifdef DEBUG_ATTN_C8
  if (threadIdx.y == 0 && blockIdx.z == 0 && blockIdx.x == gridDim.x - 1 &&
      blockIdx.y == gridDim.y - 1) {
    printf(
        "000 tid: %d, ori q_smem_offset_r: %d, k_smem_offset_r: %d, "
        "v_smem_offset_r: %d, k_smem_offset_w: %d, v_smem_offset_w: %d, "
        "cache_k: %f, cache_k_p: %p, const_k_offset: %d, const_v_offset: %d\n",
        (int)threadIdx.x,
        (int)q_smem_offset_r,
        (int)k_smem_offset_r,
        (int)v_smem_offset_r,
        (int)k_smem_offset_w,
        (int)v_smem_offset_w,
        (float)(*cache_k),
        cache_k,
        (int)const_k_offset,
        (int)const_v_offset);
  }
  __syncthreads();
#endif

  // load BLOCK_SIZE * HEAD_DIM each time
  produce_k_blockwise_c8<SharedMemFillMode::kNoFill,
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
  produce_v_blockwise_c8<SharedMemFillMode::kNoFill,
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
#ifdef DEBUG_ATTN_C8
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      printf("cache_k_smem\n");
      uint8_t *k_smem_t = reinterpret_cast<uint8_t *>(k_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
          printf("k_smem[%d][%d] = %d  ",
                 (int)i,
                 (int)j,
                 (int)k_smem_t[i * num_frags_y * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // s = qk
    compute_qk_c8<num_frags_x, num_frags_y, num_frags_z, T, CacheT>(
        &qo_smem,
        &q_smem_offset_r,
        &k_smem,
        &k_smem_offset_r,
        cache_k_scale_reg,
        s_frag);
    // mask according to kv_idx and q_idx
    if (iter >= mask_check_iteration) {
      // if (q_len > 1 && iter >= mask_check_iteration) { // not need mask in
      // decoder, v will be filled with 0
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
#ifdef DEBUG_ATTN_C8
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("mask_s s_frag[%d][%d][%d]: %f  ",
                   (int)fx,
                   (int)fz,
                   (int)k,
                   s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    // update m,d
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();
#ifdef DEBUG_ATTN_C8
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("update_mdo_states s_frag[%d][%d][%d]: %f  ",
                   (int)fx,
                   (int)fz,
                   (int)k,
                   s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    kv_idx_base += NUM_WARP_KV * num_frags_z * 16;
    produce_k_blockwise_c8<SharedMemFillMode::kNoFill,
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
#ifdef DEBUG_ATTN_C8
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      printf("cache_v_smem\n");
      uint8_t *v_smem_t = reinterpret_cast<uint8_t *>(v_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV / 2 * num_frags_y * 16; ++i) {
        for (uint32_t j = 0; j < 2 * num_frags_z * 16; ++j) {
          printf("v_smem[%d][%d] = %d  ",
                 (int)i,
                 (int)j,
                 (int)v_smem_t[i * 2 * num_frags_z * 16 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    // compute sfm * v
    compute_sfm_v_c8_iter_sq_bvec<num_frags_x,
                                  num_frags_y,
                                  num_frags_z,
                                  BLOCK_SIZE,
                                  T,
                                  CacheT>(
        &v_smem, &v_smem_offset_r, s_frag, o_frag, d_frag, cache_v_scale_reg);
    __syncthreads();

    produce_v_blockwise_c8<SharedMemFillMode::kNoFill,
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
#ifdef DEBUG_ATTN_C8
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
      blockIdx.x == gridDim.x - 1) {
    printf("before merge z\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  merge_block_res_v2<num_frags_x, num_frags_y, T>(
      o_frag, reinterpret_cast<float *>(smem), m_frag, d_frag, wid, tid);
#ifdef DEBUG_ATTN_C8
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
      blockIdx.x == 0 && blockIdx.y == 0) {
    printf("after merge z\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  if (num_chunks_this_seq <= 1) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }
#ifdef DEBUG_ATTN_C8
  __syncthreads();
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
      blockIdx.x == gridDim.x - 1) {
    printf("after normalize_d\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  // write o
  // [num_frags_x, 16, num_frags_y, 16]
  if (num_chunks_this_seq <= 1) {
    write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE, num_frags_x, num_frags_y, false>(
      o_frag,
      &qo_smem,
      o_base_ptr_int8,
      shift_bias,
      smooth_weight,
      q_base_seq_id_this_block,
      q_head_idx,
      in_scale,
      q_len,
      q_n_stride, 
      HEAD_DIM
    );
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
        in_scale,
        q_len,
        q_n_stride * num_chunks,
        HEAD_DIM);
  // } else {
  //   write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE,
  //                                                   num_frags_x,
  //                                                   num_frags_y,
  //                                                   partition_kv>(
  //       o_frag,
  //       &qo_smem,
  //       o_base_ptr_int8,
  //       shift_bias,
  //       smooth_weight,
  //       q_base_seq_id_this_block,
  //       q_head_idx,
  //       in_scale,
  //       q_len,
  //       partition_kv ? q_n_stride * num_chunks : q_n_stride,
  //       HEAD_DIM);
  }
#ifdef DEBUG_ATTN_C8
  __syncthreads();
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
      blockIdx.x == gridDim.x - 1) {
    printf("after normalize_d\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

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
#ifdef DEBUG_ATTN_C8
          if (batch_id == 0) {
            printf(
                "bid: %d, tid: %d, wid: %d, q_base_seq_id_this_block: %d, "
                "qo_idx_now: %d, qo_idx: %d, q_start_seq_id: %d, q_len: %d, m: "
                "%f, d: %f\n",
                (int)batch_id,
                (int)tid,
                (int)wid,
                (int)q_base_seq_id_this_block,
                (int)qo_idx_now,
                (int)qo_idx,
                (int)q_start_seq_id,
                (int)q_len,
                (float)m_frag[fx][j],
                (float)d_frag[fx][j]);
          }
#endif
          if (qo_idx - q_start_seq_id < q_len) {
            // const uint32_t offset = (qo_idx * num_chunks + chunk_idx) *
            // q_num_heads + qo_head_idx;

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
    T *__restrict__ q,             // [token_num. num_heads, head_dim]
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
#ifdef DEBUG_ATTN_C4
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf(
        "num_vecs_per_head: %d, num_vecs_per_head_k: %d, "
        "num_vecs_per_blocksize: %d, inv_k_stride: %d, inv_v_stride: %d\n",
        (int)num_vecs_per_head,
        (int)num_vecs_per_head_k,
        (int)num_vecs_per_blocksize,
        (int)inv_k_stride,
        (int)inv_v_stride);
  }
  __syncthreads();
#endif

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
    kv_len += q_len;  // !!!
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

  // load kv scale/zp
  // TODO(load kv scale and zp to smem)
  const T *cache_k_scale_now = cache_k_scale + kv_head_idx * HEAD_DIM;
  const T *cache_k_zp_now = cache_k_zero_point + kv_head_idx * HEAD_DIM;
  const T *cache_v_scale_now = cache_v_scale + kv_head_idx * HEAD_DIM;
  const T *cache_v_zp_now = cache_v_zero_point + kv_head_idx * HEAD_DIM;
  // constexpr uint32_t HEAD_DIM_PAD = div_up(HEAD_DIM, 4) * 4;
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
#ifdef DEBUG_ATTN_C4
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf(
        "q_base_seq_id_this_block: %d, q_start_seq_id: %d, q_offset: %d, "
        "q_ori_n_stride: %d, q_base: %f\n",
        (int)q_base_seq_id_this_block,
        (int)q_start_seq_id,
        (int)q_offset,
        (int)q_ori_n_stride,
        (float)*q_base_ptr);
  }
  __syncthreads();
#endif
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

  /*
    1 ｜ 3
    ——————
    2 ｜ 4
  */
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      wid * num_frags_x * 16 + tid % 16, tid / 16);  // 16 * 16
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
#ifdef DEBUG_ATTN_C4
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf("after scale\n");
    T *q_smem_t = reinterpret_cast<T *>(qo_smem.base);
    for (uint32_t i = 0; i < 4 * num_frags_x * 16; ++i) {
      for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
        printf("q_smem[%d][%d] = %f  ",
               (int)i,
               (int)j,
               (float)q_smem_t[i * num_frags_y * 16 + j]);
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

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
#ifdef DEBUG_ATTN
  if (tid == 0 && wid == 0) {
    printf(
        "batch_id: %d, tile_id: %d, chunk_size: %d, q_len: %d, kv_len: %d, "
        "chunk_start: %d, chunk_end: %d, num_iterations: %d, "
        "mask_check_iteration: %d\n",
        (int)batch_id,
        (int)tile_id,
        (int)chunk_size,
        (int)q_len,
        (int)kv_len,
        (int)chunk_start,
        (int)chunk_end,
        (int)num_iterations,
        (int)mask_check_iteration);
  }
  __syncthreads();
#endif

  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t k_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 8 + tid / 4,
          tid %
              4);  // 4 * 128 / 8 = 64B, 128 nums, just fot head_dim >= 128 !!!
  uint32_t v_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 16 + tid / 2, tid % 2);  // 2 * 128 / 8 = 32B, 64 nums

  uint32_t kv_idx_base = chunk_start;
  // int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  const uint32_t const_k_offset = kv_head_idx * kv_h_stride +
                                  (wid * 8 + tid / 4) * kv_b_stride +
                                  tid % 4 * num_elems_per_128b<CacheT>();
  const uint32_t const_v_offset = kv_head_idx * kv_h_stride +
                                  (wid * 16 + tid / 2) * kv_d_stride +
                                  tid % 2 * num_elems_per_128b<CacheT>();
  // CacheT *cache_k_now = cache_k + block_id * kv_n_stride + const_k_offset;
  // CacheT *cache_v_now = cache_v + block_id * kv_n_stride + const_v_offset;
#ifdef DEBUG_ATTN_C4
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
    printf(
        "752 ori q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: "
        "%d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
        (int)q_smem_offset_r,
        (int)k_smem_offset_r,
        (int)v_smem_offset_r,
        (int)k_smem_offset_w,
        (int)v_smem_offset_w);
  }
  __syncthreads();
#endif

  produce_k_blockwise_c4<SharedMemFillMode::kNoFill,
                         NUM_WARPS,
                         BLOCK_SIZE,
                         num_frags_y,
                         num_frags_z,
                         NUM_WARP_Q>(k_smem,
                                     &k_smem_offset_w,
                                     // &cache_k_now,
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
                                     // &cache_v_now,
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
#ifdef DEBUG_ATTN_C4
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
    printf(
        "782 ori q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: "
        "%d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
        (int)q_smem_offset_r,
        (int)k_smem_offset_r,
        (int)v_smem_offset_r,
        (int)k_smem_offset_w,
        (int)v_smem_offset_w);
  }
  __syncthreads();
#endif

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    wait_group<1>();
    __syncthreads();
#ifdef DEBUG_ATTN_C4
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      printf("cache_k_smem\n");
      uint8_t *k_smem_t = reinterpret_cast<uint8_t *>(k_smem.base);
      for (uint32_t i = 0; i < num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16 / 2; ++j) {
          printf("k_smem[%d][%d] = %d  ",
                 (int)i,
                 (int)j,
                 (int)k_smem_t[i * num_frags_y * 16 / 2 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // s = qk
    compute_qk_c4<num_frags_x, num_frags_y, num_frags_z, T, CacheT>(
        &qo_smem,
        &q_smem_offset_r,
        &k_smem,
        &k_smem_offset_r,
        s_frag,
        cache_k_scale_frag,
        cache_k_zp_frag);
#ifdef DEBUG_ATTN_C4
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
      printf(
          "111 iter: %d, q_smem_offset_r: %d, k_smem_offset_r: %d, "
          "v_smem_offset_r: %d, k_smem_offset_w: %d, v_smem_offset_w: %d\n",
          (int)iter,
          (int)q_smem_offset_r,
          (int)k_smem_offset_r,
          (int)v_smem_offset_r,
          (int)k_smem_offset_w,
          (int)v_smem_offset_w);
    }
    __syncthreads();
#endif
    // mask according to kv_idx and q_idx
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
#ifdef DEBUG_ATTN_C4
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("mask_s s_frag[%d][%d][%d]: %f  ",
                   (int)fx,
                   (int)fz,
                   (int)k,
                   s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    // update m,d
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();
#ifdef DEBUG_ATTN_C4
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("update_mdo_states s_frag[%d][%d][%d]: %f  ",
                   (int)fx,
                   (int)fz,
                   (int)k,
                   s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    kv_idx_base += num_frags_z * 16;
    produce_k_blockwise_c4<SharedMemFillMode::kNoFill,
                           NUM_WARPS,
                           BLOCK_SIZE,
                           num_frags_y,
                           num_frags_z,
                           NUM_WARP_Q>(k_smem,
                                       &k_smem_offset_w,
                                       // &cache_k_now,
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
#ifdef DEBUG_ATTN_C4
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      printf("cache_v_smem\n");
      uint8_t *v_smem_t = reinterpret_cast<uint8_t *>(v_smem.base);
      for (uint32_t i = 0; i < num_frags_y * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_z * 16 / 2; ++j) {
          printf("v_smem[%d][%d] = %d  ",
                 (int)(iter * 128 + i),
                 (int)j,
                 (int)v_smem_t[i * num_frags_z * 16 / 2 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

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
                                       // &cache_v_now,
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
#ifdef DEBUG_ATTN_C4
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
      blockIdx.x == gridDim.x - 1) {
    printf("tmp res\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  if constexpr (!partition_kv) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }

  // write o
  // [num_frags_x, 16, num_frags_y, 16]
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
    T *__restrict__ q,             // [token_num. num_heads, head_dim]
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
    const float in_scale,
    const uint32_t chunk_size,
    T *__restrict__ tmp_workspace,  // split kv [token_num, num_chunks,
                                    // num_heads, head_dim]
    float *__restrict__ tmp_m,      // [token_num, num_chunks, num_heads]
    float *__restrict__ tmp_d,      // [token_num, num_chunks, num_heads]
    OutT *__restrict__ out,
    const int speculate_max_draft_token_num = 5) {
  // q_len <= 32, num_frags_x = 1/2, num_frags_z = 4 / 4 * 1/2/4, num_frags_y =
  // HEAD_DIM / 16
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
#ifdef DEBUG_ATTN_C4
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf(
        "num_vecs_per_head: %d, num_vecs_per_head_k: %d, "
        "num_vecs_per_blocksize: %d, inv_k_stride: %d, inv_v_stride: %d\n",
        (int)num_vecs_per_head,
        (int)num_vecs_per_head_k,
        (int)num_vecs_per_blocksize,
        (int)inv_k_stride,
        (int)inv_v_stride);
  }
  __syncthreads();
#endif

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
    kv_len += q_len;  // !!!
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
#ifdef DEBUG_ATTN_C4
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf(
        "q_base_seq_id_this_block: %d, q_start_seq_id: %d, q_offset: %d, "
        "q_ori_n_stride: %d, q_base: %f\n",
        (int)q_base_seq_id_this_block,
        (int)q_start_seq_id,
        (int)q_offset,
        (int)q_ori_n_stride,
        (float)*q_base_ptr);
  }
  __syncthreads();
#endif
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
  // } else {
  //   o_base_ptr_int8 = out + o_offset;
  }
#ifdef DEBUG_ATTN_C4
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf(
        "q_base_seq_id_this_block: %d, q_base_seq_id_this_block: %d, q_offset: "
        "%d, o_offset: %d\n",
        (int)q_base_seq_id_this_block,
        (int)q_base_seq_id_this_block,
        (int)q_offset,
        (int)o_offset);
  }
  __syncthreads();
#endif

  smem_t qo_smem(smem);

  /*
    1 ｜ 3
    ——————
    2 ｜ 4
  */
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_vecs_per_head>(
      tid % 16, tid / 16);  // 16 * 16
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
#ifdef DEBUG_ATTN_C4
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf("before scale\n");
    T *q_smem_t = reinterpret_cast<T *>(qo_smem.base);
    for (uint32_t i = 0; i < num_frags_x * 16; ++i) {
      for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
        if (blockIdx.z == 0) {
          printf("q_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)(j),
                 (float)q_smem_t[i * num_frags_y * 16 + j]);
        } else {
          int res = q_smem_t[i * num_frags_y * 16 + j] + static_cast<T>(1.f);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  q_smem_inplace_multiply_sm_scale_multi_warps<num_frags_x, num_frags_y, T>(
      &qo_smem, scale);
#ifdef DEBUG_ATTN_C4
  if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0) {
    printf("after scale\n");
    T *q_smem_t = reinterpret_cast<T *>(qo_smem.base);
    for (uint32_t i = 0; i < num_frags_x * 16; ++i) {
      for (uint32_t j = 0; j < num_frags_y * 16; ++j) {
        if (blockIdx.z == 0) {
          printf("q_smem[%d][%d] = %f  ",
                 (int)i,
                 (int)(j),
                 (float)q_smem_t[i * num_frags_y * 16 + j]);
        } else {
          int res = q_smem_t[i * num_frags_y * 16 + j] + static_cast<T>(1.f);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

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

  // smem_t k_smem(smem + (num_frags_x + wid * num_frags_z) * 16 * HEAD_DIM *
  // sizeof(T)),
  //        v_smem(smem + (num_frags_x + (NUM_WARP_KV + wid) * num_frags_z) * 16
  //        * HEAD_DIM * sizeof(T));
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
#ifdef DEBUG_ATTN_C4
  if (tid == 0 && wid == 0 && kv_head_idx == 0) {
    printf(
        "batch_id: %d, tile_id: %d, chunk_size: %d, q_len: %d, kv_len: %d, "
        "chunk_start: %d, chunk_end: %d, num_iterations: %d, "
        "mask_check_iteration: %d\n",
        (int)batch_id,
        (int)tile_id,
        (int)chunk_size,
        (int)q_len,
        (int)kv_len,
        (int)chunk_start,
        (int)chunk_end,
        (int)num_iterations,
        (int)mask_check_iteration);
  }
  __syncthreads();
#endif
  /*
    1 ｜ 2
    ——————
    3 ｜ 4
  */
  uint32_t k_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * num_frags_z * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);
  /*
    1 ｜ 2
    ——————
    3 ｜ 4   transpose
  */
  uint32_t v_smem_offset_r =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * num_frags_y * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

  uint32_t k_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_head_k, inv_k_stride>(
          wid * 8 + tid / 4,
          tid %
              4);  // 4 * 128 / 8 = 64B, 128 nums, just fot head_dim >= 128 !!!
  uint32_t v_smem_offset_w =
      smem_t::get_permuted_offset<num_vecs_per_blocksize, inv_v_stride>(
          wid * 16 + tid / 2, tid % 2);  // 2 * 128 / 8 = 32B, 64 nums

  // uint32_t kv_idx_base = chunk_start;
  // int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  // const uint32_t const_offset = kv_head_idx * kv_h_stride + (wid * 4 + tid /
  // 8) * kv_b_stride + tid % 8 * num_elems_per_128b<T>();
  uint32_t kv_idx_base = chunk_start;
  // int block_id = __ldg(&block_table_now[kv_idx_base / BLOCK_SIZE]);
  const uint32_t const_k_offset = kv_head_idx * kv_h_stride +
                                  (wid * 8 + tid / 4) * kv_b_stride +
                                  tid % 4 * num_elems_per_128b<CacheT>();
  const uint32_t const_v_offset = kv_head_idx * kv_h_stride +
                                  (wid * 16 + tid / 2) * kv_d_stride +
                                  tid % 2 * num_elems_per_128b<CacheT>();
  // T *cache_k_now = cache_k + block_id * kv_n_stride + const_k_offset;
  // T *cache_v_now = cache_v + block_id * kv_n_stride + const_v_offset;

#ifdef DEBUG_ATTN_C4
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0) {
    printf(
        "ori q_smem_offset_r: %d, k_smem_offset_r: %d, v_smem_offset_r: %d, "
        "k_smem_offset_w: %d, v_smem_offset_w: %d, cache_k: %f, cache_k_p: %p, "
        "const_k_offset: %d, const_v_offset: %d\n",
        (int)q_smem_offset_r,
        (int)k_smem_offset_r,
        (int)v_smem_offset_r,
        (int)k_smem_offset_w,
        (int)v_smem_offset_w,
        (float)(*cache_k),
        cache_k,
        (int)const_k_offset,
        (int)const_v_offset);
  }
  __syncthreads();
#endif

  // load BLOCK_SIZE * HEAD_DIM each time
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
#ifdef DEBUG_ATTN_C4
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      printf("cache_k_smem\n");
      uint8_t *k_smem_t = reinterpret_cast<uint8_t *>(k_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_y * 16 / 2; ++j) {
          printf("k_smem[%d][%d] = %d  ",
                 (int)i,
                 (int)j,
                 (int)k_smem_t[i * num_frags_y * 16 / 2 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
    // s = qk
    compute_qk_c4<num_frags_x, num_frags_y, num_frags_z, T, CacheT>(
        &qo_smem,
        &q_smem_offset_r,
        &k_smem,
        &k_smem_offset_r,
        s_frag,
        cache_k_scale_frag,
        cache_k_zp_frag);
    // mask according to kv_idx and q_idx
    if (iter >= mask_check_iteration) {
      // if (q_len > 1 && iter >= mask_check_iteration) { // not need mask in
      // decoder, v will be filled with 0
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
#ifdef DEBUG_ATTN_C4
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("mask_s s_frag[%d][%d][%d]: %f  ",
                   (int)fx,
                   (int)fz,
                   (int)k,
                   s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

    // update m,d
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(
        s_frag, o_frag, m_frag, d_frag);
    __syncthreads();
#ifdef DEBUG_ATTN_C4
    if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          for (int k = 0; k < 8; k++) {
            printf("update_mdo_states s_frag[%d][%d][%d]: %f  ",
                   (int)fx,
                   (int)fz,
                   (int)k,
                   s_frag[fx][fz][k]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif

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
#ifdef DEBUG_ATTN_C4
    if (tid == PRINT_TID && wid == 0 && blockIdx.z == 0 &&
        blockIdx.x == gridDim.x - 1) {
      printf("cache_v_smem\n");
      uint8_t *v_smem_t = reinterpret_cast<uint8_t *>(v_smem.base);
      for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_y * 16; ++i) {
        for (uint32_t j = 0; j < num_frags_z * 16 / 2; ++j) {
          printf("v_smem[%d][%d] = %d  ",
                 (int)i,
                 (int)j,
                 (int)v_smem_t[i * num_frags_z * 16 / 2 + j]);
        }
        printf("\n");
      }
    }
    __syncthreads();
#endif
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
#ifdef DEBUG_ATTN_C4
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
      blockIdx.x == gridDim.x - 1) {
    printf("before merge z\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  merge_block_res_v2<num_frags_x, num_frags_y, T>(
      o_frag, reinterpret_cast<float *>(smem), m_frag, d_frag, wid, tid);
#ifdef DEBUG_ATTN_C4
  if (threadIdx.x == PRINT_TID && threadIdx.y == 0 && blockIdx.z == 0 &&
      blockIdx.x == gridDim.x - 1) {
    printf("after merge z\n");
    for (uint32_t i = 0; i < num_frags_x; ++i) {
      printf("m1: %f, m2: %f\n", m_frag[i][0], m_frag[i][1]);
      printf("d1: %f, d2: %f\n", d_frag[i][0], d_frag[i][1]);
      for (uint32_t j = 0; j < num_frags_y; ++j) {
        for (int r_id = 0; r_id < 8; r_id++) {
          printf("o_frag[%d][%d][%d]: %f ",
                 (int)i,
                 (int)j,
                 r_id,
                 o_frag[i][j][r_id]);
        }
      }
      printf("\n");
    }
  }
  __syncthreads();
#endif

  if (num_chunks_this_seq <= 1) {
    normalize_d<num_frags_x, num_frags_y>(o_frag, d_frag);
  }

  // write o
  // [num_frags_x, 16, num_frags_y, 16]
  if (num_chunks_this_seq <= 1) {
    write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE, num_frags_x, num_frags_y, false>(
      o_frag,
      &qo_smem,
      o_base_ptr_int8,
      shift_bias,
      smooth_weight,
      q_base_seq_id_this_block,
      q_head_idx,
      in_scale,
      q_len,
      q_n_stride, 
      HEAD_DIM
    );
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
        in_scale,
        q_len,
        q_n_stride * num_chunks,
        HEAD_DIM);
  // } else {
  //   write_o_reg_gmem_multi_warps_shift_smooth_quant<GROUP_SIZE,
  //                                                   num_frags_x,
  //                                                   num_frags_y,
  //                                                   partition_kv>(
  //       o_frag,
  //       &qo_smem,
  //       o_base_ptr_int8,
  //       shift_bias,
  //       smooth_weight,
  //       q_base_seq_id_this_block,
  //       q_head_idx,
  //       in_scale,
  //       q_len,
  //       partition_kv ? q_n_stride * num_chunks : q_n_stride,
  //       HEAD_DIM);
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
#ifdef DEBUG_ATTN_C4
          if (batch_id == 0) {
            printf(
                "bid: %d, tid: %d, wid: %d, q_base_seq_id_this_block: %d, "
                "qo_idx_now: %d, qo_idx: %d, q_start_seq_id: %d, q_len: %d, m: "
                "%f, d: %f\n",
                (int)batch_id,
                (int)tid,
                (int)wid,
                (int)q_base_seq_id_this_block,
                (int)qo_idx_now,
                (int)qo_idx,
                (int)q_start_seq_id,
                (int)q_len,
                (float)m_frag[fx][j],
                (float)d_frag[fx][j]);
          }
#endif
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
          int vec_size,
          uint32_t bdy,
          uint32_t HEAD_DIM,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
__global__ void merge_multi_chunks_decoder_kernel(
    const T *__restrict__ multi_out,    // [token_num, num_chunks, num_heads,
                                        // head_dim]
    const float *__restrict__ multi_m,  // [token_num, num_chunks, num_heads]
    const float *__restrict__ multi_d,  // [token_num, num_chunks, num_heads]
    const int *__restrict__ seq_lens_q,
    const int *__restrict__ seq_lens_kv,
    const int *__restrict__ seq_lens_encoder,
    const int *__restrict__ cum_offsets,
    const T *__restrict__ shift_bias,     // [q_num_heads * HEAD_DIM]
    const T *__restrict__ smooth_weight,  // [q_num_heads * HEAD_DIM]
    OutT *__restrict__ out,
    const float in_scale,
    const int max_seq_len,
    const int num_chunks,
    const int num_heads,
    const int chunk_size,
    const int head_dim) {
  const int vid = threadIdx.x, ty = threadIdx.y;
  const int bid = blockIdx.x, hid = blockIdx.y;
  __shared__ T smem[bdy * HEAD_DIM];
  __shared__ float md_smem[bdy * 2];
  const int start_token_idx = bid * max_seq_len - cum_offsets[bid];
  const int seq_len_q = seq_lens_q[bid];
  if (seq_len_q == 0) return;
  int seq_len_kv = seq_lens_kv[bid];

  if (ENABLE_PREFILL) {
    seq_len_kv += seq_len_q;
    if (seq_len_kv == 0) return;
  } else {
    if (seq_len_kv == 0) return;
    seq_len_kv += seq_len_q;
  }
  const int seq_len_enc = seq_lens_encoder[bid];
  if (seq_len_enc > 0) {
    return;
  }
  const int num_chunks_this_seq = div_up(seq_len_kv, chunk_size);
  if (num_chunks_this_seq <= 1) {
    return;
  }

  using LoadT = AlignedVector<T, vec_size>;
  LoadT load_vec;
  LoadT res_vec;
  if constexpr (std::is_same<T, half>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((half2 *)(&res_vec) + i) = make_half2(0, 0);
    }
  } else {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((nv_bfloat162 *)(&res_vec) + i) = make_bfloat162(0, 0);
    }
  }
  float m;
  float d = 1.f;
  if constexpr (std::is_same<T, half>::value) {
    m = -5e4f;
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    m = -3.0e+30f;
  }
#pragma unroll 2
  for (int i = ty; i < num_chunks_this_seq; i += bdy) {
    // uint32_t offset = (start_token_idx * num_chunks + i) * num_heads + hid;
    uint32_t offset = (bid * num_chunks + i) * num_heads + hid;
    float m_prev = m;
    float d_prev = d;
    const float m_now = multi_m[offset];
    const float d_now = multi_d[offset];
    m = max(m_prev, m_now);
    // offset = (start_token_idx * num_chunks * num_heads + i * num_heads + hid)
    // * head_dim + vid * vec_size;
    offset = (bid * num_chunks * num_heads + i * num_heads + hid) * head_dim +
             vid * vec_size;
    Load<T, vec_size>(&multi_out[offset], &load_vec);
    const float scale1 = __expf(m_prev - m), scale2 = __expf(m_now - m);
    const T scale1_T = static_cast<T>(scale1),
            scale2_T = static_cast<T>(scale2);
    d = d * scale1 + d_now * scale2;
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      res_vec[j] = res_vec[j] * scale1_T + load_vec[j] * scale2_T;
    }
  }
  // store ty res
  Store<T, vec_size>(res_vec, &smem[ty * head_dim + vid * vec_size]);
  md_smem[2 * ty] = m;
  md_smem[2 * ty + 1] = d;
  __syncthreads();
  if (ty == 0) {
    // merge bdy
    prefill_softmax_state_t<vec_size, T> st;
    st.init();
#pragma unroll
    for (int i = 0; i < bdy; i++) {
      Load<T, vec_size>(&smem[i * head_dim + vid * vec_size], &load_vec);
      const float m_tmp = md_smem[2 * i], d_tmp = md_smem[2 * i + 1];
      st.merge(load_vec, m_tmp, d_tmp);
    }
    st.normalize();

    const uint32_t shift_smooth_offset = hid * head_dim + vid * vec_size;
    AlignedVector<T, vec_size> shift_bias_vec;
    AlignedVector<T, vec_size> smooth_weight_vec;
    AlignedVector<OutT, vec_size> out_vec;
    if (shift_bias) {
      Load<T, vec_size>(shift_bias + shift_smooth_offset, &shift_bias_vec);
      Load<T, vec_size>(smooth_weight + shift_smooth_offset,
                        &smooth_weight_vec);
    }
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      // float quant_value  = 127.0f * static_cast<float>((st.o[i] +
      // shift_bias_vec[i]) * smooth_weight_vec[i]) * in_scale; quant_value =
      // rintf(quant_value); quant_value = quant_value > 127.0f ? 127.0f :
      // quant_value; quant_value = quant_value < -127.0f ? -127.0f :
      // quant_value; out_vec[i] = static_cast<int8_t>(quant_value);
      StoreFunc<T, vec_size, OutT>()(
          st.o, shift_bias_vec, smooth_weight_vec, out_vec, in_scale, i);
    }
    Store<OutT, vec_size>(
        out_vec,
        &out[(start_token_idx * num_heads + hid) * head_dim + vid * vec_size]);
  }
}


template <typename T,
          int vec_size,
          uint32_t bdy,
          uint32_t HEAD_DIM,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
__global__ void merge_multi_chunks_v2_kernel(
    const T *__restrict__ multi_out,    // [token_num, num_chunks, num_heads,
                                        // head_dim]
    const float *__restrict__ multi_m,  // [token_num, num_chunks, num_heads]
    const float *__restrict__ multi_d,  // [token_num, num_chunks, num_heads]
    const int *__restrict__ seq_lens_q,
    const int *__restrict__ seq_lens_kv,
    const int *__restrict__ seq_lens_encoder,
    const int *__restrict__ padding_offsets,
    const T *__restrict__ shift_bias,     // [q_num_heads * HEAD_DIM]
    const T *__restrict__ smooth_weight,  // [q_num_heads * HEAD_DIM]
    OutT *__restrict__ out,
    const float in_scale,
    const int max_seq_len,
    const int num_chunks,
    const int num_heads,
    const int chunk_size,
    const int head_dim,
    const int token_num,
    const int speculate_max_draft_token_num = 5) {
  const int vid = threadIdx.x, ty = threadIdx.y;
  // const int qid = blockIdx.x, hid = blockIdx.y;
  const int hid = blockIdx.y;
  __shared__ T smem[bdy * HEAD_DIM];
  __shared__ float md_smem[bdy * 2];
  for (int qid = blockIdx.x; qid < token_num; qid += gridDim.x) {
    const uint32_t ori_token_id = qid + padding_offsets[qid];
    const uint32_t bid = ori_token_id / max_seq_len;
    const uint32_t local_seq_id = ori_token_id % max_seq_len;
    const int seq_len_q = seq_lens_q[bid];
    if (seq_len_q == 0) continue;
    int seq_len_kv = seq_lens_kv[bid];
    if (ENABLE_PREFILL) {
      seq_len_kv += seq_len_q;
      if (seq_len_kv == 0) continue;

      const int seq_len_enc = seq_lens_encoder[bid];
      if (seq_len_enc <= 0) {
        continue;
      }
    } else {
      if (seq_len_kv == 0) continue;
      seq_len_kv += seq_len_q;
    }
    const int num_chunks_this_seq = div_up(seq_len_kv, chunk_size);
    if (num_chunks_this_seq <= 1) {
      continue;
    }

    using LoadT = AlignedVector<T, vec_size>;
    LoadT load_vec;
    LoadT res_vec;
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((half2 *)(&res_vec) + i) = make_half2(0, 0);
      }
    } else {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((nv_bfloat162 *)(&res_vec) + i) = make_bfloat162(0, 0);
      }
    }
    float m;
    float d = 1.f;
    if constexpr (std::is_same<T, half>::value) {
      m = -5e4f;
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      m = -3.0e+30f;
    }
#pragma unroll 2
    for (int i = ty; i < num_chunks_this_seq; i += bdy) {
      uint32_t offset;
      if (ENABLE_PREFILL) {
        offset = (qid * num_chunks + i) * num_heads + hid;
      } else {
        offset =
            ((bid * speculate_max_draft_token_num + local_seq_id) * num_chunks +
             i) *
                num_heads +
            hid;
      }
      float m_prev = m;
      float d_prev = d;
      const float m_now = multi_m[offset];
      const float d_now = multi_d[offset];
      m = max(m_prev, m_now);
      if (ENABLE_PREFILL) {
        offset =
            (qid * num_chunks * num_heads + i * num_heads + hid) * head_dim +
            vid * vec_size;
      } else {
        offset = ((bid * speculate_max_draft_token_num + local_seq_id) *
                      num_chunks * num_heads +
                  i * num_heads + hid) *
                     head_dim +
                 vid * vec_size;
      }
      Load<T, vec_size>(&multi_out[offset], &load_vec);
      const float scale1 = __expf(m_prev - m), scale2 = __expf(m_now - m);
      const T scale1_T = static_cast<T>(scale1),
              scale2_T = static_cast<T>(scale2);
      d = d * scale1 + d_now * scale2;
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        res_vec[j] = res_vec[j] * scale1_T + load_vec[j] * scale2_T;
      }
    }
    // store ty res
    Store<T, vec_size>(res_vec, &smem[ty * head_dim + vid * vec_size]);
    md_smem[2 * ty] = m;
    md_smem[2 * ty + 1] = d;
    __syncthreads();
    if (ty == 0) {
      // merge bdy
      prefill_softmax_state_t<vec_size, T> st;
      st.init();
#pragma unroll
      for (int i = 0; i < bdy; i++) {
        Load<T, vec_size>(&smem[i * head_dim + vid * vec_size], &load_vec);
        const float m_tmp = md_smem[2 * i], d_tmp = md_smem[2 * i + 1];
        st.merge(load_vec, m_tmp, d_tmp);
      }
      st.normalize();

      const uint32_t shift_smooth_offset = hid * head_dim + vid * vec_size;
      AlignedVector<T, vec_size> shift_bias_vec;
      AlignedVector<T, vec_size> smooth_weight_vec;
      AlignedVector<OutT, vec_size> out_vec;
      if (shift_bias) {
        Load<T, vec_size>(shift_bias + shift_smooth_offset, &shift_bias_vec);
        Load<T, vec_size>(smooth_weight + shift_smooth_offset,
                          &smooth_weight_vec);
      }
#pragma unroll
      for (int i = 0; i < vec_size; ++i) {
        // float quant_value  = 127.0f * static_cast<float>((st.o[i] +
        // shift_bias_vec[i]) * smooth_weight_vec[i]) * in_scale; quant_value =
        // rintf(quant_value); quant_value = quant_value > 127.0f ? 127.0f :
        // quant_value; quant_value = quant_value < -127.0f ? -127.0f :
        // quant_value; out_vec[i] = static_cast<int8_t>(quant_value);

        StoreFunc<T, vec_size, OutT>()(
            st.o, shift_bias_vec, smooth_weight_vec, out_vec, in_scale, i);
      }
      Store<OutT, vec_size>(
          out_vec, &out[(qid * num_heads + hid) * head_dim + vid * vec_size]);
    }
    __syncthreads();
  }
}

template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT,
          bool ENABLE_PREFILL = true>
void MultiQueryAppendAttention(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& shift_bias,
    const paddle::optional<paddle::Tensor>& smooth_weight,
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;

  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_nums;
  auto bsz = meta_data.batch_size;
  auto max_block_num_per_seq = meta_data.max_blocks_per_seq;

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q);  // 1 or 2
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  auto* allocator = paddle::GetAllocator(qkv.place());

  const float scale = 1.f / sqrt(HEAD_DIM);

  if constexpr (NUM_WARP_Q == 4) {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16;  // !!!
    // constexpr uint32_t num_frags_z = 8; // 128 per iter, 4 is better?
    constexpr uint32_t smem_size =
        (num_warps * num_frags_x + NUM_WARP_KV * num_frags_z * 2) * 16 *
        HEAD_DIM * sizeof(T);
    auto split_kv_kernel = multi_query_append_attention_kernel<NV_TYPE,
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
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_kernel<NV_TYPE,
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                        const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                        const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                        const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                            const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV;  // !!!
    constexpr uint32_t smem_size =
        (num_frags_x + NUM_WARP_KV * num_frags_z * 2) * 16 * HEAD_DIM *
        sizeof(T);
    auto split_kv_kernel =
        multi_query_append_attention_warp1_4_kernel<NV_TYPE,
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
    // int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);

    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    // dim3 grids(num_blocks_x_cpu, num_chunks, 1);
    dim3 blocks(32, num_warps);

    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_warp1_4_kernel<NV_TYPE,
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                            const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);

      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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

template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
void MultiQueryAppendC8Attention(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::Tensor& cache_k_scale,
    const paddle::Tensor& cache_v_scale,
    const paddle::optional<paddle::Tensor>& shift_bias,
    const paddle::optional<paddle::Tensor>& smooth_weight,
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;

  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_nums;
  auto bsz = meta_data.batch_size;
  auto max_block_num_per_seq = meta_data.max_blocks_per_seq;

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q);  // 1 or 2
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  auto* allocator = paddle::GetAllocator(qkv.place());

  const float scale = 1.f / sqrt(HEAD_DIM);

  if constexpr (NUM_WARP_Q == 4) {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16;  // !!!
    constexpr uint32_t smem_size =
        num_warps * num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        num_frags_z * 16 * HEAD_DIM * sizeof(uint8_t) * 2;
    auto split_kv_kernel =
        multi_query_append_attention_c8_kernel<NV_TYPE,
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
    // int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_c8_kernel<NV_TYPE,
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV * 2;  // !!!
    constexpr uint32_t smem_size =
        num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(uint8_t) * 2;
    auto split_kv_kernel =
        multi_query_append_attention_c8_warp1_4_kernel<NV_TYPE,
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
                                                       OUT_NV_TYPE>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    // int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }

    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    // dim3 grids(num_blocks_x_cpu, num_chunks, 1);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_c8_warp1_4_kernel<NV_TYPE,
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE, vec_size, blocky, HEAD_DIM>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::Tensor& cache_k_scale,
    const paddle::Tensor& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& shift_bias,
    const paddle::optional<paddle::Tensor>& smooth_weight,
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;

  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto token_num = meta_data.token_nums;
  auto bsz = meta_data.batch_size;
  auto max_block_num_per_seq = meta_data.max_blocks_per_seq;

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q);  // 1 or 2
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  auto* allocator = paddle::GetAllocator(qkv.place());

  const float scale = 1.f / sqrt(HEAD_DIM);

  if constexpr (NUM_WARP_Q == 4) {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16;  // !!!
    // constexpr uint32_t num_frags_z = 8; // 128 per iter, 4 is better?
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
    // if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(split_kv_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    // }
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

    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_k_zp.get().data<T>())) : nullptr,
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_v_zp.get().data<T>())) : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_k_zp.get().data<T>())) : nullptr,
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_v_zp.get().data<T>())) : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV * 4;  // !!!
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


    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    // dim3 grids(num_blocks_x_cpu, num_chunks, 1);
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_k_zp.get().data<T>())) : nullptr,
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_v_zp.get().data<T>())) : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_k_zp.get().data<T>())) : nullptr,
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_v_zp.get().data<T>())) : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>())) : nullptr,
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
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>())) : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>())) : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
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
