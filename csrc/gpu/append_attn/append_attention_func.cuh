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

template <typename T>
__forceinline__ __device__ float fixed_expf(float x1, float x2) {
  if constexpr (std::is_same<T, half>::value) {
    if (x1 == -5e4f) {
      return 0;
    } else {
      return __expf(x1 - x2);
    }
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    if (x1 == -3.0e+30f) {
      return 0;
    } else {
      return __expf(x1 - x2);
    }
  }
}

template <size_t vec_size, typename T>
struct prefill_softmax_state_t {
  AlignedVector<T, vec_size> o;
  float m;
  float d;

  __device__ __forceinline__ void init() {
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((half2*)(&o) + i) = make_half2(0, 0);
      }
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
#pragma unroll
      for (int i = 0; i < vec_size / 2; ++i) {
        *((nv_bfloat162*)(&o) + i) = make_bfloat162(0, 0);
      }
    }
    d = 1.f;
    if constexpr (std::is_same<T, half>::value) {
      m = -5e4f;
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
      m = -3.38953e38f;
    }
  }

  __device__ __forceinline__ void merge(
      const AlignedVector<T, vec_size>& other_o, float other_m, float other_d) {
    float m_prev = m, d_prev = d;
    m = m_prev > other_m ? m_prev : other_m;
    const float scale1 = __expf(m_prev - m), scale2 = __expf(other_m - m);
    const T scale1_T = static_cast<T>(scale1),
            scale2_T = static_cast<T>(scale2);
    d = d_prev * scale1 + other_d * scale2;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * scale1_T + other_o[i] * scale2_T;
    }
  }

  __device__ __forceinline__ void normalize() {
    const T d_t = static_cast<T>(d);
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] /= d_t;
    }
  }
};

template <typename T, uint32_t num_frags_x, uint32_t num_frags_y>
__device__ __forceinline__ void init_states(float (*o_frag)[num_frags_y][8],
                                            float (*m)[2],
                                            float (*d)[2]) {
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if constexpr (std::is_same<T, half>::value) {
        m[fx][j] = -5e4f;
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        m[fx][j] = -3.0e+30f;
      }
      d[fx][j] = 1.f;
    }
  }
}

template <uint32_t group_size,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t HEAD_DIM,
          typename T>
__device__ __forceinline__ void load_q_global_smem_multi_warps(
    T* q_ptr_base,
    smem_t* q_smem,
    uint32_t q_idx_base,
    const uint32_t qo_upper_bound,
    const uint32_t qo_n_stride,
    const uint32_t qo_h_stride) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();

  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t q_smem_offset_w =  // [NUM_WARP_Q, num_frags_x, 16, head_dim]
      smem_t::get_permuted_offset<num_vecs_per_head>(ty * 4 + tx / 8,
                                                     tx % 8);  // 4 * 64

  const uint32_t tx_offset = tx / 8;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {

    const uint32_t base_offset = q_idx_base + fx * 16 + tx_offset;
#pragma unroll
    const int j = ty;
    const uint32_t offset_now = base_offset + j * 4;
    const uint32_t n_offset = offset_now / group_size;
    const uint32_t h_offset = offset_now % group_size;
    T* q_ptr = q_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
#pragma unroll
    for (uint32_t fyo = 0; fyo < num_frags_y / 4;
         ++fyo) {
      q_smem->load_128b_async<SharedMemFillMode::kNoFill>(
          q_smem_offset_w, q_ptr, n_offset < qo_upper_bound);
      q_smem_offset_w =
          q_smem->advance_offset_by_column<8>(q_smem_offset_w, fyo);
      q_ptr += 8 * num_elems_per_128b<T>();
    }
    q_smem_offset_w =
        q_smem->advance_offset_by_row<16, num_vecs_per_head>(q_smem_offset_w) -
        2 * num_frags_y;
  }
}

template <uint32_t group_size,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t HEAD_DIM,
          typename T>
__device__ __forceinline__ void load_q_global_smem(
    T* q_ptr_base,
    smem_t* q_smem,
    uint32_t q_idx_base,
    const uint32_t qo_upper_bound,
    const uint32_t qo_n_stride,
    const uint32_t qo_h_stride) {
  constexpr uint32_t num_vecs_per_head = HEAD_DIM / num_elems_per_128b<T>();

  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

  uint32_t q_smem_offset_w =  // [NUM_WARP_Q, num_frags_x, 16, head_dim]
      smem_t::get_permuted_offset<num_vecs_per_head>(
          ty * num_frags_x * 16 + tx / 8, tx % 8);  // 4 * 64

  const uint32_t tx_offset = tx / 8;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    const uint32_t base_offset = q_idx_base + fx * 16 + tx_offset;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t offset_now = base_offset + j * 4;
      const uint32_t n_offset = offset_now / group_size;
      const uint32_t h_offset = offset_now % group_size;
      T* q_ptr = q_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
#pragma unroll
      for (uint32_t fyo = 0; fyo < num_frags_y / 4;
           ++fyo) {
        q_smem->load_128b_async<SharedMemFillMode::kNoFill>(
            q_smem_offset_w, q_ptr, n_offset < qo_upper_bound);
        q_smem_offset_w =
            q_smem->advance_offset_by_column<8>(q_smem_offset_w, fyo);
        q_ptr += 8 * num_elems_per_128b<T>();
      }
      q_smem_offset_w =
          q_smem->advance_offset_by_row<4, num_vecs_per_head>(q_smem_offset_w) -
          2 * num_frags_y;  // num_frags_y / 4 * 8
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale_multi_warps(
    smem_t* q_smem,  // [num_frags_x * 16, num_frags_y * 16]
    const float sm_scale) {
  constexpr int vec_size = 16 / sizeof(T);
  using LoadT = AlignedVector<T, vec_size>;
  LoadT tmp_vec;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();

#pragma unroll
  for (uint32_t i = 0; i < num_frags_x * 16 * head_dim / 1024;
       ++i) {
    const int offset = i * 1024 + ty * 256 + tx * 8;
    Load<T, vec_size>(reinterpret_cast<T*>(q_smem->base) + offset, &tmp_vec);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp_vec[reg_id] *= sm_scale;
    }
    Store<T, vec_size>(tmp_vec, reinterpret_cast<T*>(q_smem->base) + offset);
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(
    smem_t* q_smem,  // [num_frags_x * 16, num_frags_y * 16]
    const float sm_scale) {
  constexpr int vec_size = 16 / sizeof(T);
  using LoadT = AlignedVector<T, vec_size>;
  LoadT tmp_vec;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();

#pragma unroll
  for (uint32_t i = 0; i < num_frags_x * 16 * head_dim / 256;
       ++i) {  // 32 * 8 per warp
    Load<T, vec_size>(
        reinterpret_cast<T*>(q_smem->base +
                             ty * num_frags_x * 16 * num_vecs_per_head) +
            i * 256 + tx * 8,
        &tmp_vec);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp_vec[reg_id] *= sm_scale;
    }
    Store<T, vec_size>(
        tmp_vec,
        reinterpret_cast<T*>(q_smem->base +
                             ty * num_frags_x * 16 * num_vecs_per_head) +
            i * 256 + tx * 8);
  }
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t block_size,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename T>
__device__ __forceinline__ void produce_kv_blockwise(
    smem_t smem,
    uint32_t* smem_offset,
    T** gptr,  // [max_block_num, num_heads, block_size, head_dim]
    const uint32_t kv_head_idx,
    const uint32_t kv_n_stride,
    const uint32_t kv_h_stride,
    const uint32_t kv_b_stride,
    const uint32_t kv_idx_base,
    const uint32_t kv_len) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8;  // kv_idx used to check
#pragma unroll
  for (uint32_t i = 0; i < NUM_WARP_KV * num_frags_z * 4 / num_warps;
       ++i) {
#pragma unroll
    for (uint32_t j = 0; j < num_frags_y / 4;
         ++j) {
      smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
      *gptr += 8 * num_elems_per_128b<T>();
    }
    kv_idx += num_warps * 4;
    *smem_offset = smem.advance_offset_by_row<num_warps * 4, num_vecs_per_head>(
                       *smem_offset) -
                   2 * num_frags_y;  // num_frags_y / 4 * 8
    *gptr +=
        num_warps * 4 * kv_b_stride - 2 * num_frags_y * num_elems_per_128b<T>();
  }
  *gptr -= NUM_WARP_KV * num_frags_z * 16 * kv_b_stride;
  *smem_offset -= NUM_WARP_KV * num_frags_z * 16 * num_vecs_per_head;
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t block_size,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename CacheT>
__device__ __forceinline__ void produce_v_blockwise_c8(
    smem_t smem,
    uint32_t* smem_offset,
    CacheT* cache_v,
    const int* block_table_now,
    const uint32_t kv_head_idx,
    const uint32_t kv_n_stride,
    const uint32_t kv_h_stride,
    const uint32_t kv_d_stride,
    const uint32_t kv_idx_base,
    const uint32_t kv_len,
    const uint32_t const_v_offset) {
  constexpr uint32_t num_vecs_per_blocksize =
      block_size / num_elems_per_128b<CacheT>();  // 8
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx =
      kv_idx_base +
      tx % 4 * num_elems_per_128b<CacheT>();
  if constexpr (NUM_WARP_Q == 4) {
    int block_id = __ldg(&block_table_now[kv_idx / block_size]);
    if (block_id < 0) block_id = 0;
    CacheT* cache_v_now = cache_v + block_id * kv_n_stride + const_v_offset;
#pragma unroll
    for (uint32_t i = 0; i < num_frags_y * 2 / num_warps;
         ++i) {  // m (num_frags_y * 16 / (num_warps * 8))
#pragma unroll
      for (uint32_t j = 0; j < num_frags_z / 4;
           ++j) { 
        smem.load_128b_async<fill_mode>(*smem_offset, cache_v_now, true);
        *smem_offset = smem.advance_offset_by_column<4, num_vecs_per_blocksize>(
            *smem_offset, j);
        cache_v_now += 4 * num_elems_per_128b<CacheT>();
      }
      *smem_offset =
          smem.advance_offset_by_row<num_warps * 8, num_vecs_per_blocksize>(
              *smem_offset) -
          num_frags_z;  // num_frags_z / 4 * 4
      cache_v_now += num_warps * 8 * kv_d_stride -
                     num_frags_z * num_elems_per_128b<CacheT>();
    }
    *smem_offset -= num_frags_y * 16 * num_vecs_per_blocksize;
  } else {
#pragma unroll
    for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
      int block_id = __ldg(&block_table_now[kv_idx / block_size]);
      if (block_id < 0) block_id = 0;
      CacheT* cache_v_now = cache_v + block_id * kv_n_stride + const_v_offset;

#pragma unroll
      for (uint32_t i = 0; i < num_frags_y * 2 / num_warps;
           ++i) {  // m (num_frags_y * 16 / (num_warps * 8))
#pragma unroll
        for (uint32_t j = 0; j < 2 * num_frags_z / 4;
             ++j) {
          smem.load_128b_async<fill_mode>(*smem_offset, cache_v_now, true);
          *smem_offset =
              smem.advance_offset_by_column<4, num_vecs_per_blocksize>(
                  *smem_offset, j);
          cache_v_now += 4 * num_elems_per_128b<CacheT>();
          kv_idx += 4 * num_elems_per_128b<CacheT>();
        }
        kv_idx -= 2 * num_frags_z * num_elems_per_128b<CacheT>();
        *smem_offset =
            smem.advance_offset_by_row<num_warps * 8, num_vecs_per_blocksize>(
                *smem_offset) -
            2 * num_frags_z;  // num_frags_z / 4 * 4
        cache_v_now += num_warps * 8 * kv_d_stride -
                       2 * num_frags_z * num_elems_per_128b<CacheT>();
      }
      kv_idx += block_size;
    }
    *smem_offset -= NUM_WARP_KV / 2 * num_frags_y * 16 * num_vecs_per_blocksize;
  }
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t block_size,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename CacheT>
__device__ __forceinline__ void produce_k_blockwise_c8(
    smem_t smem,
    uint32_t* smem_offset,
    CacheT* cache_k,
    const int* block_table_now,
    const uint32_t kv_head_idx,
    const uint32_t kv_n_stride,
    const uint32_t kv_h_stride,
    const uint32_t kv_b_stride,
    const uint32_t kv_idx_base,
    const uint32_t kv_len,
    const uint32_t const_k_offset) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head =
      head_dim / num_elems_per_128b<CacheT>();  // 8
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8;
  if constexpr (NUM_WARP_Q == 4) {
    int block_id = __ldg(&block_table_now[kv_idx / block_size]);
    if (block_id < 0) block_id = 0;
    CacheT* cache_k_now = cache_k + block_id * kv_n_stride + const_k_offset;
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 4 / num_warps;
         ++i) {  // m num_frags_z * 16 / (num_warps * 4)
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / 8;
           ++j) {  // k num_frags_y * 16 / 8 / num_elems_per_128b<CacheT>()
        // smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx <
        // kv_len);
        smem.load_128b_async<fill_mode>(*smem_offset, cache_k_now, true);
        *smem_offset = smem.advance_offset_by_column<8, num_vecs_per_head>(
            *smem_offset, j);
        cache_k_now += 8 * num_elems_per_128b<CacheT>();
      }
      // kv_idx += num_warps * 4;
      *smem_offset =
          smem.advance_offset_by_row<num_warps * 4, num_vecs_per_head>(
              *smem_offset) -
          num_frags_y;  // num_frags_y / 4 * 4
      cache_k_now += num_warps * 4 * kv_b_stride -
                     num_frags_y * num_elems_per_128b<CacheT>();
    }
    *smem_offset -= num_frags_z * 16 * num_vecs_per_head;
  } else {
#pragma unroll
    for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV / 2; ++kv_i) {
      int block_id = __ldg(&block_table_now[kv_idx / block_size]);
      if (block_id < 0) block_id = 0;
      CacheT* cache_k_now = cache_k + block_id * kv_n_stride + const_k_offset;
#pragma unroll
      for (uint32_t i = 0; i < 2 * num_frags_z * 4 / num_warps;
           ++i) {  // m num_frags_z * 16 / (num_warps * 4)
#pragma unroll
        for (uint32_t j = 0; j < num_frags_y / 8;
             ++j) {
          smem.load_128b_async<fill_mode>(*smem_offset, cache_k_now, true);
          *smem_offset = smem.advance_offset_by_column<8, num_vecs_per_head>(
              *smem_offset, j);
          cache_k_now += 8 * num_elems_per_128b<CacheT>();
        }
        kv_idx += num_warps * 4;
        *smem_offset =
            smem.advance_offset_by_row<num_warps * 4, num_vecs_per_head>(
                *smem_offset) -
            num_frags_y;  // num_frags_y / 4 * 4
        cache_k_now += num_warps * 4 * kv_b_stride -
                       num_frags_y * num_elems_per_128b<CacheT>();
      }
    }
    *smem_offset -= NUM_WARP_KV * num_frags_z * 16 * num_vecs_per_head;
  }
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t block_size,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename CacheT>
__device__ __forceinline__ void produce_v_blockwise_c4(
    smem_t smem,
    uint32_t* smem_offset,
    CacheT* cache_v,
    const int* block_table_now,
    const uint32_t kv_head_idx,
    const uint32_t kv_n_stride,
    const uint32_t kv_h_stride,
    const uint32_t kv_d_stride,
    const uint32_t kv_idx_base,
    const uint32_t kv_len,
    const uint32_t const_v_offset) {
  constexpr uint32_t num_vecs_per_blocksize =
      block_size / 2 / num_elems_per_128b<CacheT>();  // 2
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  uint32_t kv_idx =
      kv_idx_base +
      tx % 2 * 2 * num_elems_per_128b<CacheT>();  // kv_idx used to check
#pragma unroll
  for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV; ++kv_i) {
    int block_id = __ldg(&block_table_now[(kv_idx) / block_size]);
    if (block_id < 0) block_id = 0;
    CacheT* cache_v_now = cache_v + block_id * kv_n_stride + const_v_offset;
#pragma unroll
    for (uint32_t i = 0; i < num_frags_y / num_warps; ++i) {  // m
#pragma unroll
      for (uint32_t j = 0; j < num_frags_z / 4;
           ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, cache_v_now, true);
        *smem_offset = smem.advance_offset_by_column<2, num_vecs_per_blocksize>(
            *smem_offset, j);
        cache_v_now += 2 * num_elems_per_128b<CacheT>();
        kv_idx += 4 * num_elems_per_128b<CacheT>();
      }
      kv_idx -= num_frags_z * num_elems_per_128b<CacheT>();
      *smem_offset =
          smem.advance_offset_by_row<num_warps * 16, num_vecs_per_blocksize>(
              *smem_offset) -
          num_frags_z / 2;  // num_frags_y / 4 * 2
      cache_v_now += num_warps * 16 * kv_d_stride -
                     num_frags_z / 2 * num_elems_per_128b<CacheT>();
    }
    kv_idx += block_size;
  }
  *smem_offset -= NUM_WARP_KV * num_frags_y * 16 * num_vecs_per_blocksize;
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t block_size,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename CacheT>
__device__ __forceinline__ void produce_k_blockwise_c4(
    smem_t smem,
    uint32_t* smem_offset,
    CacheT* cache_k,
    const int* block_table_now,
    const uint32_t kv_head_idx,
    const uint32_t kv_n_stride,
    const uint32_t kv_h_stride,
    const uint32_t kv_b_stride,
    const uint32_t kv_idx_base,
    const uint32_t kv_len,
    const uint32_t const_k_offset) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head =
      head_dim / 2 / num_elems_per_128b<CacheT>();  // 4
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx = kv_idx_base + ty * 8 + tx / 4;  // kv_idx used to check

#pragma unroll
  for (uint32_t kv_i = 0; kv_i < NUM_WARP_KV; ++kv_i) {
    int block_id = __ldg(&block_table_now[kv_idx / block_size]);
    if (block_id < 0) block_id = 0;
    CacheT* cache_k_now = cache_k + block_id * kv_n_stride + const_k_offset;

#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 2 / num_warps;
         ++i) {  // m num_frags_z * 16 / (num_warps * 8)
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / 8;
           ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, cache_k_now, true);
        *smem_offset = smem.advance_offset_by_column<4, num_vecs_per_head>(
            *smem_offset, j);
        cache_k_now += 4 * num_elems_per_128b<CacheT>();
      }
      kv_idx += num_warps * 8;
      *smem_offset =
          smem.advance_offset_by_row<num_warps * 8, num_vecs_per_head>(
              *smem_offset) -
          num_frags_y / 2;  // num_frags_y / 8 * 4
      cache_k_now += num_warps * 8 * kv_b_stride -
                     num_frags_y / 2 * num_elems_per_128b<CacheT>();
    }
  }
  *smem_offset -= NUM_WARP_KV * num_frags_z * 16 * num_vecs_per_head;
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t block_size,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t NUM_WARP_Q,
          typename T>
__device__ __forceinline__ void block_produce_kv(
    smem_t smem,
    uint32_t* smem_offset,
    T* gptr_base,  // [max_block_num, num_heads, block_size, head_dim]
    const int* block_table,
    const uint32_t kv_head_idx,
    const uint32_t kv_n_stride,
    const uint32_t kv_h_stride,
    const uint32_t kv_b_stride,
    const uint32_t kv_idx_base,
    const uint32_t kv_len) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  if constexpr (NUM_WARP_Q == 4) {
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 4 / num_warps;
         ++i) {  // m num_frags_z * 16 / (num_warps * 4)
      const uint32_t row_now =
          kv_idx_base + (i * 4 * num_warps + ty * 4 + tx / 8);
      const uint32_t kv_n_idx = row_now / block_size;
      const uint32_t kv_bid = row_now % block_size;
      T* gptr = gptr_base + __ldg(&block_table[kv_n_idx]) * kv_n_stride +
                kv_head_idx * kv_h_stride + kv_bid * kv_b_stride +
                tx % 8 * num_elems_per_128b<T>();
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / 4;
           ++j) {  // k num_frags_y * 16 / 8 / num_elems_per_128b<T>()
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, row_now < kv_len);
        *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * num_elems_per_128b<T>();
      }
      *smem_offset =
          smem.advance_offset_by_row<num_warps * 4, num_vecs_per_head>(
              *smem_offset) -
          2 * num_frags_y;  // num_frags_y / 4 * 8
    }
    *smem_offset -= num_frags_z * 16 * num_vecs_per_head;
  } else {
    const uint32_t row_id_per_tx = tx / 8;
    const uint32_t col_id_per_tx = tx % 8;
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z;
         ++i) {  // m num_warps * num_frags_z * 16
#pragma unroll
      for (uint32_t j = 0; j < 4; j++) {
        const uint32_t row_now = kv_idx_base + (i * 16 + j * 4 + row_id_per_tx);
        const uint32_t kv_n_idx = row_now / block_size;
        const uint32_t kv_bid = row_now % block_size;
        T* gptr = gptr_base + __ldg(&block_table[kv_n_idx]) * kv_n_stride +
                  kv_head_idx * kv_h_stride + kv_bid * kv_b_stride +
                  col_id_per_tx * num_elems_per_128b<T>();
#pragma unroll
        for (uint32_t fy = 0; fy < num_frags_y / 4;
             ++fy) {  // k num_frags_y * 16 / 8 / num_elems_per_128b<T>()
          smem.load_128b_async<fill_mode>(*smem_offset, gptr, row_now < kv_len);
          *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, fy);
          gptr += 8 * num_elems_per_128b<T>();
        }
        *smem_offset =
            smem.advance_offset_by_row<4, num_vecs_per_head>(*smem_offset) -
            2 * num_frags_y;  // num_frags_y / 4 * 8
      }
    }
    *smem_offset -= num_frags_z * 16 * num_vecs_per_head;
  }
}

template <SharedMemFillMode fill_mode,
          uint32_t num_warps,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          typename T>
__device__ __forceinline__ void produce_kv(smem_t smem,
                                           uint32_t* smem_offset,
                                           T** gptr,
                                           const uint32_t kv_n_stride,
                                           const uint32_t kv_idx_base,
                                           const uint32_t kv_len) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8;  // kv_idx used to check
#pragma unroll
  for (uint32_t i = 0; i < num_frags_z * 4 / num_warps;
       ++i) {  // m num_frags_z * 16 / (num_warps * 4)
#pragma unroll
    for (uint32_t j = 0; j < num_frags_y / 4;
         ++j) {  // k num_frags_y * 16 / 8 / num_elems_per_128b<T>()
      smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
      *gptr += 8 * num_elems_per_128b<T>();
    }
    kv_idx += num_warps * 4;
    *smem_offset = smem.advance_offset_by_row<num_warps * 4, num_vecs_per_head>(
                       *smem_offset) -
                   2 * num_frags_y;  // num_frags_y / 4 * 8
    *gptr +=
        num_warps * 4 * kv_n_stride - 2 * num_frags_y * num_elems_per_128b<T>();
  }
  *smem_offset -= num_frags_z * 16 * num_vecs_per_head;
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          typename T>
__device__ __forceinline__ void compute_qk(smem_t* q_smem,
                                           uint32_t* q_smem_offset_r,
                                           smem_t* k_smem,
                                           uint32_t* k_smem_offset_r,
                                           float (*s_frag)[num_frags_z][8]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  uint32_t a_frag[num_frags_x][4], b_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {  // k
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {  // m
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);
      *q_smem_offset_r = q_smem->advance_offset_by_row<16, num_vecs_per_head>(
          *q_smem_offset_r);
    }

    *q_smem_offset_r =
        q_smem->advance_offset_by_column<2>(*q_smem_offset_r, fy) -
        num_frags_x * 16 * num_vecs_per_head;

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {  // n
      k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      *k_smem_offset_r = k_smem->advance_offset_by_row<16, num_vecs_per_head>(
          *k_smem_offset_r);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        if (fy == 0) {
          mma_sync_m16n16k16_row_col_f16f16f32<T, MMAMode::kInit>(
              s_frag[fx][fz], a_frag[fx], b_frag);
        } else {
          mma_sync_m16n16k16_row_col_f16f16f32<T>(
              s_frag[fx][fz], a_frag[fx], b_frag);
        }
      }
    }
    *k_smem_offset_r =
        k_smem->advance_offset_by_column<2>(*k_smem_offset_r, fy) -
        num_frags_z * 16 * num_vecs_per_head;
  }
  *q_smem_offset_r -= num_frags_y * 2;
  *k_smem_offset_r -= num_frags_y * 2;
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          typename T,
          typename CacheT>
__device__ __forceinline__ void compute_qk_c4(smem_t* q_smem,
                                              uint32_t* q_smem_offset_r,
                                              smem_t* k_smem,
                                              uint32_t* k_smem_offset_r,
                                              float (*s_frag)[num_frags_z][8],
                                              T (*cache_k_scale_frag)[4],
                                              T (*cache_k_zp_frag)[4]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head_q = head_dim / num_elems_per_128b<T>();
  constexpr uint32_t num_vecs_per_head_k =
      head_dim / 2 / num_elems_per_128b<CacheT>();

  uint32_t a_frag[num_frags_x][4][4], b_frag[4], b_frag_dq[4];

#pragma unroll
  for (uint32_t ky = 0; ky < num_frags_y / 4; ++ky) {  // k
                                                       // load q
#pragma unroll
    for (uint32_t fy = 0; fy < 4; ++fy) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx][fy]);
        *q_smem_offset_r =
            q_smem->advance_offset_by_row<16, num_vecs_per_head_q>(
                *q_smem_offset_r);
      }
      *q_smem_offset_r =
          q_smem->advance_offset_by_column<2>(*q_smem_offset_r, ky * 4 + fy) -
          num_frags_x * 16 * num_vecs_per_head_q;
    }

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      // load
      k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      *k_smem_offset_r = k_smem->advance_offset_by_row<16, num_vecs_per_head_k>(
          *k_smem_offset_r);
#pragma unroll
      for (uint32_t fy = 0; fy < 4; ++fy) {
        T* b_frag_dq_T = reinterpret_cast<T*>(b_frag_dq);
        convert_int4(b_frag_dq_T, b_frag[fy]);
#pragma unroll
        for (uint32_t b_i = 0; b_i < 8; ++b_i) {
          const int b_offset = b_i % 4;
          b_frag_dq_T[b_i] =
              (b_frag_dq_T[b_i] - cache_k_zp_frag[ky * 4 + fy][b_offset]) *
              cache_k_scale_frag[ky * 4 + fy][b_offset];
        }

#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
          if (ky == 0 && fy == 0) {
            mma_sync_m16n16k16_row_col_f16f16f32<T, MMAMode::kInit>(
                s_frag[fx][fz], a_frag[fx][fy], b_frag_dq);
          } else {
            mma_sync_m16n16k16_row_col_f16f16f32<T>(
                s_frag[fx][fz], a_frag[fx][fy], b_frag_dq);
          }
        }
      }
    }
    *k_smem_offset_r = k_smem->advance_offset_by_column<2, num_vecs_per_head_k>(
                           *k_smem_offset_r, ky) -
                       num_frags_z * 16 * num_vecs_per_head_k;
  }
  // advance by col
  *q_smem_offset_r -= num_frags_y * 2;
  *k_smem_offset_r -= num_frags_y / 4 * 2;
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          typename T,
          typename CacheT>
__device__ __forceinline__ void compute_qk_c8(smem_t* q_smem,
                                              uint32_t* q_smem_offset_r,
                                              smem_t* k_smem,
                                              uint32_t* k_smem_offset_r,
                                              const T cache_k_scale,
                                              float (*s_frag)[num_frags_z][8]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head_q = head_dim / num_elems_per_128b<T>();
  constexpr uint32_t num_vecs_per_head_k =
      head_dim / num_elems_per_128b<CacheT>();

  uint32_t a_frag[num_frags_x][2][4], b_frag[4], b_frag_dq[4];

#pragma unroll
  for (uint32_t ky = 0; ky < num_frags_y / 2; ++ky) {  // k
                                                       // load q
#pragma unroll
    for (uint32_t fy = 0; fy < 2; ++fy) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx][fy]);

        *q_smem_offset_r =
            q_smem->advance_offset_by_row<16, num_vecs_per_head_q>(
                *q_smem_offset_r);
      }
      *q_smem_offset_r =
          q_smem->advance_offset_by_column<2>(*q_smem_offset_r, ky * 2 + fy) -
          num_frags_x * 16 * num_vecs_per_head_q;
    }

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      // load
      k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      *k_smem_offset_r = k_smem->advance_offset_by_row<16, num_vecs_per_head_k>(
          *k_smem_offset_r);
#pragma unroll
      for (uint32_t fy = 0; fy < 2; ++fy) {
        T* b_frag_dq_T = reinterpret_cast<T*>(b_frag_dq);
        convert_int8(b_frag_dq_T, b_frag[fy * 2]);
        convert_int8(b_frag_dq_T + 4, b_frag[fy * 2 + 1]);
        // scale zp
#pragma unroll
        for (uint32_t b_i = 0; b_i < 8; ++b_i) {

          b_frag_dq_T[b_i] *= cache_k_scale;
        }
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
          if (ky == 0 && fy == 0) {
            mma_sync_m16n16k16_row_col_f16f16f32<T, MMAMode::kInit>(
                s_frag[fx][fz], a_frag[fx][fy], b_frag_dq);
          } else {
            mma_sync_m16n16k16_row_col_f16f16f32<T>(
                s_frag[fx][fz], a_frag[fx][fy], b_frag_dq);
          }
        }
      }
    }
    *k_smem_offset_r = k_smem->advance_offset_by_column<2, num_vecs_per_head_k>(
                           *k_smem_offset_r, ky) -
                       num_frags_z * 16 * num_vecs_per_head_k;
  }
  *q_smem_offset_r -= num_frags_y * 2;
  *k_smem_offset_r -= num_frags_y / 2 * 2;
}

template <typename T,
          bool partition_kv,
          bool causal,
          uint32_t group_size,
          uint32_t num_warps,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          bool IS_SYSTEM = false>
__device__ __forceinline__ void mask_s(const uint32_t qo_idx_base,
                                       const uint32_t kv_idx_base,
                                       const uint32_t qo_len,
                                       const uint32_t kv_len,
                                       const uint32_t chunk_end,
                                       float (*s_frag)[num_frags_z][8]) {
  const uint32_t tx = threadIdx.x;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        if constexpr (!IS_SYSTEM) {
          const uint32_t q_idx = (qo_idx_base + fx * 16 + tx / 4 +
                                  8 * ((reg_id % 4) / 2)) /
                                 group_size,
                         kv_idx = kv_idx_base + fz * 16 + 2 * (tx % 4) +
                                  8 * (reg_id / 4) + reg_id % 2;
          const bool out_of_boundary =
              (causal
                   ? (kv_idx > kv_len + q_idx - qo_len || (kv_idx >= chunk_end))
                   : kv_idx >= chunk_end);
          if constexpr (std::is_same<T, half>::value) {
            s_frag[fx][fz][reg_id] =
                out_of_boundary ? -5e4f : s_frag[fx][fz][reg_id];
          } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            s_frag[fx][fz][reg_id] =
                out_of_boundary ? -3.0e+30f : s_frag[fx][fz][reg_id];
          }
        } else {
          const uint32_t q_idx = qo_idx_base,
                         kv_idx = kv_idx_base + fz * 16 + 2 * (tx % 4) +
                                  8 * (reg_id / 4) + reg_id % 2;
          const bool out_of_boundary =
              (causal
                   ? (kv_idx > kv_len + q_idx - qo_len || (kv_idx >= chunk_end))
                   : kv_idx >= chunk_end);
          if constexpr (std::is_same<T, half>::value) {
            s_frag[fx][fz][reg_id] =
                out_of_boundary ? -5e4f : s_frag[fx][fz][reg_id];
          } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            s_frag[fx][fz][reg_id] =
                out_of_boundary ? -3.0e+30f : s_frag[fx][fz][reg_id];
          }
        }
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z>
__device__ __forceinline__ void update_mdo_states(
    float (*s_frag)[num_frags_z][8],
    float (*o_frag)[num_frags_y][8],
    float (*m)[2],
    float (*d)[2]) {
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      float m_prev = m[fx][j];
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        float m_local =
            max(max(s_frag[fx][fz][j * 2 + 0], s_frag[fx][fz][j * 2 + 1]),
                max(s_frag[fx][fz][j * 2 + 4], s_frag[fx][fz][j * 2 + 5]));
        m[fx][j] = max(m[fx][j], m_local);
      }
      m[fx][j] = max(m[fx][j], __shfl_xor_sync(-1, m[fx][j], 0x2, 32));
      m[fx][j] = max(m[fx][j], __shfl_xor_sync(-1, m[fx][j], 0x1, 32));
      float o_scale = __expf(m_prev - m[fx][j]);
      d[fx][j] *= o_scale;
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        o_frag[fx][fy][j * 2 + 0] *= o_scale;
        o_frag[fx][fy][j * 2 + 1] *= o_scale;
        o_frag[fx][fy][j * 2 + 4] *= o_scale;
        o_frag[fx][fy][j * 2 + 5] *= o_scale;
      }
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        s_frag[fx][fz][j * 2 + 0] =
            __expf(s_frag[fx][fz][j * 2 + 0] - m[fx][j]);
        s_frag[fx][fz][j * 2 + 1] =
            __expf(s_frag[fx][fz][j * 2 + 1] - m[fx][j]);
        s_frag[fx][fz][j * 2 + 4] =
            __expf(s_frag[fx][fz][j * 2 + 4] - m[fx][j]);
        s_frag[fx][fz][j * 2 + 5] =
            __expf(s_frag[fx][fz][j * 2 + 5] - m[fx][j]);
      }
    }
  }
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t block_size,
          typename T,
          typename CacheT>
__device__ __forceinline__ void compute_sfm_v_c4(
    smem_t* v_smem,
    uint32_t* v_smem_offset_r,
    float (*s_frag)[num_frags_z][8],
    float (*o_frag)[num_frags_y][8],
    float (*d)[2],
    T (*cache_v_scale_frag)[2],
    T (*cache_v_zp_frag)[2]) {
  constexpr uint32_t num_vecs_per_blocksize =
      block_size / 2 / num_elems_per_128b<CacheT>();

  T s_frag_f16[num_frags_x][num_frags_z][8];
  uint32_t b_frag[4], b_frag_dq[4];
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      vec_cast<T, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t kz = 0; kz < num_frags_z / 4; ++kz) {  // k
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      v_smem->ldmatrix_m8n8x4(*v_smem_offset_r, b_frag);
      *v_smem_offset_r =
          v_smem->advance_offset_by_row<16, num_vecs_per_blocksize>(
              *v_smem_offset_r);
#pragma unroll
      for (uint32_t fz = 0; fz < 4; ++fz) {
        T* b_frag_dq_T = reinterpret_cast<T*>(b_frag_dq);
        convert_int4(b_frag_dq_T, b_frag[fz]);
        // scale zp
#pragma unroll
        for (uint32_t b_i = 0; b_i < 8; ++b_i) {
          const int b_offset = b_i / 4;
          b_frag_dq_T[b_i] =
              (b_frag_dq_T[b_i] - cache_v_zp_frag[fy][b_offset]) *
              cache_v_scale_frag[fy][b_offset];
        }
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {  // m: num_frags_x * 16
          mma_sync_m16n16k16_row_col_f16f16f32<T>(
              o_frag[fx][fy],
              (uint32_t*)(s_frag_f16[fx][kz * 4 + fz]),
              b_frag_dq);
        }
      }
    }
    *v_smem_offset_r =
        v_smem->advance_offset_by_column<2, num_vecs_per_blocksize>(
            *v_smem_offset_r, kz) -
        num_frags_y * 16 * num_vecs_per_blocksize;
  }
  *v_smem_offset_r -= num_frags_z / 4 * 2;
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t block_size,
          typename T,
          typename CacheT>
__device__ __forceinline__ void compute_sfm_v_c8(
    smem_t* v_smem,
    uint32_t* v_smem_offset_r,
    float (*s_frag)[num_frags_z][8],
    float (*o_frag)[num_frags_y][8],
    float (*d)[2],
    T cache_v_scale) {
  constexpr uint32_t num_vecs_per_blocksize =
      block_size / num_elems_per_128b<CacheT>();
  T s_frag_f16[num_frags_x][num_frags_z][8];
  uint32_t b_frag[4], b_frag_dq[4];
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      vec_cast<T, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t kz = 0; kz < num_frags_z / 2; ++kz) {  // k
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      v_smem->ldmatrix_m8n8x4(*v_smem_offset_r, b_frag);
      *v_smem_offset_r =
          v_smem->advance_offset_by_row<16, num_vecs_per_blocksize>(
              *v_smem_offset_r);
#pragma unroll
      for (uint32_t fz = 0; fz < 2; ++fz) {
        T* b_frag_dq_T = reinterpret_cast<T*>(b_frag_dq);
        convert_int8(b_frag_dq_T, b_frag[fz * 2]);
        convert_int8(b_frag_dq_T + 4, b_frag[fz * 2 + 1]);
        // scale zp
#pragma unroll
        for (uint32_t b_i = 0; b_i < 8; ++b_i) {
          b_frag_dq_T[b_i] *= cache_v_scale;
        }
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {  // m: num_frags_x * 16
          mma_sync_m16n16k16_row_col_f16f16f32<T>(
              o_frag[fx][fy],
              (uint32_t*)(s_frag_f16[fx][kz * 2 + fz]),
              b_frag_dq);

        }
      }
    }

    *v_smem_offset_r =
        v_smem->advance_offset_by_column<2, num_vecs_per_blocksize>(
            *v_smem_offset_r, kz) -
        num_frags_y * 16 * num_vecs_per_blocksize;
  }
  *v_smem_offset_r -= num_frags_z / 2 * 2;
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          uint32_t block_size,
          typename T,
          typename CacheT>
__device__ __forceinline__ void compute_sfm_v_c8_iter_sq_bvec(
    smem_t* v_smem,
    uint32_t* v_smem_offset_r,
    float (*s_frag)[num_frags_z][8],
    float (*o_frag)[num_frags_y][8],
    float (*d)[2],
    T cache_v_scale) {
  constexpr uint32_t num_vecs_per_blocksize =
      block_size / num_elems_per_128b<CacheT>();

  T s_frag_f16[num_frags_x][num_frags_z][8];
  uint32_t b_frag[4], b_frag_dq[4];
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      vec_cast<T, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t kz = 0; kz < num_frags_z / 2; ++kz) {  // k
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      v_smem->ldmatrix_m8n8x4(*v_smem_offset_r, b_frag);
      *v_smem_offset_r =
          v_smem->advance_offset_by_row<16, num_vecs_per_blocksize>(
              *v_smem_offset_r);
#pragma unroll
      for (uint32_t fz = 0; fz < 2; ++fz) {
        // dequant b_frag -> b_frag_dq
        T* b_frag_dq_T = reinterpret_cast<T*>(b_frag_dq);
        convert_int8(b_frag_dq_T, b_frag[fz * 2]);
        convert_int8(b_frag_dq_T + 4, b_frag[fz * 2 + 1]);
        // scale zp
#pragma unroll
        for (uint32_t b_i = 0; b_i < 8; ++b_i) {
          b_frag_dq_T[b_i] *= cache_v_scale;
        }
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {  // m: num_frags_x * 16
          mma_sync_m16n16k16_row_col_f16f16f32<T>(
              o_frag[fx][fy],
              (uint32_t*)(s_frag_f16[fx][kz * 2 + fz]),
              b_frag_dq);
        }
      }
    }
    *v_smem_offset_r -= num_frags_y * 16 * num_vecs_per_blocksize;
  }
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t num_frags_z,
          typename T>
__device__ __forceinline__ void compute_sfm_v(smem_t* v_smem,
                                              uint32_t* v_smem_offset_r,
                                              float (*s_frag)[num_frags_z][8],
                                              float (*o_frag)[num_frags_y][8],
                                              float (*d)[2]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();

  T s_frag_f16[num_frags_x][num_frags_z][8];
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      vec_cast<T, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
    }
  }

#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z;
       ++fz) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t b_frag[4];
      v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {  // m: num_frags_x * 16
        mma_sync_m16n16k16_row_col_f16f16f32<T>(
            o_frag[fx][fy], (uint32_t*)(s_frag_f16[fx][fz]), b_frag);
      }
      *v_smem_offset_r =
          v_smem->advance_offset_by_column<2>(*v_smem_offset_r, fy);
    }
    *v_smem_offset_r =
        v_smem->advance_offset_by_row<16, num_vecs_per_head>(*v_smem_offset_r) -
        2 * num_frags_y;
  }
  *v_smem_offset_r -= 16 * num_frags_z * num_vecs_per_head;
}

template <uint32_t num_frags_x, uint32_t num_frags_y>
__device__ __forceinline__ void normalize_d(float (*o_frag)[num_frags_y][8],
                                            float (*d)[2]) {
  float d_rcp[num_frags_x][2];
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d_rcp[fx][j] = 1.f / d[fx][j];
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] =
            o_frag[fx][fy][reg_id] * d_rcp[fx][(reg_id % 4) / 2];
      }
    }
  }
}

template <uint32_t num_frags_x,
          uint32_t num_frags_y,
          uint32_t NUM_WARPS,
          typename T>
__device__ __forceinline__ void merge_res_multi_warps(
    T* o_smem,   // [num_threads, num_frags_x, num_frags_y, 8]
    T* md_smem,  // [num_warps, num_frags_x * 16 * 2]
    T (*o_frag)[num_frags_y][8],
    T (*m_frag)[2],
    T (*d_frag)[2]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t tidx = ty * 32 + tx;
  const uint32_t row_id = tx / 4;

  // [num_frags_x * 16, num_frags_y * 16]
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      const int offset =
          tidx * num_frags_x * num_frags_y * 8 + fx * num_frags_y * 8 + fy * 8;
      *(b128_t*)(&o_smem[offset]) = *(b128_t*)&o_frag[fx][fy];
      *(b128_t*)(&o_smem[offset + 4]) = *(b128_t*)&o_frag[fx][fy][4];
    }
  }
  if (tx % 4 == 0) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      const int offset = ty * num_frags_x * 16 + fx * 16 + row_id;
      md_smem[offset * 2] = m_frag[fx][0];
      md_smem[offset * 2 + 1] = d_frag[fx][0];
      md_smem[(offset + 8) * 2] = m_frag[fx][1];
      md_smem[(offset + 8) * 2 + 1] = d_frag[fx][1];
    }
  }
  __syncthreads();

  if (ty == 0) {
#pragma unroll
    for (uint32_t warp_id = 0; warp_id < NUM_WARPS; ++warp_id) {
      const int tmp_tidx = warp_id * 32 + tx;
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        const int offset = warp_id * num_frags_x * 16 + fx * 16 + row_id;
#pragma unroll
        for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
          const int o_offset = tmp_tidx * num_frags_x * num_frags_y * 8 +
                               fx * num_frags_y * 8 + fy * 8;
          AlignedVector<float, 8> o_now;
          Load(&o_smem[o_offset], &o_now);

          float m_prev = m_frag[fx][0], d_prev = d_frag[fx][0];
          float m_now = md_smem[offset * 2], d_now = md_smem[offset * 2 + 1];
          float tmp_m = max(m_prev, m_now);
          float scale1 = __expf(m_prev - tmp_m), scale2 = __expf(m_now - tmp_m);
          float tmp_d = scale1 * d_prev + scale2 * d_now;
          o_frag[fx][fx][0] = scale1 * o_frag[fx][fx][0] + scale2 * o_now[0];
          o_frag[fx][fx][1] = scale1 * o_frag[fx][fx][1] + scale2 * o_now[1];
          o_frag[fx][fx][4] = scale1 * o_frag[fx][fx][4] + scale2 * o_now[4];
          o_frag[fx][fx][5] = scale1 * o_frag[fx][fx][5] + scale2 * o_now[5];
          m_frag[fx][0] = tmp_m;
          d_frag[fx][0] = tmp_d;

          m_prev = m_frag[fx][1], d_prev = d_frag[fx][1];
          m_now = md_smem[(offset + 8) * 2],
          d_now = md_smem[(offset + 8) * 2 + 1];
          tmp_m = max(m_prev, m_now);
          scale1 = __expf(m_prev - tmp_m), scale2 = __expf(m_now - tmp_m);
          tmp_d = scale1 * d_prev + scale2 * d_now;
          o_frag[fx][fx][2] = scale1 * o_frag[fx][fx][2] + scale2 * o_now[2];
          o_frag[fx][fx][3] = scale1 * o_frag[fx][fx][3] + scale2 * o_now[3];
          o_frag[fx][fx][6] = scale1 * o_frag[fx][fx][6] + scale2 * o_now[6];
          o_frag[fx][fx][7] = scale1 * o_frag[fx][fx][7] + scale2 * o_now[7];
          m_frag[fx][1] = tmp_m;
          d_frag[fx][1] = tmp_d;
        }
      }
    }
  }
}

template <uint32_t group_size,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          typename T>
__device__ __forceinline__ void write_o_reg_gmem_kv_multi_warps(
    float (*o_frag)[num_frags_y][8],
    smem_t* o_smem,
    T* o_ptr_base,
    uint32_t o_idx_base,
    const uint32_t qo_upper_bound,
    const uint32_t qo_n_stride,
    const uint32_t qo_h_stride) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

  if (ty == 0) {
    // [num_frags_x * 16, num_frags_y * 16]
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        uint32_t o_frag_f16[4];
        vec_cast<T, float, 8>((T*)o_frag_f16, o_frag[fx][fy]);
        uint32_t o_smem_offset_w = smem_t::get_permuted_offset<
            num_vecs_per_head>(
            fx * 16 + tx / 4,
            fy * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[tx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w +
                     8 * num_vecs_per_head))[tx % 4] = o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[tx % 4] =
            o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * num_vecs_per_head))[tx % 4] = o_frag_f16[3];
      }
    }
  }
  __syncthreads();

  uint32_t o_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      ty * 4 + tx / 8, tx % 8);

  o_idx_base += (tx / 8) / group_size;
  o_ptr_base += ((tx / 8) / group_size) * qo_n_stride +
                ((tx / 8) % group_size) * qo_h_stride;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    // for (uint32_t j = 0; j < 4; ++j) { // 4 * 4 = 16
    const int j = ty;
    const uint32_t o_idx = o_idx_base + (fx * 16 + j * 4) / group_size;
    T* o_ptr = o_ptr_base + ((fx * 16 + j * 4) / group_size) * qo_n_stride +
               ((fx * 16 + j * 4) % group_size) * qo_h_stride;
#pragma unroll
    for (uint32_t fyo = 0; fyo < num_frags_y / 4;
         ++fyo) {
      if (o_idx < qo_upper_bound) {
        // need write
        o_smem->store_128b(o_smem_offset_w, o_ptr);
      }
      o_ptr += 8 * num_elems_per_128b<T>();
      o_smem_offset_w =
          o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
    }
    o_smem_offset_w =
        o_smem->advance_offset_by_row<16, num_vecs_per_head>(o_smem_offset_w) -
        2 * num_frags_y;
  }
}


template <typename T, int VEC_SIZE, typename OutT>
struct StoreFunc {
  __device__ __forceinline__ void operator()(
      const AlignedVector<T, VEC_SIZE>& ori_out_vec,
      const AlignedVector<T, VEC_SIZE>& shift_bias_vec,
      const AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
      AlignedVector<OutT, VEC_SIZE>& out_vec,
      const float quant_max_bound,
      const float quant_min_bound,
      const float in_scale,
      const int i) {
    out_vec[i] = static_cast<OutT>(ori_out_vec[i]);
    printf("Fatal! Unimplemented StoreFunc for cascade append attention\n");
  }
};

template <typename T, int VEC_SIZE>
struct StoreFunc<T, VEC_SIZE, int8_t> {
  __device__ __forceinline__ void operator()(
      const AlignedVector<T, VEC_SIZE>& ori_out_vec,
      const AlignedVector<T, VEC_SIZE>& shift_bias_vec,
      const AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
      AlignedVector<int8_t, VEC_SIZE>& out_vec,
      const float quant_max_bound,
      const float quant_min_bound,
      const float in_scale,
      const int i) {
    float quant_value =
        quant_max_bound *
        static_cast<float>((ori_out_vec[i] + shift_bias_vec[i]) *
                           smooth_weight_vec[i]) *
        in_scale;
    quant_value = rintf(quant_value);
    quant_value = quant_value > quant_max_bound ? quant_max_bound : quant_value;
    quant_value = quant_value < quant_min_bound ? quant_min_bound : quant_value;
    out_vec[i] = static_cast<int8_t>(quant_value);
  }
};

template <typename T, int VEC_SIZE>
struct StoreFunc<T, VEC_SIZE, __nv_fp8_e4m3> {
  __device__ __forceinline__ void operator()(
      const AlignedVector<T, VEC_SIZE>& ori_out_vec,
      const AlignedVector<T, VEC_SIZE>& shift_bias_vec,
      const AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
      AlignedVector<__nv_fp8_e4m3, VEC_SIZE>& out_vec,
      const float quant_max_bound,
      const float quant_min_bound,
      const float in_scale,
      const int i) {
    float quant_value =
        quant_max_bound * static_cast<float>(ori_out_vec[i]) * in_scale;
    quant_value = quant_value > quant_max_bound ? quant_max_bound : quant_value;
    quant_value = quant_value < quant_min_bound ? quant_min_bound : quant_value;
    out_vec[i] = static_cast<__nv_fp8_e4m3>(quant_value);
  }
};

template <typename T, int VEC_SIZE>
struct StoreFunc<T, VEC_SIZE, T> {
  __device__ __forceinline__ void operator()(
      const AlignedVector<T, VEC_SIZE>& ori_out_vec,
      const AlignedVector<T, VEC_SIZE>& shift_bias_vec,
      const AlignedVector<T, VEC_SIZE>& smooth_weight_vec,
      AlignedVector<T, VEC_SIZE>& out_vec,
      const float quant_max_bound,
      const float quant_min_bound,
      const float in_scale,
      const int i) {
    out_vec[i] = ori_out_vec[i];
  }
};

template <uint32_t group_size,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          bool partition_kv,
          typename T,
          typename OutT>
__device__ __forceinline__ void write_o_reg_gmem_multi_warps_shift_smooth_quant(
    float (*o_frag)[num_frags_y][8],
    smem_t* o_smem,
    OutT* o_ptr_base,
    const T* shift_bias,
    const T* smooth_weight,
    uint32_t o_idx_base,
    const uint32_t q_head_idx_base,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const uint32_t qo_upper_bound,
    const uint32_t qo_n_stride,
    const uint32_t qo_h_stride) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr int VEC_SIZE = 16 / sizeof(T);
  AlignedVector<T, VEC_SIZE> ori_out_vec;
  AlignedVector<T, VEC_SIZE> shift_bias_vec;
  AlignedVector<T, VEC_SIZE> smooth_weight_vec;
  AlignedVector<OutT, VEC_SIZE> out_vec;
  // [num_warps * num_frags_x * 16, num_frags_y * 16]
  if (ty == 0) {
    // [num_frags_x * 16, num_frags_y * 16]
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        uint32_t o_frag_f16[4];
        vec_cast<T, float, 8>((T*)o_frag_f16, o_frag[fx][fy]);
        uint32_t o_smem_offset_w = smem_t::get_permuted_offset<
            num_vecs_per_head>(
            fx * 16 + tx / 4,
            fy * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[tx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w +
                     8 * num_vecs_per_head))[tx % 4] = o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[tx % 4] =
            o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * num_vecs_per_head))[tx % 4] = o_frag_f16[3];
      }
    }
  }
  __syncthreads();

  uint32_t o_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      ty * 4 + tx / 8, tx % 8); 

  const uint32_t tx_offset = tx / 8;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    const uint32_t base_offset = o_idx_base + fx * 16 + tx_offset;
#pragma unroll
    const int j = ty;
    const uint32_t offset_now = base_offset + j * 4;
    const uint32_t n_offset = offset_now / group_size;
    const uint32_t h_offset = offset_now % group_size;

    OutT* o_ptr = o_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;

    uint32_t shift_smooth_offset = (q_head_idx_base + h_offset) * head_dim +
                                   tx % 8 * num_elems_per_128b<T>();
#pragma unroll
    for (uint32_t fyo = 0; fyo < num_frags_y / 4;
         ++fyo) {
      if (n_offset < qo_upper_bound) {
        if constexpr (!partition_kv) {
          if (in_scale > 0.0) {
            if (shift_bias) {
              Load<T, VEC_SIZE>(shift_bias + shift_smooth_offset,
                                &shift_bias_vec);
              Load<T, VEC_SIZE>(smooth_weight + shift_smooth_offset,
                                &smooth_weight_vec);
            }
          }
          Load<T, VEC_SIZE>(
              reinterpret_cast<T*>(o_smem->base + o_smem_offset_w),
              &ori_out_vec);

#pragma unroll
          for (int i = 0; i < VEC_SIZE; ++i) {
            StoreFunc<T, VEC_SIZE, OutT>()(ori_out_vec,
                                           shift_bias_vec,
                                           smooth_weight_vec,
                                           out_vec,
                                           quant_max_bound,
                                           quant_min_bound,
                                           in_scale,
                                           i);
          }
          Store<OutT, VEC_SIZE>(out_vec, o_ptr);
        } else {
          o_smem->store_128b(o_smem_offset_w, o_ptr);
        }
      }
      o_ptr += 8 * num_elems_per_128b<T>();
      shift_smooth_offset += 8 * num_elems_per_128b<T>();
      o_smem_offset_w =
          o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
    }
    o_smem_offset_w =
        o_smem->advance_offset_by_row<16, num_vecs_per_head>(o_smem_offset_w) -
        2 * num_frags_y;
  }
}

template <uint32_t group_size,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          bool partition_kv,
          typename T,
          typename OutT>
__device__ __forceinline__ void write_o_reg_gmem_shift_smooth_quant(
    float (*o_frag)[num_frags_y][8],
    smem_t* o_smem,
    OutT* o_ptr_base,
    const T* shift_bias,
    const T* smooth_weight,
    uint32_t o_idx_base,
    const uint32_t q_head_idx_base,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const uint32_t qo_upper_bound,
    const uint32_t qo_n_stride,
    const uint32_t qo_h_stride) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr int VEC_SIZE = 8;
  AlignedVector<T, VEC_SIZE> ori_out_vec;
  AlignedVector<T, VEC_SIZE> shift_bias_vec;
  AlignedVector<T, VEC_SIZE> smooth_weight_vec;
  AlignedVector<OutT, VEC_SIZE> out_vec;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t o_frag_f16[4];
      vec_cast<T, float, 8>((T*)o_frag_f16, o_frag[fx][fy]);
      uint32_t o_smem_offset_w = smem_t::get_permuted_offset<
          num_vecs_per_head>(
          (ty * num_frags_x + fx) * 16 + tx / 4,
          fy * 2);
      ((uint32_t*)(o_smem->base + o_smem_offset_w))[tx % 4] = o_frag_f16[0];
      ((uint32_t*)(o_smem->base + o_smem_offset_w +
                   8 * num_vecs_per_head))[tx % 4] = o_frag_f16[1];
      ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[tx % 4] =
          o_frag_f16[2];
      ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                   8 * num_vecs_per_head))[tx % 4] = o_frag_f16[3];
    }
  }
  __syncthreads();

  uint32_t o_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      ty * num_frags_x * 16 + tx / 8,
      tx % 8);

  const uint32_t tx_offset = tx / 8;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    const uint32_t base_offset = o_idx_base + fx * 16 + tx_offset;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {  // 4 * 4 = 16
      const uint32_t offset_now = base_offset + j * 4;
      const uint32_t n_offset = offset_now / group_size;
      const uint32_t h_offset = offset_now % group_size;
      OutT* o_ptr =
          o_ptr_base + n_offset * qo_n_stride + h_offset * qo_h_stride;
      uint32_t shift_smooth_offset = (q_head_idx_base + h_offset) * head_dim +
                                     tx % 8 * num_elems_per_128b<T>();
#pragma unroll
      for (uint32_t fyo = 0; fyo < num_frags_y / 4;
           ++fyo) {
        if (n_offset < qo_upper_bound) {
          if (!partition_kv && in_scale > 0.0) {
            if (shift_bias) {
              Load<T, VEC_SIZE>(shift_bias + shift_smooth_offset,
                                &shift_bias_vec);
              Load<T, VEC_SIZE>(smooth_weight + shift_smooth_offset,
                                &smooth_weight_vec);
            }
            Load<T, VEC_SIZE>(
                reinterpret_cast<T*>(o_smem->base + o_smem_offset_w),
                &ori_out_vec);
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
              StoreFunc<T, VEC_SIZE, OutT>()(ori_out_vec,
                                             shift_bias_vec,
                                             smooth_weight_vec,
                                             out_vec,
                                             quant_max_bound,
                                             quant_min_bound,
                                             in_scale,
                                             i);
            }
            Store<OutT, VEC_SIZE>(out_vec, o_ptr);
          } else {
            o_smem->store_128b(o_smem_offset_w, o_ptr);
          }
        }
        o_ptr += 8 * num_elems_per_128b<T>();
        shift_smooth_offset += 8 * num_elems_per_128b<T>();
        o_smem_offset_w =
            o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
      }
      o_smem_offset_w =
          o_smem->advance_offset_by_row<4, num_vecs_per_head>(o_smem_offset_w) -
          2 * num_frags_y;
    }
  }
}

template <uint32_t group_size,
          uint32_t num_frags_x,
          uint32_t num_frags_y,
          typename T>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[num_frags_y][8],
    smem_t* o_smem,
    T* o_ptr_base,
    uint32_t o_idx_base,
    const uint32_t qo_upper_bound,
    const uint32_t qo_n_stride,
    const uint32_t qo_h_stride) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_vecs_per_head = head_dim / num_elems_per_128b<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t o_frag_f16[4];
      vec_cast<T, float, 8>((T*)o_frag_f16, o_frag[fx][fy]);
      uint32_t o_smem_offset_w = smem_t::get_permuted_offset<
          num_vecs_per_head>(
          (ty * num_frags_x + fx) * 16 + tx / 4,
          fy * 2);
      ((uint32_t*)(o_smem->base + o_smem_offset_w))[tx % 4] = o_frag_f16[0];
      ((uint32_t*)(o_smem->base + o_smem_offset_w +
                   8 * num_vecs_per_head))[tx % 4] = o_frag_f16[1];
      ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[tx % 4] =
          o_frag_f16[2];
      ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                   8 * num_vecs_per_head))[tx % 4] = o_frag_f16[3];
    }
  }
  __syncthreads();

  uint32_t o_smem_offset_w = smem_t::get_permuted_offset<num_vecs_per_head>(
      ty * num_frags_x * 16 + tx / 8,
      tx % 8);

  o_idx_base += (tx / 8) / group_size;
  o_ptr_base += ((tx / 8) / group_size) * qo_n_stride +
                ((tx / 8) % group_size) * qo_h_stride;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {  // 4 * 4 = 16
      const uint32_t o_idx = o_idx_base + (fx * 16 + j * 4) / group_size;
      T* o_ptr = o_ptr_base + ((fx * 16 + j * 4) / group_size) * qo_n_stride +
                 ((fx * 16 + j * 4) % group_size) * qo_h_stride;
#pragma unroll
      for (uint32_t fyo = 0; fyo < num_frags_y / 4;
           ++fyo) {  
        if (o_idx < qo_upper_bound) {
          o_smem->store_128b(o_smem_offset_w, o_ptr);
        }
        o_ptr += 8 * num_elems_per_128b<T>();
        o_smem_offset_w =
            o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
      }
      o_smem_offset_w =
          o_smem->advance_offset_by_row<4, num_vecs_per_head>(o_smem_offset_w) -
          2 * num_frags_y;
    }
  }
}

template <uint32_t GROUP_SIZE>
__global__ void split_q_block(const int* __restrict__ seq_lens_q,
                              int* __restrict__ batch_ids,
                              int* __restrict__ tile_ids_per_batch,
                              int* __restrict__ num_blocks_x,
                              const uint32_t bsz,
                              const uint32_t num_rows_per_block) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      const int seq_len = seq_lens_q[bid];
      const int loop_times = div_up(seq_len * GROUP_SIZE, num_rows_per_block);
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

template <typename T, int vec_size>
__global__ void merge_multi_chunks_kernel(
    const T* __restrict__ multi_out,    // [token_num, num_chunks, num_heads,
                                        // head_dim]
    const float* __restrict__ multi_m,  // [token_num, num_chunks, num_heads]
    const float* __restrict__ multi_d,  // [token_num, num_chunks, num_heads]
    const int* __restrict__ seq_lens_q,
    const int* __restrict__ seq_lens_kv,
    const int* __restrict__ padding_offsets,
    const T* __restrict__ shift_bias,     // [q_num_heads * HEAD_DIM]
    const T* __restrict__ smooth_weight,  // [q_num_heads * HEAD_DIM]
    T* __restrict__ out,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int max_seq_len,
    const int num_chunks,
    const int num_heads,
    const int chunk_size,
    const int head_dim) {
  const int vid = threadIdx.x, hid = threadIdx.y;
  const int qid = blockIdx.x;
  const uint32_t ori_token_id = qid + padding_offsets[qid];
  const uint32_t bid = ori_token_id / max_seq_len;
  if (seq_lens_q[bid] <= 0 || seq_lens_kv[bid] <= 0) {
    return;
  }
  const int seq_len_kv = seq_lens_kv[bid];
  const int num_chunks_this_seq = div_up(seq_len_kv, chunk_size);

  using LoadT = AlignedVector<T, vec_size>;
  LoadT load_vec;
  LoadT res_vec;
  if constexpr (std::is_same<T, half>::value) {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((half2*)(&res_vec) + i) = make_half2(0, 0);
    }
  } else {
#pragma unroll
    for (int i = 0; i < vec_size / 2; ++i) {
      *((nv_bfloat162*)(&res_vec) + i) = make_bfloat162(0, 0);
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
  for (int i = 0; i < num_chunks_this_seq; ++i) {
    uint32_t offset = (qid * num_chunks + i) * num_heads + hid;
    float m_prev = m;
    float d_prev = d;
    const float m_now = multi_m[offset];
    const float d_now = multi_d[offset];
    m = max(m_prev, m_now);
    offset = (qid * num_chunks * num_heads + i * num_heads + hid) * head_dim +
             vid * vec_size;
    Load<T, vec_size>(&multi_out[offset], &load_vec);
    const float scale1 = __expf(m_prev - m), scale2 = __expf(m_now - m);
    const T scale1_T = static_cast<T>(scale1),
            scale2_T = static_cast<T>(scale2);
    d = d * scale1 + d_now * scale2;
    for (int j = 0; j < vec_size; j++) {
      res_vec[j] = res_vec[j] * scale1_T + load_vec[j] * scale2_T;
    }
  }
#pragma unroll
  for (int j = 0; j < vec_size; j++) {
    res_vec[j] /= d;
  }
  Store<T, vec_size>(res_vec,
                     &out[(qid * num_heads + hid) * head_dim + vid * vec_size]);
}


template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void merge_block_res(float (*o_frag)[num_frags_y][8],
                                                float* md_smem,
                                                float (*m)[2],
                                                float (*d)[2],
                                                const uint32_t wid,
                                                const uint32_t tid) {
  float2* smem_md = reinterpret_cast<float2*>(md_smem);
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      smem_md[((wid * num_frags_x + fx) * 2 + j) * 32 + tid] =
          make_float2(m[fx][j], d[fx][j]);
    }
  }
  __syncthreads();
  float o_scale[4][num_frags_x][2];

  // deal md/scale
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      float m_new;
      float d_new = 1.f;
      if constexpr (std::is_same<T, half>::value) {
        m_new = -5e4f;
      } else {
        m_new = -3.0e+30f;
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        float m_prev = m_new, d_prev = d_new;
        m_new = max(m_new, md.x);
        d_new = d_prev * __expf(m_prev - m_new) + md.y * __expf(md.x - m_new);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        o_scale[i][fx][j] = __expf(md.x - m_new);
      }
      m[fx][j] = m_new;
      d[fx][j] = d_new;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      AlignedVector<float, 8> o_new;
#pragma
      for (uint32_t o_id = 0; o_id < 8; ++o_id) {
        o_new[o_id] = 0.f;
      }
      *(reinterpret_cast<float4*>(md_smem + (wid * 32 + tid) * 8)) =
          *(reinterpret_cast<float4*>(&o_frag[fx][fy][0]));
      *(reinterpret_cast<float4*>(md_smem + (wid * 32 + tid) * 8 + 4)) =
          *(reinterpret_cast<float4*>(&o_frag[fx][fy][4]));
      __syncthreads();
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        AlignedVector<float, 8> oi;
        Load<float, 8>(md_smem + (i * 32 + tid) * 8, &oi);
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_new[reg_id] += oi[reg_id] * o_scale[i][fx][(reg_id % 4) / 2];
        }
      }
      *(reinterpret_cast<float4*>(&o_frag[fx][fy][0])) =
          *(reinterpret_cast<float4*>(&o_new[0]));
      *(reinterpret_cast<float4*>(&o_frag[fx][fy][4])) =
          *(reinterpret_cast<float4*>(&o_new[4]));
      __syncthreads();
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename T>
__device__ __forceinline__ void merge_block_res_v2(
    float (*o_frag)[num_frags_y][8],
    float* md_smem,
    float (*m)[2],
    float (*d)[2],
    const uint32_t wid,
    const uint32_t tid) {
  float2* smem_md = reinterpret_cast<float2*>(
      md_smem + num_frags_x * num_frags_y * 1024);  // 4 * 32 * 8
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      smem_md[((wid * num_frags_x + fx) * 2 + j) * 32 + tid] =
          make_float2(m[fx][j], d[fx][j]);
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      *(reinterpret_cast<float4*>(
          md_smem + (((wid * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) *
                        8)) = *(reinterpret_cast<float4*>(&o_frag[fx][fy][0]));
      *(reinterpret_cast<float4*>(
          md_smem +
          (((wid * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) * 8 + 4)) =
          *(reinterpret_cast<float4*>(&o_frag[fx][fy][4]));
    }
  }
  __syncthreads();
  float o_scale[4][num_frags_x][2];

  // deal md/scale
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      float m_new;
      float d_new = 1.f;
      if constexpr (std::is_same<T, half>::value) {
        m_new = -5e4f;
      } else {
        m_new = -3.0e+30f;
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        float m_prev = m_new, d_prev = d_new;
        m_new = max(m_new, md.x);
        d_new = d_prev * __expf(m_prev - m_new) + md.y * __expf(md.x - m_new);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        float2 md = smem_md[((i * num_frags_x + fx) * 2 + j) * 32 + tid];
        o_scale[i][fx][j] = __expf(md.x - m_new);
      }
      m[fx][j] = m_new;
      d[fx][j] = d_new;
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      // num_warps * 32 * 8 each time
      AlignedVector<float, 8> o_new;
#pragma
      for (uint32_t o_id = 0; o_id < 4; ++o_id) {
        *(reinterpret_cast<float2*>(&o_new[o_id * 2])) = make_float2(0.f, 0.f);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        AlignedVector<float, 8> oi;
        Load<float, 8>(
            md_smem +
                (((i * num_frags_x + fx) * num_frags_y + fy) * 32 + tid) * 8,
            &oi);
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_new[reg_id] += oi[reg_id] * o_scale[i][fx][(reg_id % 4) / 2];
        }
      }
      *(reinterpret_cast<float4*>(&o_frag[fx][fy][0])) =
          *(reinterpret_cast<float4*>(&o_new[0]));
      *(reinterpret_cast<float4*>(&o_frag[fx][fy][4])) =
          *(reinterpret_cast<float4*>(&o_new[4]));
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
    const float quant_max_bound,
    const float quant_min_bound,
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
    uint32_t offset = (bid * num_chunks + i) * num_heads + hid;
    float m_prev = m;
    float d_prev = d;
    const float m_now = multi_m[offset];
    const float d_now = multi_d[offset];
    m = max(m_prev, m_now);
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
      StoreFunc<T, vec_size, OutT>()(
          st.o, shift_bias_vec, smooth_weight_vec, out_vec, quant_max_bound, quant_min_bound, in_scale, i);
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
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int max_seq_len,
    const int num_chunks,
    const int num_heads,
    const int chunk_size,
    const int head_dim,
    const int token_num,
    const int speculate_max_draft_token_num = 5) {
  const int vid = threadIdx.x, ty = threadIdx.y;
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
        StoreFunc<T, vec_size, OutT>()(
            st.o, shift_bias_vec, smooth_weight_vec, out_vec, quant_max_bound, quant_min_bound, in_scale, i);
      }
      Store<OutT, vec_size>(
          out_vec, &out[(qid * num_heads + hid) * head_dim + vid * vec_size]);
    }
    __syncthreads();
  }
}
