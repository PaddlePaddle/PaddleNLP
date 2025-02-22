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


#include "multi_head_latent_attention_kernel.h"

template <size_t vec_size, typename T>
struct softmax_state_t {
  AlignedVector<T, vec_size> o;
  T m;
  T d;
  
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
      m = __float2half(-5e4f);
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
      m = __float2bfloat16(-3.38953e38f);
    }
  }

  __device__ __forceinline__ softmax_state_t() {
    init();
  }

  __device__ __forceinline__ void merge(const AlignedVector<T, vec_size>& other_o, 
                                        T other_m,
                                        T other_d) {
    // using kType = typename cascade_attn_nv_type2_traits<T>::type;
    T m_prev = m, d_prev = d;
    m = m_prev > other_m ? m_prev : other_m;
    T scale1 = hexp(m_prev - m), scale2 = hexp(other_m - m);

    d = d_prev * scale1 + other_d * scale2;

#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * scale1 + other_o[i] * scale2;
    }
  }

  __device__ __forceinline__ void normalize() {

#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] /= d;
    }
  }

};

template <size_t vec_size, typename T, uint32_t num_tiles = 0>
struct softmax_state_ts {
  uint32_t num_tiles_ = num_tiles;
  AlignedVector<T, vec_size> o[num_tiles];
  float m;
  float d;
  
  __device__ __forceinline__ void init() {
#pragma unroll
    for (uint32_t tile_id = 0; tile_id < num_tiles_; ++tile_id) {
      if constexpr (std::is_same<T, half>::value) {
#pragma unroll
        for (int i = 0; i < vec_size / 2; ++i) {
          *((half2*)(&o[tile_id]) + i) = make_half2(0, 0);
        }
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
#pragma unroll
        for (int i = 0; i < vec_size / 2; ++i) {
          *((nv_bfloat162*)(&o[tile_id]) + i) = make_bfloat162(0, 0);
        }
      }
    }
    d = 1.f;
    if constexpr (std::is_same<T, half>::value) {
      m = -5e4f;
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
      m = -3.38953e38f;
    }
  }

  __device__ __forceinline__ softmax_state_ts() {
    init();
  }

  __device__ __forceinline__ void normalize(const uint32_t tile_id) {

#pragma unroll
    for (size_t i = 0; i < vec_size; i++) {
      o[tile_id][i] /= d;
    }
  }

};

template <SharedMemFillMode fill_mode, uint32_t HEAD_DIM_QK, uint32_t vec_size, uint32_t NUM_VEC_PER_HEAD, uint32_t bdx, uint32_t BLOCK_SIZE, uint32_t CACHE_VEC_SIZE, typename CacheT>
__device__ __forceinline__ void produce_kv(CacheT *smem,
                                          CacheT *kv_base_gptr,
                                          const int * block_table_smem,
                                          const uint32_t seq_offset_gmem,
                                          const uint32_t seq_offset_smem,
                                          const uint32_t kv_head_idx,
                                          const uint32_t kv_num_heads,
                                          const uint32_t tidx,
                                          const uint32_t chunk_start,
                                          const uint32_t chunk_end) {
  int block_id = __ldg(&block_table_smem[seq_offset_gmem / BLOCK_SIZE]);
  if (block_id < 0) {
    block_id = 0;
  }
  const uint32_t block_offset = seq_offset_gmem % BLOCK_SIZE;
  // 8/16 T/int8 each time
  const uint32_t k_offset_base = ((block_id * kv_num_heads + kv_head_idx) * BLOCK_SIZE + block_offset) * HEAD_DIM_QK;
  const uint32_t smem_offset_base = seq_offset_smem * HEAD_DIM_QK;
  for(uint32_t vid = tidx; vid < NUM_VEC_PER_HEAD; vid += bdx) {
    pred_load<128, PrefetchMode::kPrefetch, fill_mode, CacheT>(
      smem + smem_offset_base + vid * CACHE_VEC_SIZE,
      kv_base_gptr + k_offset_base + vid * CACHE_VEC_SIZE,
      seq_offset_gmem < chunk_end
    );
  }
}

template <uint32_t vec_size, uint32_t NUM_VEC_PER_HEAD, uint32_t bdx, uint32_t bdy, uint32_t HEAD_DIM, uint32_t DEAL_EACH_TIME, uint32_t num_tile_v, typename T, typename CacheT>
__device__ __forceinline__ void compute_qk(const T* cu_q_smem,
                                           const CacheT* k_smem,
                                           const uint32_t kv_idx_base,
                                           const uint32_t stage_idx,
                                           const uint32_t iter_base, 
                                           const uint32_t iter_bound,
                                           const uint32_t tidx,
                                           const uint32_t gid,
                                           const float scale,
                                           float *s,
                                           softmax_state_ts<vec_size, T, num_tile_v>& st) {
  const CacheT* smem;
  AlignedVector<T, vec_size> q_vec;
  AlignedVector<T, vec_size> k_vec;
  float m_prev = st.m;
  // smem = base_smem + (stage_idx * DEAL_EACH_TIME + zid * tile_size) * HEAD_DIM;
  smem = k_smem + stage_idx * DEAL_EACH_TIME * HEAD_DIM;
#pragma unroll
  for (uint32_t j = 0; j < DEAL_EACH_TIME; ++j) {
    if (iter_base + j < iter_bound) {
      if constexpr (std::is_same<T, half>::value) {
        s[j] = 0.f;
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        s[j] = 0.f;
      }
#pragma unroll
      for(uint32_t vid = tidx; vid < NUM_VEC_PER_HEAD; vid += bdx) {
        Load<T, vec_size>(cu_q_smem + vid * vec_size, &q_vec);
        Load<CacheT, vec_size>(smem + j * HEAD_DIM + vid * vec_size, &k_vec);
        for (uint32_t i = 0; i < vec_size; ++i) {
          s[j] += static_cast<float>(q_vec[i] * k_vec[i]);
        }
      }
#pragma unroll
      for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
        s[j] += __shfl_xor_sync(-1, s[j], offset, 32);
      }    
      __syncthreads();
    } else {
      if constexpr (std::is_same<T, half>::value) {
        s[j] = -5e4f;
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        s[j] = -3.38953e38f;
      }
    }
    st.m = st.m > s[j] ? st.m : s[j];
  }

  // T o_scale = hexp(m_prev - st.m);
  float o_scale = __expf(m_prev - st.m);
  st.d *= o_scale;
  
#pragma unroll
  for (uint32_t j = 0; j < DEAL_EACH_TIME; ++j) {
    // s[j] = hexp(s[j] - st.m);
    s[j] = __expf(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t tile_id = 0; tile_id < num_tile_v; ++tile_id) {
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[tile_id][i] *= o_scale;
    }
  }
}

template<uint32_t vec_size, uint32_t NUM_VEC_PER_HEAD, uint32_t bdx, uint32_t DEAL_EACH_TIME, uint32_t HEAD_DIM_QK, uint32_t num_tile, typename T, typename CacheT>
__device__ __forceinline__ void compute_sv(const float *s,
                                           const CacheT *base_v_smem,
                                           const uint32_t stage_idx,
                                           const uint32_t iter_base, 
                                           const uint32_t iter_bound,
                                           const uint32_t tidx,
                                           softmax_state_ts<vec_size, T, num_tile>& st) {
  const CacheT* v_smem;
  AlignedVector<T, vec_size> v_vec;
#pragma unroll
  for (int j = 0; (j < DEAL_EACH_TIME) && (iter_base + j < iter_bound); ++j) {
    v_smem = base_v_smem + stage_idx * DEAL_EACH_TIME * HEAD_DIM_QK + j * HEAD_DIM_QK;
    for(uint32_t vid = tidx; vid < NUM_VEC_PER_HEAD; vid += bdx) {
      Load<T, vec_size>(v_smem + vid * vec_size, &v_vec);
      uint32_t tile_id = vid / bdx;
#pragma unroll
      for (int reg_id = 0; reg_id < vec_size; ++reg_id) {
        st.o[tile_id][reg_id] += static_cast<T>(s[j]) * v_vec[reg_id];
      }
    }
  }
}

