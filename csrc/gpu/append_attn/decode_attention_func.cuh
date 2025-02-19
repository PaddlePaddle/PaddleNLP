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

    // T scale1 = __expf(m_prev - m), scale2 = __expf(other_m - m);

    // kType scale1_2, scale2_2;
    // if constexpr (std::is_same<T, half>::value) {
    //   scale1_2 = make_half2(scale1, scale1);
    //   scale2_2 = make_half2(scale2, scale2);
    // } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    //   scale1_2 = make_bfloat162(scale1, scale1);
    //   scale2_2 = make_bfloat162(scale2, scale2);
    // }
    d = d_prev * scale1 + other_d * scale2;
// #pragma unroll
//     for (size_t i = 0; i < vec_size / 2; ++i) {
//       *(kType*)&o[2 * i] = __hadd2(__hmul2(*(kType*)&o[2 * i], scale1_2), __hmul2(*(kType*)&other_o[2 * i], scale2_2));
//     }
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] = o[i] * scale1 + other_o[i] * scale2;
    }
  }

  __device__ __forceinline__ void normalize() {
    // using kType = typename cascade_attn_nv_type2_traits<T>::type;
    // kType d2;
    // if constexpr (std::is_same<T, half>::value) {
    //   d2 = make_half2(d, d);
    // } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    //   d2 = make_bfloat162(d, d);
    // }
// #pragma unroll
//     for (size_t i = 0; i < vec_size / 2; ++i) {
//       *(kType*)&o[2 * i] /= d2;
//     }
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      o[i] /= d;
    }
  }

};

template <uint32_t VEC_SIZE, uint32_t HEAD_DIM, typename T>
__device__ __forceinline__ void vec_apply_llama_rope(
    const T* x, const AlignedVector<T, VEC_SIZE>& freq, int32_t offset, AlignedVector<T, VEC_SIZE>& res_vec) {
  const uint32_t vid = threadIdx.x;
  AlignedVector<T, VEC_SIZE> permuted_vec;

  Load<T, VEC_SIZE>(x + vid * VEC_SIZE, &res_vec);
  const uint32_t tmp_offset = vid * VEC_SIZE < HEAD_DIM / 2 ? vid * VEC_SIZE + HEAD_DIM / 2 : vid * VEC_SIZE - HEAD_DIM / 2;
  Load<T, VEC_SIZE>(x + tmp_offset, &permuted_vec);
  const T offset_T = static_cast<T>(offset);
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    const T embed = offset_T * freq[i];
    const T sin = hsin(embed);
    const T cos = hcos(embed);
    res_vec[i] = res_vec[i] * cos +
                 ((vid * VEC_SIZE < HEAD_DIM / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
  }
}

template <SharedMemFillMode fill_mode, CacheType cache_type, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_V, uint32_t VEC_SIZE, uint32_t HALF_VEC_SIZE, uint32_t BLOCK_SIZE, uint32_t bdxc, uint32_t CACHE_VEC_SIZE, typename CacheT>
__device__ __forceinline__ void produce_kv(CacheT *smem,
                                          CacheT *kv_base_gptr,
                                          const int * block_table_smem,
                                          const uint32_t seq_offset_gmem,
                                          const uint32_t seq_offset_smem,
                                          const uint32_t kv_head_idx,
                                          const uint32_t kv_num_heads,
                                          const uint32_t vec_id,
                                          const uint32_t chunk_start,
                                          const uint32_t chunk_end) {
  int block_id = __ldg(&block_table_smem[seq_offset_gmem / BLOCK_SIZE]);
  if (block_id < 0) {
    block_id = 0;
  }
  const uint32_t block_offset = seq_offset_gmem % BLOCK_SIZE;
  // 8/16 T/int8 each time
  const uint32_t k_offset = ((block_id * kv_num_heads + kv_head_idx) * BLOCK_SIZE + block_offset) * HEAD_DIM_QK + vec_id * CACHE_VEC_SIZE;
  const uint32_t smem_offset = seq_offset_smem * HEAD_DIM_QK + vec_id * CACHE_VEC_SIZE;
  pred_load<128, PrefetchMode::kPrefetch, fill_mode, CacheT>(
    smem + smem_offset,
    kv_base_gptr + k_offset,
    seq_offset_gmem < chunk_end
  );
  
}

// template <SharedMemFillMode fill_mode, CacheType cache_type, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_V, uint32_t VEC_SIZE, uint32_t HALF_VEC_SIZE, uint32_t BLOCK_SIZE, uint32_t bdxc, uint32_t CACHE_VEC_SIZE, typename CacheT>
// __device__ __forceinline__ void produce_v(CacheT *smem,
//                                            CacheT *kv_base_gptr,
//                                            const int * block_table_smem,
//                                            const uint32_t seq_offset_gmem,
//                                            const uint32_t seq_offset_smem,
//                                            const uint32_t kv_head_idx,
//                                            const uint32_t kv_num_heads,
//                                            const uint32_t vec_id,
//                                            const uint32_t chunk_start,
//                                            const uint32_t chunk_end) {
//   int block_id = __ldg(&block_table_smem[seq_offset_gmem / BLOCK_SIZE]);
//   if (block_id < 0) {
//     block_id = 0;
//   }
//   const uint32_t block_offset = seq_offset_gmem % BLOCK_SIZE;
//     // 8/16 T/int8 each time
//   const uint32_t v_offset = ((block_id * kv_num_heads + kv_head_idx) * BLOCK_SIZE + block_offset) * HEAD_DIM_QK + vec_id * CACHE_VEC_SIZE;
//   const uint32_t smem_offset = seq_offset_smem * HEAD_DIM_V + vec_id * CACHE_VEC_SIZE;
//   pred_load<128, PrefetchMode::kPrefetch, fill_mode, CacheT>(
//     smem + smem_offset,
//     kv_base_gptr + v_offset,
//     seq_offset_gmem < chunk_end
//   ); 
// }

template <uint32_t vec_size, uint32_t half_vec_size, uint32_t bdx, uint32_t bdy, uint32_t HEAD_DIM, uint32_t HALF_HEAD_DIM, uint32_t DEAL_EACH_TIME, PosEncMode pos_enc_mode, CacheType cache_type, typename T, typename CacheT>
__device__ __forceinline__ void compute_qk(const CacheT* base_smem,
                                           const AlignedVector<T, vec_size>& q_vec,
                                           const uint32_t kv_idx_base,
                                           const uint32_t stage_idx,
                                           const uint32_t iter_base, 
                                           const uint32_t iter_bound,
                                           const uint32_t vid,
                                           T *tmp_smem,
                                           T *s,
                                           softmax_state_t<vec_size, T>& st) {
  const uint32_t bidy = threadIdx.y;
  const CacheT* smem;
  AlignedVector<T, vec_size> k_vec;
  T m_prev = st.m;
  // smem = base_smem + (stage_idx * DEAL_EACH_TIME + zid * tile_size) * HEAD_DIM;
  smem = base_smem + stage_idx * DEAL_EACH_TIME * HEAD_DIM;
#pragma unroll
  for (uint32_t j = 0; j < DEAL_EACH_TIME; ++j) {
    Load<CacheT, vec_size>(smem + j * HEAD_DIM + vid * vec_size, &k_vec);
#ifdef DEBUG_DEC_PRE
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("q_vec:\n");
      for (uint32_t i = 0; i < vec_size; ++i) {
        printf("%f ", static_cast<float>(q_vec[i]));
      }
      printf("\n");
      printf("k_vec:\n");
      for (uint32_t i = 0; i < vec_size; ++i) {
        printf("%f ", static_cast<float>(k_vec[i]));
      }
      printf("\n");
    }
    __syncthreads();
#endif
    if constexpr (std::is_same<T, half>::value) {
      s[j] = __float2half(0.f);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      s[j] = __float2bfloat16(0.f);
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#ifdef DEBUG_DEC_PRE
    __syncthreads();
    if (threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("be t%d-s[2]:%f \n", threadIdx.x, static_cast<float>(s[j]));
    }
    __syncthreads();
#endif
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += __shfl_xor_sync(-1, s[j], offset, 32);
    }
    
    __syncthreads();
#ifdef DEBUG_DEC_PRE
    __syncthreads();
    if (threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("af t%d-s[2]:%f \n", threadIdx.x, static_cast<float>(s[j]));
    }
    __syncthreads();
#endif
    tmp_smem[bidy] = s[j];
#ifdef DEBUG_DEC_PRE
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("tmp_smem:\n");
      for (uint32_t i = 0; i < bdy; ++i) {
        printf("%f ", static_cast<float>(tmp_smem[i]));
      }
      printf("\n");
    }
    __syncthreads();
#endif
    __syncthreads();
    if constexpr (std::is_same<T, half>::value) {
      s[j] = __float2half(0.f);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      s[j] = __float2bfloat16(0.f);
    }
    for(uint32_t i = 0; i < bdy; ++i) {
      s[j] += tmp_smem[i];
    }
#ifdef DEBUG_DEC_PRE
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("s[%d]: %f\n", j, static_cast<float>(s[j]));
    }
    __syncthreads();
#endif
    if constexpr (std::is_same<T, half>::value) {
      s[j] = (iter_base + j < iter_bound) ? s[j] : __float2half(-5e4f);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      s[j] = (iter_base + j < iter_bound) ? s[j] : __float2bfloat16(-3.38953e38f);
    }
    st.m = st.m > s[j] ? st.m : s[j];
  }

  T o_scale = hexp(m_prev - st.m);
  // T o_scale = __expf(m_prev - st.m);
  st.d *= o_scale;
  
#pragma unroll
  for (uint32_t j = 0; j < DEAL_EACH_TIME; ++j) {
    s[j] = hexp(s[j] - st.m);
    // s[j] = __expf(s[j] - st.m);
    st.d += s[j];
#ifdef DEBUG_DEC_ATTN
    int tile_id = iter_base + j;
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && (iter_base + j * bdz + zid < iter_bound)) {
      printf("update s and d, zip: %d, gid: %d, vid: %d, tile_id: %d, j: %d, s[%d]: %f, m: %f, d: %f\n",
              (int)zid, (int)threadIdx.y, (int)vid, (int)tile_id, (int)j, (int)j, static_cast<float>(s[j]), static_cast<float>(st.m), static_cast<float>(st.d));
    }
    __syncthreads();
#endif
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }
}

template<uint32_t vec_size, uint32_t half_vec_size, uint32_t DEAL_EACH_TIME, uint32_t HEAD_DIM_QK, uint32_t HALF_HEAD_DIM, CacheType cache_type, typename T, typename CacheT>
__device__ __forceinline__ void compute_sv(const T *s,
                                           const CacheT *base_v_smem,
                                           const uint32_t stage_idx,
                                           const uint32_t vid,
                                           softmax_state_t<vec_size, T>& st) {
  uint32_t zid = threadIdx.z;
  const CacheT* v_smem;
  AlignedVector<T, vec_size> v_vec;
  // v_smem = base_v_smem + (stage_idx * DEAL_EACH_TIME + zid * tile_size) * HEAD_DIM;
  v_smem = base_v_smem + (stage_idx * DEAL_EACH_TIME + zid) * HEAD_DIM_QK;
#pragma unroll
  for (int j = 0; j < DEAL_EACH_TIME; ++j) {
    // Load<T, vec_size>(v_smem + j * HEAD_DIM + vid * vec_size, &v_vec);
    Load<T, vec_size>(v_smem + j * HEAD_DIM_QK + vid * vec_size, &v_vec);
#ifdef DEBUG_DEC_PRE
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      printf("v_vec:\n");
      for (uint32_t i = 0; i < vec_size; ++i) {
        printf("%f ", static_cast<float>(v_vec[i]));
      }
      printf("\n");
    }
    __syncthreads();
#endif
#pragma unroll
    for (int reg_id = 0; reg_id < vec_size; ++reg_id) {
      st.o[reg_id] += s[j] * v_vec[reg_id];
#ifdef DEBUG_DEC_ATTN
      if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && j < 31) {
        printf("zip: %d, gid: %d, vid: %d, j: %d, s_vec[%d]: %f, v_vec[%d]: %f, o[%d]: %f, d: %f\n",
                (int)zid, (int)threadIdx.y, (int)vid, (int)j, (int)j, static_cast<float>(s[j]), (int)reg_id, static_cast<float>(v_vec[reg_id]), (int)reg_id, static_cast<float>(st.o[reg_id]), static_cast<float>(st.d));
      }
      __syncthreads();
#endif
    }
  }
#ifdef DEBUG_DEC_PRE
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("st.o:\n");
    for (uint32_t i = 0; i < vec_size; ++i) {
      printf("%f ", static_cast<float>(st.o[i]));
    }
    printf("\n");
    printf("st.d:\n");
    for (uint32_t i = 0; i < vec_size; ++i) {
      printf("%f ", static_cast<float>(st.d));
    }
    printf("\n");
  }
  __syncthreads();
#endif
}

// template<uint32_t vec_size, uint32_t HEAD_DIM, uint32_t bdy, uint32_t bdz, typename T>
// __device__ __forceinline__ void merge_res_per_block(softmax_state_t<vec_size, T>& st,
//                                                     T *smem,
//                                                     T *md_smem) {
//   if constexpr (bdz > 1) {
//     uint32_t vid = threadIdx.x, gid = threadIdx.y, zid = threadIdx.z;
//     if (vid * vec_size < HEAD_DIM) return;
//     Store<T, vec_size>(st.o, smem + vid * vec_size); // [bdz, bdy, head_dim]
//     md_smem[0] = st.m; // [bdz, bdy]
//     md_smem[1] = st.d;
//     __syncthreads();
//     st.init();
//     AlignedVector<T, vec_size> o_vec;
// #pragma unroll
//     for (uint32_t j = 0; j < bdz; ++j) {
//       T mz = md_smem[0], dz = md_smem[(j * bdy + gid) * 2 + 1];
//       Load<T, vec_size>(smem + (j * bdy + gid) * HEAD_DIM + vid * vec_size, &o_vec);
//       st.merge(o_vec, mz, dz);
//     }
//   }
// }
