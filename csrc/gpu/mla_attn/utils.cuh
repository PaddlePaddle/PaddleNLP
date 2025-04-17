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

#pragma once

#ifndef ATTENTION_HOPPER_UTILS_CUH_
#define ATTENTION_HOPPER_UTILS_CUH_

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include "cutlass/fast_math.h"

namespace mla_attn {

using namespace cute;

template <typename TensorT>
CUTLASS_HOST_DEVICE auto flatten_1(TensorT tensor) {
  Tensor tensor_flatten = cute::flatten(tensor);
  return cute::group_modes<1, rank(tensor_flatten)>(tensor_flatten);
}

CUTLASS_HOST_DEVICE auto get_gmem_layout(int nnz, int num_heads, int head_dim, int64_t n_stride,
                                         int64_t h_stride) {
  return make_layout(make_shape(nnz, head_dim, num_heads),
                     make_stride(n_stride, cute::_1{}, h_stride));
}

CUTLASS_HOST_DEVICE auto get_lse_gmem_layout(int nnz, int num_heads) {
  return make_layout(make_shape(num_heads, nnz), make_stride(cute::_1{}, int64_t(num_heads)));
}

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_local_tile_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                          int head_idx, int offset, int seq_len) {
  auto g_offset = local_tile(m_tensor(_, _, head_idx), cute::make_shape(1, get<1>(tile_shape)),
                             make_coord(offset, _0{}));
  auto g_sequence =
      make_tensor(g_offset.data(),
                  make_layout(cute::make_shape(seq_len, get<1>(tile_shape)), g_offset.stride()));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_lse_local_tile_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                              int head_idx, int offset, int seq_len) {
  auto g_offset = local_tile(m_tensor(head_idx, _), cute::make_shape(_1{}), make_coord(offset));

  auto g_sequence = make_tensor(g_offset.data(), make_layout(cute::make_shape(seq_len),
                                                             cute::make_shape(shape<0>(m_tensor))));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_));
  return g_tensor;
}

// For SM90, convert acc_layout from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V,
// MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = acc_layout;
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
};

// For SM90, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16,
// MMA_N))
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
  static_assert(decltype(rank(acc_layout))::value == 3);
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
  auto l = logical_divide(get<0>(acc_layout), Shape<X, X, _2>{});  // (2, 2, (2, N / 16)))
  return make_layout(make_layout(get<0>(l), get<1>(l), get<2, 0>(l)), get<1>(acc_layout),
                     make_layout(get<2, 1>(l), get<2>(acc_layout)));
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <bool init = false, int wg_wait = 0, typename TensorA, typename TensorB, typename TensorC,
          typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma& tiled_mma, TensorA const& tCrA, TensorB const& tCrB,
                                     TensorC& tCrC) {
  constexpr bool Is_RS =
      !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
  // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<TensorA&>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  if constexpr (init) {
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  } else {
    // cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
  }
  warpgroup_commit_batch();
  if constexpr (wg_wait >= 0) {
    warpgroup_wait<wg_wait>();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<TensorA&>(tCrA));
  }
}

#define HOSTDEVICE __host__ __device__

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];

  HOSTDEVICE inline const T& operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T& operator[](int i) { return val[i]; }
};

template <typename T, int Size>
HOSTDEVICE inline void Load(const T* addr, AlignedVector<T, Size>* vec) {
  const AlignedVector<T, Size>* addr_vec =
      reinterpret_cast<const AlignedVector<T, Size>*>(addr);
  *vec = *addr_vec;
}

template <typename T, int Size>
HOSTDEVICE inline void Store(const AlignedVector<T, Size>& vec, T* addr) {
  AlignedVector<T, Size>* addr_vec =
      reinterpret_cast<AlignedVector<T, Size>*>(addr);
  *addr_vec = vec;
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

  __device__ __forceinline__ void merge(const AlignedVector<T, vec_size>& other_o, 
                                        const float other_m,
                                        const float other_d) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, other_m);
    const float scale1 = __expf(m_prev - m), scale2 = __expf(other_m - m);
    const T scale1_T = static_cast<T>(scale1), scale2_T = static_cast<T>(scale2);
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

template <typename T, int vec_size, uint32_t bdy, uint32_t HEAD_DIM>
__global__ void merge_multi_chunks_kernel(const T * __restrict__ multi_out, // [num_chunks, bsz, max_draft_token, num_heads, head_dim]
                                          const float * __restrict__ multi_m, // [num_chunks, bsz, max_draft_token, num_heads]
                                          const float * __restrict__ multi_d, // [num_chunks, bsz, max_draft_token, num_heads]
                                          const int * __restrict__ seq_lens_this_time,
                                          const int * __restrict__ seq_lens_decoder,
                                          const int * __restrict__ seq_lens_encoder,
                                          const int * __restrict__ padding_offsets,
                                          T * __restrict__ out, // [token_num, num_heads, head_dim]
                                          const int max_seq_len,
                                          const int num_chunks,
                                          const int num_heads,
                                          const int chunk_size,
                                          const int head_dim,
                                          const int token_num,
                                          const int bsz,
                                          const int draft_total_token_num) {
  const int vid = threadIdx.x, ty = threadIdx.y;
  const int hid = blockIdx.y;
  __shared__ T smem[bdy * HEAD_DIM];
  __shared__ float md_smem[bdy * 2];
  for (int qid = blockIdx.x; qid < token_num; qid += gridDim.x) {
    const uint32_t ori_token_id = qid + padding_offsets[qid];
    const uint32_t bid = ori_token_id / max_seq_len;
    const int seq_len_q = seq_lens_this_time[bid];
    if (seq_len_q == 0) continue;
    const uint32_t local_seq_id = ori_token_id % max_seq_len;
    int seq_len_kv = seq_lens_decoder[bid];
    if (seq_len_kv == 0) continue;
    seq_len_kv += seq_len_q;
    const int num_chunks_this_seq = cute::ceil_div(seq_len_kv, chunk_size);
    if (num_chunks_this_seq <= 1) {
      // not need merge
      continue;
    }

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

    for (int i = ty; i < num_chunks_this_seq; i += bdy) {
      uint32_t offset;
      offset = ((i * bsz + bid) * draft_total_token_num + local_seq_id) * num_heads + hid;
      float m_prev = m;
      float d_prev = d;
      const float m_now = multi_m[offset];
      const float d_now = multi_d[offset];
      m = max(m_prev, m_now);
      offset = (((i * bsz + bid) * draft_total_token_num + local_seq_id) * num_heads + hid) * head_dim + vid * vec_size;
      Load<T, vec_size>(&multi_out[offset], &load_vec);
      const float scale1 = __expf(m_prev - m), scale2 = __expf(m_now - m);
      const T scale1_T = static_cast<T>(scale1), scale2_T = static_cast<T>(scale2);
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
      Store<T, vec_size>(st.o, &out[(qid * num_heads + hid) * head_dim + vid * vec_size]);
    }
    __syncthreads();
  }
}

}  // namespace mla_attn

#endif  // ATTENTION_HOPPER_UTILS_CUH_
