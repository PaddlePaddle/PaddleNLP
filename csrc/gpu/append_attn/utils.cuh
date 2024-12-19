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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "mem_util.cuh"
    
struct AppendAttnMetaData {
  int batch_size;
  int block_size;
  int q_num_heads;
  int kv_num_heads;
  int token_nums;
  int head_dims;
  int max_blocks_per_seq;
};

__forceinline__ __host__ __device__ int div_up(int a, int b) {
  return (a + b - 1) / b;
}

enum PosEncMode { kNonePos, kRoPE, kAliBi };

enum CacheType { CacheT, CacheInt8Hw, CacheInt4CwZp };

template <typename T>
struct cascade_attn_type_traits {
  using type = T;
};

template <>
struct cascade_attn_type_traits<phi::dtype::bfloat16> {
  using type = __nv_bfloat16;
};

template <>
struct cascade_attn_type_traits<phi::dtype::float16> {
  using type = half;
};

template <>
struct cascade_attn_type_traits<phi::dtype::float8_e4m3fn> {
  using type = __nv_fp8_e4m3;
};

template <typename T>
struct cascade_attn_nv_type2_traits {
  using type = T;
};

template <>
struct cascade_attn_nv_type2_traits<__nv_bfloat16> {
  using type = __nv_bfloat162;
};

template <>
struct cascade_attn_nv_type2_traits<half> {
  using type = half2;
};

template <CacheType cache_type>
struct vec_traits {
  using type = b128_t;
};

template <>
struct vec_traits<CacheType::CacheInt8Hw> {
  using type = b64_t;
};

template <>
struct vec_traits<CacheType::CacheInt4CwZp> {
  using type = b32_t;
};

template <typename T, CacheType cache_type>
struct cache_type_traits {
  using type = T;
};

template <typename T>
struct cache_type_traits<T, CacheType::CacheInt8Hw> {
  using type = uint8_t;
};

template <typename T>
struct cache_type_traits<T, CacheType::CacheInt4CwZp> {
  using type = uint8_t;
};

__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x,
                                                           uint32_t y) {
  return (x > y) ? x - y : 0U;
}

/******************************FASTER CAST*********************************/
inline __device__ static void convert_int8(
    __nv_bfloat16* result, const uint32_t& source) {  // 4 int8 each time
  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t fp32_base = 0x4B000000;
  float fp32_intermediates[4];

  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);
  fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);
  fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);
  fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

#pragma unroll
  for (int ii = 0; ii < 4; ++ii) {
    fp32_intermediates[ii] -= 8388736.f;  // (8388608.f + 128.f);
  }

#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    bf16_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0],
                                      fp32_intermediates_casted[2 * ii + 1],
                                      0x7632);
  }
}

inline __device__ static void convert_int8(
    half* result, const uint32_t& source) {  // 4 int8 each time
  uint32_t* fp16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(fp16_result_ptr[0])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(fp16_result_ptr[1])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(fp16_result_ptr[0])
               : "r"(fp16_result_ptr[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(fp16_result_ptr[1])
               : "r"(fp16_result_ptr[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
}

inline __device__ static void convert_int4(
    __nv_bfloat16* result, const uint32_t& source) {  // 8 int4 each time
  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(result);

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t MASK = 0x0f0f0f0f;  // 0xf -> 0b1111 select 0,4
  static constexpr uint32_t I4s_TO_FP32s_MAGIC_NUM = 0x43434343;
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;

  uint32_t tmp1 = source & MASK;       // 0 1 2 3
  uint32_t tmp2 = source >> 4 & MASK;  // 4 5 6 7

  bf16_result_ptr[0] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01);  // 0 1
  bf16_result_ptr[1] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23);  // 2 3
  bf16_result_ptr[2] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01);  // 4 5
  bf16_result_ptr[3] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23);  // 6 7
}

inline __device__ static void convert_int4(
    half* result, const uint32_t& source) {  // 7 5 3 1 6 4 2 0
  uint32_t* fp16_result_ptr = reinterpret_cast<uint32_t*>(result);

  static constexpr uint32_t MASK =
      0x0f0f0f0f;  // 0xf -> 0b1111 select 0,1;   7 5 3 1 6 4 2 0
  static constexpr uint32_t I4s_TO_FP32s_MAGIC_NUM = 0x64646464;
  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;

  uint32_t tmp1 = source & MASK;       // 0 1 2 3
  uint32_t tmp2 = source >> 4 & MASK;  // 4 5 6 7
  fp16_result_ptr[0] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01);  // 0 1
  fp16_result_ptr[1] = __byte_perm(tmp1,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23);  // 2 3
  fp16_result_ptr[2] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_01);  // 4 5
  fp16_result_ptr[3] = __byte_perm(tmp2,
                                   I4s_TO_FP32s_MAGIC_NUM,
                                   mask_for_elt_23);  // 6 7
}

/******************* vec_t type cast *******************/

template <typename dst_t, typename src_t, size_t vec_size>
__forceinline__ __host__ __device__ void vec_cast(dst_t* dst,
                                                  const src_t* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    dst[i] = src[i];
  }
}

template <size_t vec_size>
__forceinline__ __host__ __device__ void vec_cast<float, half>(
    float* dst, const half* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((float2*)dst)[i] = __half22float2(((half2*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ __host__ __device__ void vec_cast<half, float>(
    half* dst, const float* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((half2*)dst)[i] = __float22half2_rn(((float2*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ __host__ __device__ void vec_cast<float, nv_bfloat16>(
    float* dst, const nv_bfloat16* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((float2*)dst)[i] = __bfloat1622float2(((nv_bfloat162*)src)[i]);
  }
}

template <size_t vec_size>
__forceinline__ __host__ __device__ void vec_cast<nv_bfloat16, float>(
    nv_bfloat16* dst, const float* src) {
#pragma unroll
  for (size_t i = 0; i < vec_size / 2; ++i) {
    ((nv_bfloat162*)dst)[i] = __float22bfloat162_rn(((float2*)src)[i]);
  }
}

#define CHECK_CUDA_CALL(func, ...)                                      \
  {                                                                     \
    cudaError_t e = (func);                                             \
    if (e != cudaSuccess) {                                             \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e \
                << ") " << __FILE__ << ": line " << __LINE__            \
                << " at function " << STR(func) << std::endl;           \
      return e;                                                         \
    }                                                                   \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...) \
  switch (head_dim) {                              \
    case 128: {                                    \
      constexpr size_t HEAD_DIM = 128;             \
      __VA_ARGS__                                  \
      break;                                       \
    }                                              \
    default: {                                     \
      PD_THROW("not support the head_dim: ", head_dim);        \
    }                                              \
  }

#define DISPATCH_NUM_STAGE(num_stage, NUM_STAGE, ...) \
  if (num_stage == 2) {                               \
    constexpr size_t NUM_STAGE = 2;                   \
    __VA_ARGS__                                       \
  }

#define DISPATCH_CACHE_TYPE(cache_type, cache_type_now, cache_bytes, ...) \
  if (cache_type == 0) {                                                  \
    constexpr CacheType cache_type_now = CacheType::CacheT;               \
    constexpr size_t cache_bytes = 16;                                    \
    __VA_ARGS__                                                           \
  } else if (cache_type == 1) {                                           \
    constexpr CacheType cache_type_now = CacheType::CacheInt8Hw;          \
    constexpr size_t cache_bytes = 8;                                     \
    __VA_ARGS__                                                           \
  } else if (cache_type == 2) {                                           \
    constexpr CacheType cache_type_now = CacheType::CacheInt4CwZp;        \
    constexpr size_t cache_bytes = 4;                                     \
    __VA_ARGS__                                                           \
  }

#define DISPATCH_DEAL_EACH_TIME(deal_each_time, DEAL_EACH_TIME, ...) \
  if (deal_each_time == 32) {                                        \
    constexpr size_t DEAL_EACH_TIME = 32;                            \
    __VA_ARGS__                                                      \
  } else if (deal_each_time == 64) {                                 \
    constexpr size_t DEAL_EACH_TIME = 64;                            \
    __VA_ARGS__                                                      \
  }

#define DISPATCH_NUM_THREADS(num_threads, NUM_THREADS, ...) \
  if (num_threads == 128) {                                 \
    constexpr size_t NUM_THREADS = 128;                     \
    __VA_ARGS__                                             \
  } else if (num_threads == 256) {                          \
    constexpr size_t NUM_THREADS = 256;                     \
    __VA_ARGS__                                             \
  }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 2) {                              \
    constexpr size_t GROUP_SIZE = 2;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 3) {                              \
    constexpr size_t GROUP_SIZE = 3;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 5) {                              \
    constexpr size_t GROUP_SIZE = 5;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 6) {                              \
    constexpr size_t GROUP_SIZE = 6;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 7) {                              \
    constexpr size_t GROUP_SIZE = 7;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 12) {                             \
    constexpr size_t GROUP_SIZE = 12;                        \
    __VA_ARGS__                                              \
  }

#define DISPATCH_BLOCKSHAPE_Q(block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, ...) \
  if (block_shape_q <= 16) {                                                 \
    constexpr size_t BLOCK_SHAPE_Q = 16;                                     \
    constexpr size_t NUM_WARP_Q = 1;                                         \
    __VA_ARGS__                                                              \
  } else if (block_shape_q <= 32) {                                          \
    constexpr size_t BLOCK_SHAPE_Q = 32;                                     \
    constexpr size_t NUM_WARP_Q = 1;                                         \
    __VA_ARGS__                                                              \
  } else if (block_shape_q <= 64) {                                          \
    constexpr size_t BLOCK_SHAPE_Q = 64;                                     \
    constexpr size_t NUM_WARP_Q = 4;                                         \
    __VA_ARGS__                                                              \
  } else {                                                                   \
    constexpr size_t BLOCK_SHAPE_Q = 128;                                    \
    constexpr size_t NUM_WARP_Q = 4;                                         \
    __VA_ARGS__                                                              \
  }

#define DISPATCH_CAUSAL(causal, CAUSAL, ...) \
  if (causal) {                              \
    constexpr bool CAUSAL = true;            \
    __VA_ARGS__                              \
  }

#define DISPATCH_ENABLE_PREFILL(enable_prefill, ENABLE_PREFILL, ...) \
  if (enable_prefill) {                                              \
    constexpr bool ENABLE_PREFILL = 1;                               \
    __VA_ARGS__                                                      \
  } else {                                                           \
    constexpr bool ENABLE_PREFILL = 0;                               \
    __VA_ARGS__                                                      \
  }

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...) \
  if (block_size == 64) {                                \
    constexpr size_t BLOCK_SIZE = 64;                    \
    __VA_ARGS__                                          \
  }

#define DISPATCH_BLOCKSHAPE_Q_SYSTEM(              \
    block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, ...) \
  if (block_shape_q <= 16) {                       \
    constexpr size_t BLOCK_SHAPE_Q = 16;           \
    constexpr size_t NUM_WARP_Q = 1;               \
    __VA_ARGS__                                    \
  } else if (block_shape_q <= 32) {                \
    constexpr size_t BLOCK_SHAPE_Q = 32;           \
    constexpr size_t NUM_WARP_Q = 1;               \
    __VA_ARGS__                                    \
  }

template <typename T>
inline HOSTDEVICE T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}