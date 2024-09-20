// Copyright © 2024 PaddlePaddle Name. All Rights Reserved.
//
// This code is partially inspired by and references the implementation found in FlashInfer.
// Specifically, the implementation of Top-p Sampling functionality in this code is inspired by the logic of FlashInfer’s flashinfer.sampling.top_p_sampling_from_probs function.
// For more details on FlashInfer’s documentation, please refer to: https://docs.flashinfer.ai/generated/flashinfer.sampling.top_p_sampling_from_probs.html#flashinfer-sampling-top-p-sampling-from_probs
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

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

/******************* utils *******************/
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#ifndef NDEBUG
#define CUDA_CALL(func, ...)                                            \
  {                                                                     \
    cudaError_t e = (func);                                             \
    if (e != cudaSuccess) {                                             \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e \
                << ") " << __FILE__ << ": line " << __LINE__            \
                << " at function " << STR(func) << std::endl;           \
      return e;                                                         \
    }                                                                   \
  }
#else
#define CUDA_CALL(func, ...) \
  {                          \
    cudaError_t e = (func);  \
    if (e != cudaSuccess) {  \
      return e;              \
    }                        \
  }
#endif

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
  if (deterministic) {                                            \
    constexpr bool DETERMINISTIC = true;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    constexpr bool DETERMINISTIC = false;                         \
    __VA_ARGS__                                                   \
  }

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      throw std::invalid_argument(err_msg.str());                          \
    }                                                                      \
  }

/******************* vec_t<float> *******************/

#define SAMPLING_INLINE inline __attribute__((always_inline)) __device__
template <typename float_t, size_t vec_size>
struct vec_t {
  SAMPLING_INLINE float_t& operator[](size_t i);
  SAMPLING_INLINE const float_t& operator[](size_t i) const;
  SAMPLING_INLINE void fill(float_t val);
  SAMPLING_INLINE void load(const float_t* ptr);
  SAMPLING_INLINE void store(float_t* ptr) const;
  template <typename T>
  SAMPLING_INLINE void cast_from(const vec_t<T, vec_size>& src);
  template <typename T>
  SAMPLING_INLINE void cast_load(const T* ptr);
  template <typename T>
  SAMPLING_INLINE void cast_store(T* ptr) const;
  SAMPLING_INLINE static void memcpy(float_t* dst, const float_t* src);
  SAMPLING_INLINE float_t* ptr();
};

// float x 1
template <>
struct vec_t<float, 1> {
  float data;

  SAMPLING_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  SAMPLING_INLINE const float& operator[](size_t i) const {
    return ((const float*)(&data))[i];
  }
  SAMPLING_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  SAMPLING_INLINE void fill(float val);
  SAMPLING_INLINE void load(const float* ptr);
  SAMPLING_INLINE void store(float* ptr) const;
  template <typename T>
  SAMPLING_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  SAMPLING_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  SAMPLING_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  SAMPLING_INLINE static void memcpy(float* dst, const float* src);
};

SAMPLING_INLINE void vec_t<float, 1>::fill(float val) { data = val; }

SAMPLING_INLINE void vec_t<float, 1>::load(const float* ptr) { data = *ptr; }

SAMPLING_INLINE void vec_t<float, 1>::store(float* ptr) const { *ptr = data; }

SAMPLING_INLINE void vec_t<float, 1>::memcpy(float* dst, const float* src) {
  *dst = *src;
}

// float x 2
template <>
struct vec_t<float, 2> {
  float2 data;

  SAMPLING_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  SAMPLING_INLINE const float& operator[](size_t i) const {
    return ((const float*)(&data))[i];
  }
  SAMPLING_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  SAMPLING_INLINE void fill(float val);
  SAMPLING_INLINE void load(const float* ptr);
  SAMPLING_INLINE void store(float* ptr) const;
  template <typename T>
  SAMPLING_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  SAMPLING_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  SAMPLING_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  SAMPLING_INLINE static void memcpy(float* dst, const float* src);
};

SAMPLING_INLINE void vec_t<float, 2>::fill(float val) {
  data = make_float2(val, val);
}

SAMPLING_INLINE void vec_t<float, 2>::load(const float* ptr) {
  data = *((float2*)ptr);
}

SAMPLING_INLINE void vec_t<float, 2>::store(float* ptr) const {
  *((float2*)ptr) = data;
}

SAMPLING_INLINE void vec_t<float, 2>::memcpy(float* dst, const float* src) {
  *((float2*)dst) = *((float2*)src);
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  float4 data[vec_size / 4];

  SAMPLING_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  SAMPLING_INLINE const float& operator[](size_t i) const {
    return ((const float*)(data))[i];
  }
  SAMPLING_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  SAMPLING_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  SAMPLING_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  SAMPLING_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  SAMPLING_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  SAMPLING_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  SAMPLING_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
  SAMPLING_INLINE static void memcpy(float* dst, const float* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)dst)[i] = ((float4*)src)[i];
    }
  }
};

inline std::pair<int, int> GetCudaComputeCapability() {
  int device_id = 0;
  cudaGetDevice(&device_id);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  return std::make_pair(major, minor);
}

/******************* math *******************/
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}
