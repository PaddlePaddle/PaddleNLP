// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hipcub/hipcub.hpp>
#include <hiprand.h>
#include <hiprand_kernel.h>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cuda_fp8.h>
#endif
#include <iostream>
#include <fstream>

#include "env.h"
#include "paddle/extension.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "nlohmann/json.hpp"


using json = nlohmann::json;

#define CUDA_CHECK(call)                           \
  do {                                             \
    const cudaError_t error_code = call;           \
    if (error_code != cudaSuccess) {               \
      std::printf("at %s:%d - %s.\n",              \
                  __FILE__,                        \
                  __LINE__,                        \
                  cudaGetErrorString(error_code)); \
      exit(1);                                     \
    }                                              \
  } while (0)

#ifdef PADDLE_WITH_HIP
template<size_t kBlockSize = 256, size_t kNumWaves = 16>
inline hipError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    hipError_t err = hipGetDevice(&dev);
    if (err != hipSuccess) { return err; }
  }
  int sm_count;
  {
    hipError_t err = hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, dev);
    if (err != hipSuccess) { return err; }
  }
  int tpm;
  {
    hipError_t err = hipDeviceGetAttribute(&tpm, hipDeviceAttributeMaxThreadsPerMultiProcessor, dev);
    if (err != hipSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                    sm_count * tpm / kBlockSize * kNumWaves));
  return hipSuccess;
}
#else
template<size_t kBlockSize = 256, size_t kNumWaves = 16>
inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                    sm_count * tpm / kBlockSize * kNumWaves));
  return cudaSuccess;
}

inline int GetGPUComputeCapability(int id) {
  int major, minor;
  auto major_error_code =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id);
  auto minor_error_code =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id);
  return major * 10 + minor;
}
#endif

template<typename T>
__device__ T max_func(const T a, const T b) {
  return a > b ? a : b;
}

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max_func(a, b);
  }
};

template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
public:
  typedef half DataType;
  typedef paddle::float16 data_t;
};

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
public:
#ifdef PADDLE_WITH_HIP
  typedef hip_bfloat16 DataType;
#else
  typedef __nv_bfloat16 DataType;
#endif
  typedef paddle::bfloat16 data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT8_E4M3FN> {
public:
  typedef __nv_fp8_e4m3 DataType;
  typedef paddle::float8_e4m3fn data_t;
};

template <>
class PDTraits<paddle::DataType::INT8> {
public:
  typedef int8_t DataType;
  typedef int8_t data_t;
};

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

#ifdef PADDLE_WITH_HIP
template <int Size>
HOSTDEVICE inline void Store(const AlignedVector<hip_bfloat16, Size>& vec, int8_t* addr) {
  printf("Error: Store hip_bfloat16 to int8_t is not supported!");
}
#else
template <int Size>
HOSTDEVICE inline void Store(const AlignedVector<__nv_bfloat16, Size>& vec, int8_t* addr) {
  printf("Error: Store __nv_bfloat16 to int8_t is not supported!");
}
#endif

template <int Size>
HOSTDEVICE inline void Store(const AlignedVector<half, Size>& vec, int8_t* addr) {
  printf("Error: Store half to int8_t is not supported!");
}

constexpr int VEC_16B = 16;

inline json ReadJsonFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    json j;
    file >> j;
    return j;
}

// place must be an existing place object and cannot use paddle::CPUPlace() or paddle::GPUPlace()
inline paddle::Tensor GetEmptyTensor(const common::DDim& dims, const paddle::DataType& dtype, const paddle::Place& place){
  auto* allocator = paddle::GetAllocator(place);
  phi::DenseTensor dense_tensor;
  dense_tensor.Resize(dims);
  dense_tensor.AllocateFrom(allocator, dtype, dense_tensor.numel() * phi::SizeOf(dtype));
  return paddle::Tensor(std::make_shared<phi::DenseTensor>(dense_tensor));
}

__device__ inline bool is_in_end(const int64_t id, const int64_t *end_ids, int length) {
    bool flag = false;
    for (int i = 0; i < length; i++) {
        if (id == end_ids[i]) {
            return true;
        }
    }
    return flag;
}

inline int GetSMVersion() {
  static int sm_version = phi::backends::gpu::GetGPUComputeCapability(
      phi::backends::gpu::GetCurrentDeviceId());
  return sm_version;
}

inline bool GetMlaUseTensorcore() {
  static const bool flags_mla_use_tensorcore = get_flags_mla_use_tensorcore();
  static const bool enable_mla_tensorcore = GetSMVersion() >= 90 ? true : false;
  const bool mla_use_tensorcore = flags_mla_use_tensorcore && enable_mla_tensorcore;
  return mla_use_tensorcore;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                        : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

__device__ __forceinline__ float warpReduceMax(float max_value) {
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 16));
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 8));
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 4));
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 2));
    max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, 1));
    return max_value;
}

__device__ __forceinline__ float blockReduceMax(float max_value) {
    static __shared__ float warpLevelMaxs[32];
    const int laneId = threadIdx.x & 0x1f;;
    const int warpId = threadIdx.x >> 5;

    max_value = warpReduceMax(max_value);

    if (laneId == 0) warpLevelMaxs[warpId] = max_value;
        __syncthreads();

    max_value = (threadIdx.x < blockDim.x / 32) ? warpLevelMaxs[laneId] : 0;
    if (warpId == 0) max_value = warpReduceMax(max_value);

    return max_value;
}
