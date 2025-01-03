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
#endif
#include <iostream>
#include <fstream>

#include "paddle/extension.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/allocator.h"
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

inline uint32_t get_decoder_block_shape_q() {
    static const char* decoder_block_shape_q_env = std::getenv("FLAGS_dec_block_shape_q");
    static const uint32_t decoder_block_shape_q =
            decoder_block_shape_q_env == nullptr ? 16 : std::stoi(std::string(decoder_block_shape_q_env));
    return decoder_block_shape_q;
}

inline uint32_t get_encoder_block_shape_q() {
    static const char* encoder_block_shape_q_env = std::getenv("FLAGS_enc_block_shape_q");
    static const uint32_t encoder_block_shape_q =
            encoder_block_shape_q_env == nullptr ? 64 : std::stoi(std::string(encoder_block_shape_q_env));
    return encoder_block_shape_q;
}
