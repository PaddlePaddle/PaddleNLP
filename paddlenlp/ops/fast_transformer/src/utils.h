/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <stdint.h>
#include <vector>

#ifdef WITH_FT5
#include "src/fastertransformer5/utils/Tensor.h"
#endif

#ifdef PADDLE_ON_INFERENCE
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvToolsExt.h>


const int64_t numel(const std::vector<int64_t>& tensor_shape);

#ifdef WITH_FT5
template<typename T, typename D>
inline T* get_ptr(paddle::Tensor& t) {
    return reinterpret_cast<T*>(t.data<D>());
}

template<typename T, typename D>
inline const T* get_ptr(const paddle::Tensor& t) {
    return reinterpret_cast<const T*>(t.data<D>());
}

std::vector<size_t> convert_shape(paddle::Tensor tensor);

template<typename T, typename D>
fastertransformer5::Tensor convert_tensor(paddle::Tensor tensor);

template<typename T, typename D>
fastertransformer5::Tensor convert_tensor(paddle::Tensor tensor, fastertransformer5::MemoryType memory_type);

size_t sizeBytes(paddle::Tensor tensor);

struct cudaDeviceProp GetCudaDeviceProp();

class CudaDeviceProp {
 private:
  CudaDeviceProp() { prop_ = GetCudaDeviceProp(); }

 public:
  CudaDeviceProp(CudaDeviceProp& other) = delete;

  void operator=(const CudaDeviceProp&) = delete;

  static CudaDeviceProp* GetInstance();

  ~CudaDeviceProp();

  struct cudaDeviceProp prop_;
  
};

std::mutex* GetCublasWrapperMutex();

class CublasWrapperMutex {
 private:
  CublasWrapperMutex() { mutex_ = GetCublasWrapperMutex(); }

 public:
  CublasWrapperMutex(CublasWrapperMutex& other) = delete;

  void operator=(const CublasWrapperMutex&) = delete;

  static CublasWrapperMutex* GetInstance();

  ~CublasWrapperMutex();

  std::mutex* mutex_;
};
#endif
