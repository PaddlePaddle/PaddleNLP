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

#include "utils.h"


const int64_t numel(const std::vector<int64_t>& tensor_shape) {
    int size = tensor_shape.size();
    int64_t n = 1;
    for (int i = 0; i < size; ++i) {
        n *= tensor_shape[i];
    }
    return n;
}


#ifdef WITH_FT5

namespace ft = fastertransformer5;

std::vector<size_t> convert_shape(paddle::Tensor tensor) {
    std::vector<size_t> v_shape;
    for (int i = 0; i < tensor.shape().size(); i++) {
        v_shape.push_back(tensor.shape()[i]);
    }
    return v_shape;
}

template <typename T, typename D>
ft::Tensor convert_tensor(paddle::Tensor tensor) {
    ft::MemoryType mtype = tensor.is_gpu() ? ft::MEMORY_GPU : ft::MEMORY_CPU;
    return convert_tensor<T, D>(tensor, mtype);
}

template ft::Tensor convert_tensor<int8_t, int8_t>(paddle::Tensor tensor);
template ft::Tensor convert_tensor<float, float>(paddle::Tensor tensor);
template ft::Tensor convert_tensor<half, paddle::float16>(paddle::Tensor tensor);
#ifdef ENABLE_BF16
template ft::Tensor convert_tensor<__nv_bfloat16, paddle::bfloat16>(paddle::Tensor tensor);
#endif
template ft::Tensor convert_tensor<int32_t, int32_t>(paddle::Tensor tensor);
template ft::Tensor convert_tensor<unsigned int64_t, unsigned int64_t>(paddle::Tensor tensor);
template ft::Tensor convert_tensor<unsigned int32_t, unsigned int32_t>(paddle::Tensor tensor);
template ft::Tensor convert_tensor<bool, bool>(paddle::Tensor tensor);

template <typename T, typename D>
ft::Tensor convert_tensor(paddle::Tensor tensor, ft::MemoryType memory_type) {
    return ft::Tensor{memory_type, ft::getTensorType<T>(), convert_shape(tensor), get_ptr<T, D>(tensor)};
}

template ft::Tensor convert_tensor<int8_t, int8_t>(paddle::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<float, float>(paddle::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<half, paddle::float16>(paddle::Tensor tensor, ft::MemoryType memory_type);
#ifdef ENABLE_BF16
template ft::Tensor convert_tensor<__nv_bfloat16, paddle::bfloat16>(paddle::Tensor tensor, ft::MemoryType memory_type);
#endif
template ft::Tensor convert_tensor<int32_t, int32_t>(paddle::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<unsigned int64_t, unsigned int64_t>(paddle::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<unsigned int32_t, unsigned int32_t>(paddle::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<bool, bool>(paddle::Tensor tensor, ft::MemoryType memory_type);

size_t sizeBytes(paddle::Tensor tensor) {
    return tensor.numel() * paddle::experimental::SizeOf(tensor.dtype());
}

struct cudaDeviceProp GetCudaDeviceProp() {
  int device{-1};
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  return props;
}

CudaDeviceProp* CudaDeviceProp::GetInstance() {
  static CudaDeviceProp* cuda_device_prop = nullptr;
  if (cuda_device_prop == nullptr) {
    cuda_device_prop = new CudaDeviceProp();
  }
  return cuda_device_prop;
}

std::mutex* GetCublasWrapperMutex() {
    return new std::mutex();
}

CublasWrapperMutex* CublasWrapperMutex::GetInstance() {
  static CublasWrapperMutex* cublas_wrapper_mutex = nullptr;
  if (cublas_wrapper_mutex == nullptr) {
    cublas_wrapper_mutex = new CublasWrapperMutex();
  }
  return cublas_wrapper_mutex;
}

CublasWrapperMutex::~CublasWrapperMutex() {
  delete mutex_;
}
#endif
