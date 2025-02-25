#pragma once

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvToolsExt.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "paddle/extension.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/allocator.h"
#include "helper.h"

namespace kernels = tensorrt_llm::kernels;
namespace cutlass_extensions = tensorrt_llm::cutlass_extensions;
using paddle::Tensor;

#define PADDLE_CHECK(condition) \
  do { \
    if (!(condition)) { \
      std::cerr << "PADDLE_CHECK failed: " << #condition << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
      assert(false);  /* This will terminate the program if the condition fails */ \
    } \
  } while (0)

template<typename T>
inline T* get_ptr(paddle::Tensor& t)
{
    return reinterpret_cast<T*>(t.data<T>());
}

template<typename T>
inline T* get_ptr(paddle::optional<paddle::Tensor>& t)
{   
    return reinterpret_cast<T*>(t.get_ptr()->data<T>());
}

template <typename T>
class DataTypeMapper;

template <>
class DataTypeMapper<float> {
public:
  typedef paddle::DataType DataType;
  typedef float data_t;
};

template <>
class DataTypeMapper<half> {
public:
  typedef paddle::DataType DataType;
  typedef paddle::float16 data_t;
};

template <>
class DataTypeMapper<__nv_bfloat16> {
public:
  typedef paddle::DataType DataType;
  typedef paddle::bfloat16 data_t;
};

template <>
class DataTypeMapper<__nv_fp8_e4m3> {
public:
  typedef paddle::DataType DataType;
  typedef paddle::float8_e4m3fn data_t;
};


template <>
class DataTypeMapper<uint8_t> {
public:
  typedef paddle::DataType DataType;
  typedef int8_t data_t;
};

template <>
class DataTypeMapper<cutlass::uint4b_t> {
public:
  typedef paddle::DataType DataType;
  typedef int8_t data_t;
};


inline int next_positive_power_of_2(int x) {
    if (x < 1) {
        return 1;
    }
    return 1 << (int)(log2(x - 1) + 1);
}

inline std::vector<int64_t> get_power_of_2_num_tokens_buckets(int max_num_tokens) {
    max_num_tokens = next_positive_power_of_2(max_num_tokens);
    std::vector<int64_t> num_token_buckets;
    int m = 1;
    while (m <= max_num_tokens) {
        num_token_buckets.push_back(m);
        m *= 2;
    }
    return num_token_buckets;
}

template <typename T>
void print_gpu_data(T* gpu_data, size_t num_elements, size_t num) {
    float* host_data = new float[num_elements];
    T* temp_data = new T[num_elements];
    cudaError_t err = cudaMemcpy(temp_data, gpu_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
    for (size_t i = 0; i < num_elements; i++) {
        host_data[i] = static_cast<float>(temp_data[i]);
    }
    for (size_t i = 0; i < num; i++) {
        printf("gpu_data from 0 [%zu] = %f\n", i, host_data[i]);
    }

    // 释放内存
    delete[] host_data;
    delete[] temp_data;
}


// tensorrt_llm::ActivationType getTRTActivationType(std::string activation_type_str)
// {
//     if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
//         return tensorrt_llm::ActivationType::Gelu;
//     }
//     else if (activation_type_str == "Relu" || activation_type_str == "relu") {
//         return tensorrt_llm::ActivationType::Relu;
//     }
//     else if (activation_type_str == "Silu" || activation_type_str == "silu") {
//         return tensorrt_llm::ActivationType::Silu;
//     }
//     else if (activation_type_str == "GeGLU" || activation_type_str == "geglu" || activation_type_str == "gated-gelu") {
//         return tensorrt_llm::ActivationType::Geglu;
//     }
//     else if (activation_type_str == "Swiglu") {
//         return tensorrt_llm::ActivationType::Swiglu;
//     }
//     else {
//         std::cout << "Activation Type: " <<  activation_type_str << " not supported !";
//     }
//     return tensorrt_llm::ActivationType::InvalidType;
// }
