/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvToolsExt.h>
// #include <torch/custom_class.h>
// #include <torch/script.h>
#include <vector>
#include "paddle/extension.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/allocator.h"


#include "helper.h"

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
