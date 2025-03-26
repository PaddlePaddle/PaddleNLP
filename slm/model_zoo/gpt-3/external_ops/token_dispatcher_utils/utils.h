#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>

#include "paddle/extension.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

template <paddle::DataType DType> struct TypeMap;

template <> struct TypeMap<paddle::DataType::BFLOAT16> { using type = phi::bfloat16; };
template <> struct TypeMap<paddle::DataType::FLOAT16>  { using type = phi::float16; };
template <> struct TypeMap<paddle::DataType::FLOAT32>  { using type = float; };
template <> struct TypeMap<paddle::DataType::INT32>    { using type = int; };
template <> struct TypeMap<paddle::DataType::INT64>    { using type = int64_t; };
