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

#include "cutlass/half.h"
#include "cutlass/bfloat16.h"
#include "paddle/extension.h"

template <paddle::DataType D>
class CutlassDtypeTraits;

template <>
class CutlassDtypeTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class CutlassDtypeTraits<paddle::DataType::FLOAT16> {
public:
  typedef cutlass::half_t DataType;
  typedef paddle::float16 data_t;
};

template <>
class CutlassDtypeTraits<paddle::DataType::BFLOAT16> {
public:
  typedef cutlass::bfloat16_t DataType;
  typedef paddle::bfloat16 data_t;
};