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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied fron NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#pragma once

#include <cstdint>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <unordered_map>

namespace layer_norm {

template <typename Params>
struct LaunchParams {
  size_t workspace_bytes;
  size_t barrier_size;

  cudaDeviceProp *props;

  cudaStream_t stream;

  Params params;
};

struct ParamsBase {
  ParamsBase()
      : ctas_per_col(0),
        rows(0),
        cols(0),
        x(nullptr),
        mean(nullptr),
        invvar(nullptr),
        scale(nullptr),
        workspace(nullptr),
        barrier(nullptr) {}

  // For Multi-CTA, number of different CTA groups. Otherwise same as gridDim.x.
  int ctas_per_col;

  // Input is interpreted as matrix. We normalize across columns.
  int rows;
  int cols;

  // Common data pointers.
  void *x;
  void *mean;
  void *invvar;
  void *scale;

  // Multi-CTA workspace in gmem.
  void *workspace;

  // Multi-CTA sync barriers in gmem.
  int *barrier;
};

struct FwdParams : public ParamsBase {
  FwdParams() : ParamsBase(), y(nullptr), bias(nullptr), epsilon(0.f) {}

  // Output of LN FWD.
  void *y;
  void *bias;
  float epsilon;
};

struct BwdParams : public ParamsBase {
  BwdParams()
      : ParamsBase(),
        dy(nullptr),
        dbias_part(nullptr),
        dscale_part(nullptr),
        dx(nullptr),
        dbias(nullptr),
        dscale(nullptr) {}

  // Input: gradient wrt. LN FWD output.
  void *dy;

  // Workspace for Wgrad pre-reduction.
  void *dbias_part;
  void *dscale_part;

  // Output: Dgrad.
  void *dx;
  // Output: Wgrad.
  void *dbias;
  void *dscale;
};

using FwdFunction = std::function<void(LaunchParams<FwdParams> &, const bool)>;
using BwdFunction = std::function<void(LaunchParams<BwdParams> &, const bool)>;
using FunctionKey = uint64_t;
using FwdRegistry = std::unordered_map<FunctionKey, FwdFunction>;
using BwdRegistry = std::unordered_map<FunctionKey, BwdFunction>;

extern FwdRegistry FWD_FUNCS;
extern BwdRegistry BWD_FUNCS;

using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;

template <typename T>
struct TypeToIdTrait {};

template <>
struct TypeToIdTrait<fp16> {
  constexpr static uint32_t Value = 0;
};

template <>
struct TypeToIdTrait<bf16> {
  constexpr static uint32_t Value = 1;
};

template <>
struct TypeToIdTrait<fp32> {
  constexpr static uint32_t Value = 2;
};

template <typename T, int Significant>
struct Type2KeyTrait {
  constexpr static uint32_t Value = TypeToIdTrait<T>::Value << Significant;
};

template <typename T>
struct WeightType2KeyTrait : public Type2KeyTrait<T, 0> {};

template <typename T>
struct InputType2KeyTrait : public Type2KeyTrait<T, 2> {};

template <typename T>
struct OutputType2KeyTrait : public Type2KeyTrait<T, 4> {};

template <typename T>
struct ComputeType2KeyTrait : public Type2KeyTrait<T, 6> {};

template <typename WeightT,
          typename InputT,
          typename OutputT,
          typename ComputeT>
struct Types2KeyTrait {
  constexpr static uint32_t Value = WeightType2KeyTrait<WeightT>::Value |
                                    InputType2KeyTrait<InputT>::Value |
                                    OutputType2KeyTrait<OutputT>::Value |
                                    ComputeType2KeyTrait<ComputeT>::Value;
  constexpr static inline uint64_t get(const uint64_t hidden_size) {
    constexpr uint64_t type_key = Value;
    return (type_key << 32) | hidden_size;
  }
};

template <typename WeightT,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          uint64_t HIDDEN_SIZE>
struct FwdRegistrar {
  FwdRegistrar(FwdFunction f) {  // NOLINT
    uint64_t key =
        Types2KeyTrait<WeightT, InputT, OutputT, ComputeT>::get(HIDDEN_SIZE);
    FWD_FUNCS.insert({key, f});
  }
};

template <typename WeightT,
          typename InputT,
          typename OutputT,
          typename ComputeT,
          uint64_t HIDDEN_SIZE>
struct BwdRegistrar {
  BwdRegistrar(BwdFunction f) {  // NOLINT
    uint64_t key =
        Types2KeyTrait<WeightT, InputT, OutputT, ComputeT>::get(HIDDEN_SIZE);
    BWD_FUNCS.insert({key, f});
  }
};

}  // namespace layer_norm
