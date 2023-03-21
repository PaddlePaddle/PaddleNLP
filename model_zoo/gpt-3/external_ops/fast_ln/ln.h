/*  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
    Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
    This code is copied fron NVIDIA apex: 
    https://github.com/NVIDIA/apex with minor changes.

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

#include <unordered_map>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace layer_norm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Params> 
struct LaunchParams{

    size_t workspace_bytes;
    size_t barrier_size;

    cudaDeviceProp * props;

    cudaStream_t stream;

    Params params;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ParamsBase {
    ParamsBase()
        : ctas_per_col(0)
        , rows(0)
        , cols(0)
        , x(nullptr)
        , mu(nullptr)
        , rs(nullptr)
        , gamma(nullptr)
        , workspace(nullptr)
        , barrier(nullptr)
    {
    }

    // For Multi-CTA, number of different CTA groups. Otherwise same as gridDim.x.
    int ctas_per_col;

    // Input is interpreted as matrix. We normalize across columns.
    int rows;
    int cols;

    // Common data pointers.
    void *x;
    void *mu;
    void *rs;
    void *gamma;

    // Multi-CTA workspace in gmem.
    void *workspace;

    // Multi-CTA sync barriers in gmem.
    int *barrier;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FwdParams : public ParamsBase {
    FwdParams()
        : ParamsBase()
        , z(nullptr)
        , beta(nullptr)
        , epsilon(0.f)
    {
    }

    // Output of LN FWD.
    void *z;
    void *beta;
    float epsilon;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct BwdParams : public ParamsBase {
    BwdParams()
        : ParamsBase()
        , dz(nullptr)
        , dbeta_part(nullptr)
        , dgamma_part(nullptr)
        , dx(nullptr)
        , dbeta(nullptr)
        , dgamma(nullptr)
    {
    }

    // Input: gradient wrt. LN FWD output.
    void *dz;

    // Workspace for Wgrad pre-reduction.
    void *dbeta_part;
    void *dgamma_part;

    // Output: Dgrad.
    void *dx;
    // Output: Wgrad.
    void *dbeta;
    void *dgamma;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

using FwdFunction = std::function<void(LaunchParams<FwdParams>&, const bool)>;
using BwdFunction = std::function<void(LaunchParams<BwdParams>&, const bool)>;
using FunctionKey = uint64_t;
using FwdRegistry = std::unordered_map<FunctionKey, FwdFunction>;
using BwdRegistry = std::unordered_map<FunctionKey, BwdFunction>;

extern FwdRegistry FWD_FUNCS;
extern BwdRegistry BWD_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct TypeId{};

template<>
struct TypeId<fp16>{
    constexpr static uint32_t Value = 0;
};

template<>
struct TypeId<bf16>{
    constexpr static uint32_t Value = 1;
};

template<>
struct TypeId<fp32>{
    constexpr static uint32_t Value = 2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int S>
struct Type2Key{
    constexpr static uint32_t Value = TypeId<T>::Value << S;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct WeightType2Key : public Type2Key<T, 0>{};

template<typename T>
struct InputType2Key : public Type2Key<T, 2>{};

template<typename T>
struct OutputType2Key : public Type2Key<T, 4>{};

template<typename T>
struct ComputeType2Key : public Type2Key<T, 6>{};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename O, typename C>
struct Types2Key{
    constexpr static uint32_t Value = WeightType2Key<W>::Value | InputType2Key<I>::Value | OutputType2Key<O>::Value | ComputeType2Key<C>::Value;
    constexpr static inline uint64_t get(const uint64_t hidden_size){
        constexpr uint64_t type_key = Value;
        return (type_key << 32) | hidden_size;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct FwdRegistrar{
    FwdRegistrar(FwdFunction f){
        uint64_t key = Types2Key<W,I,O,C>::get(HIDDEN_SIZE);
        FWD_FUNCS.insert({ key, f });
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct BwdRegistrar{
    BwdRegistrar(BwdFunction f){
        uint64_t key = Types2Key<W,I,O,C>::get(HIDDEN_SIZE);
        BWD_FUNCS.insert({ key, f });
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace layer_norm
