// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "fp8_common.h"  // NOLINT

typedef struct {
    void const* act;
    void const* weight;
    void const* bias;
    void* output;
    int32_t m, n, k;
    float alpha;
    cudaStream_t stream;
} GemmParams;

inline bool enable_cuda_core_fp8_gemm() {
    static const char* enable_cuda_core_fp8_env = std::getenv("FLAGS_cuda_core_fp8_gemm");
    static const bool enable_cuda_core_fp8_gemm =
            enable_cuda_core_fp8_env != nullptr && std::string(enable_cuda_core_fp8_env) == "1";
    return enable_cuda_core_fp8_gemm;
}

template <typename InputType, typename OutputType>
bool cuda_core_gemm_launcher(GemmParams const& params);
