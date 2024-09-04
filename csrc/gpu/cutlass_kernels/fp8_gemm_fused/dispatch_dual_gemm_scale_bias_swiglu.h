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
#include "fp8_common.h"

#include "fp8_gemm_fused/dual_gemm_scale_bias_swiglu_16_32_64_stages3.h"
#include "fp8_gemm_fused/dual_gemm_scale_bias_swiglu_16_64_64_stages4.h"
#include "fp8_gemm_fused/dual_gemm_scale_bias_swiglu_64_64_64_stages3.h"

template <typename InputType, typename BiasType, typename OutType>
bool dispatch_dual_gemm_scale_bias_swiglu(DualGemmEpilogueAllParams params) {
    if(params.M<=32){
        return dual_gemm_scale_bias_swiglu_16_32_64_stages3<InputType, BiasType, OutType>(params);
    } else if(params.M>32 && params.M<=64) {
        return dual_gemm_scale_bias_swiglu_16_64_64_stages4<InputType, BiasType, OutType>(params);
    } else {
        return dual_gemm_scale_bias_swiglu_64_64_64_stages4<InputType, BiasType, OutType>(params);
    }
}

