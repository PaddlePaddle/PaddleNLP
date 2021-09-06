/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "fastertransformer/arguments.h"
#include "fastertransformer/cuda/topk_kernels.cuh"

namespace fastertransformer {

template <typename T>
void add_bias_input_kernelLauncher(T* out,
                                   const T* input_tensor,
                                   const T* bias,
                                   int m,
                                   int n,
                                   cudaStream_t stream);

template <typename T>
void layernorm_kernelLauncher(T* out,
                              const T* input_tensor,
                              const T* gamma,
                              const T* beta,
                              int m,
                              int n,
                              cudaStream_t stream);
template <typename T>
void add_bias_input_pre_layernorm_kernelLauncher(T* out,
                                                 T* bias_and_input,
                                                 const T* input,
                                                 const T* bias,
                                                 const T* gamma,
                                                 const T* beta,
                                                 int m,
                                                 int n,
                                                 cudaStream_t stream);
}  // namespace fastertransformer
