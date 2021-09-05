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

#include "fastertransformer/common.h"

#include <assert.h>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include "cuda_kernels.h"
namespace fastertransformer {

template <typename T>
__global__ void add_bias_input_pre_layernorm(T* out,
                                             T* bias_and_input,
                                             const T* input,
                                             const T* bias,
                                             const T* gamma,
                                             const T* beta,
                                             int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  local_out += (float)(out[blockIdx.x * n + tid] + input[blockIdx.x * n + tid] +
                       __ldg(&bias[tid]));
  bias_and_input[blockIdx.x * n + tid] = (T)(local_out);

  mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if (threadIdx.x == 0) s_variance = variance / n + 1e-6f;
  __syncthreads();

  out[blockIdx.x * n + tid] = (T)(((local_out - s_mean) * rsqrtf(s_variance)) *
                                      (float)(__ldg(&gamma[tid])) +
                                  (float)(__ldg(&beta[tid])));
}

template <typename T>
__global__ void add_bias_input(T* out,
                               T* bias_and_input,
                               const T* bias,
                               int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  float local_out =
      (float)(out[blockIdx.x * n + tid] + bias_and_input[blockIdx.x * n + tid]);

  out[blockIdx.x * n + tid] = local_out;
}

template <typename T>
__global__ void encoder_layernorm(
    T* out, const T* input, const T* gamma, const T* beta, int n) {
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;

  float mean = 0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  local_out += (float)(input[blockIdx.x * n + tid]);

  mean = blockReduceSum<float>(input[blockIdx.x * n + tid]);
  if (threadIdx.x == 0) s_mean = mean / n;
  __syncthreads();

  variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
  if (threadIdx.x == 0) s_variance = rsqrtf(variance / n + 1e-6f);
  __syncthreads();

  out[blockIdx.x * n + tid] =
      (T)(((local_out - s_mean) * s_variance) * (float)(__ldg(&gamma[tid])) +
          (float)(__ldg(&beta[tid])));
}

template <typename T>
void add_bias_input_pre_layernorm_kernelLauncher(T* out,
                                                 T* bias_and_input,
                                                 const T* input,
                                                 const T* bias,
                                                 const T* gamma,
                                                 const T* beta,
                                                 int m,
                                                 int n,
                                                 cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input_pre_layernorm<T><<<grid, block, 0, stream>>>(
      out, bias_and_input, input, bias, gamma, beta, n);
}

template <typename T>
void add_bias_input_kernelLauncher(T* out,
                                   const T* bias_and_input,
                                   const T* bias,
                                   int m,
                                   int n,
                                   cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  add_bias_input<T><<<grid, block, 0, stream>>>(out, bias_and_input, bias, n);
}

template <typename T>
void layernorm_kernelLauncher(T* out,
                              const T* input,
                              const T* gamma,
                              const T* beta,
                              int m,
                              int n,
                              cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(n);
  assert(n <= 1024);
  // if(n == 768 || n == 1024)
  //   encoder_layernorm_v2<T><<<grid, n / 4, 0, stream>>>(out, input, bias,
  //   gamma, beta, n);
  // else
  encoder_layernorm<T><<<grid, block, 0, stream>>>(out, input, gamma, beta, n);
}

}  // namespace
