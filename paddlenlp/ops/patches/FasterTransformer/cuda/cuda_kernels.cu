/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

namespace fastertransformer {

template <typename T, bool ALIVE>
__global__ void update_logits_kernel(T* logits,
                                     const T* bias,
                                     const int end_id,
                                     const bool* finished,
                                     const int n) {
  int bid = blockIdx.x;
  bool finish = ALIVE ? false : finished[bid];
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    if (finish)
      logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
    else
      logits[offset + tid] += bias[tid];
    max_val = max(max_val, logits[offset + tid]);
  }

  max_val = blockReduceMax<float>((float)max_val);
  if (threadIdx.x == 0) s_max_val = max_val;
  __syncthreads();

  float sum_val = 0.0f;
  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    logits[offset + tid] = __expf((float)logits[offset + tid] - s_max_val);
    sum_val += (float)logits[offset + tid];
  }

  sum_val = blockReduceSum<float>(sum_val);
  if (threadIdx.x == 0) s_sum_val = sum_val;
  __syncthreads();

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    logits[offset + tid] = logf((float)logits[offset + tid] / s_sum_val);
  }
}

void update_logits_v2(float* logits,
                      const float* bias,
                      const int end_id,
                      const bool* finished,
                      const int m,
                      const int n,
                      cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big.
   */
  update_logits_kernel<float, true><<<grid, block, 0, stream>>>(
      logits, bias, end_id, finished, n);
}

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
                               const T* bias_and_input,
                               const T* bias,
                               int n) {
  int tid = threadIdx.x;

  // __shared__ float s_mean;
  // __shared__ float s_variance;
  // float mean = 0.0f;
  // float variance = 0.0f;
  // out: [bs, max_len, hidden_size]

  float local_out = (float)(out[blockIdx.x * n + tid] +
                            bias_and_input[blockIdx.x * n + tid]) +
                    bias[tid];

  out[blockIdx.x * n + tid] = local_out;
}

// Encoder kernels
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
  encoder_layernorm<T><<<grid, block, 0, stream>>>(out, input, gamma, beta, n);
}

template void add_bias_input_pre_layernorm_kernelLauncher(float* out,
                                                          float* bias_and_input,
                                                          const float* input,
                                                          const float* bias,
                                                          const float* gamma,
                                                          const float* beta,
                                                          int m,
                                                          int n,
                                                          cudaStream_t stream);

template void add_bias_input_kernelLauncher(float* out,
                                            const float* bias_and_input,
                                            const float* bias,
                                            int m,
                                            int n,
                                            cudaStream_t stream);

template void layernorm_kernelLauncher(float* out,
                                       const float* input,
                                       const float* gamma,
                                       const float* beta,
                                       int m,
                                       int n,
                                       cudaStream_t stream);
// End of encoder kernels

}  // namespace fastertransformer
