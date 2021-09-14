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
__global__ void add_bias_relu_encoder(T* out, const T* bias, int m, int n) {
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    T reg_bias = __ldg(&bias[id % n]);
    T val = out[id] + reg_bias;
    out[id] = (T)(val > 0.0f ? val : 0.0f);
  }
}

template <>
__global__ void add_bias_relu_encoder(half* out,
                                      const half* bias,
                                      int m,
                                      int n) {
  half2* out_ptr = (half2*)out;
  const half2* bias_ptr = (half2*)bias;

  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    half2 reg_bias = __ldg(&bias_ptr[id % n]);
    half2 val = out_ptr[id] + reg_bias;
    val.x = val.x > (half)0.0f ? val.x : (half)0.0f;
    val.y = val.y > (half)0.0f ? val.y : (half)0.0f;
    out_ptr[id] = val;
  }
}

template void add_bias_act_kernelLauncher<float>(float* out,
                                                 const float* bias,
                                                 int m,
                                                 int n,
                                                 cudaStream_t stream,
                                                 bool is_gelu);

template void add_bias_act_kernelLauncher<half>(half* out,
                                                const half* bias,
                                                int m,
                                                int n,
                                                cudaStream_t stream,
                                                bool is_gelu);

template <typename T>
void add_bias_act_kernelLauncher(
    T* out, const T* bias, int m, int n, cudaStream_t stream, bool is_gelu) {
  dim3 grid(ceil(m / 4.));
  dim3 block(n / 4);
  assert(block.x <= 1024);
  if (is_gelu)
    add_bias_act<T><<<grid, block, 0, stream>>>(out, bias, m, n);
  else
    add_bias_relu_encoder<T><<<grid, block, 0, stream>>>(out, bias, m, n);
}

}  // namespace fastertransformer
