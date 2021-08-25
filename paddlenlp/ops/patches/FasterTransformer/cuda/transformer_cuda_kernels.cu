/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

template <typename T>
__global__ void update_logits_kernel(T* logits,
                                     const T* bias,
                                     const T* logits_mask,
                                     const int end_id,
                                     const bool* finished,
                                     const int n) {
  int bid = blockIdx.x;
  bool finish = finished[bid];
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    if (finish) {
      logits[offset + tid] = (tid == end_id) ? FLT_MAX : -1 * FLT_MAX;
    } else {
      logits[offset + tid] += bias[tid];
    }
    if (logits_mask) {
      logits[offset + tid] += logits_mask[tid];
    }
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

template <typename T>
void update_logits_with_mask(T* logits,
                             const T* bias,
                             const T* logits_mask,
                             const int end_id,
                             const bool* finished,
                             const int m,
                             const int n,
                             cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big.
   */
  update_logits_kernel<<<grid, block, 0, stream>>>(
      logits, bias, logits_mask, end_id, finished, n);
}

template <typename T>
__global__ void softmax_with_mask_kernel(T* logits,
                                         const T* bias,
                                         const T* logits_mask,
                                         const int end_id,
                                         const bool* finished,
                                         const int n) {
  int bid = blockIdx.x;
  bool finish = (finished != nullptr) ? finished[bid] : false;
  int offset = bid * n;

  float max_val = -1 * FLT_MAX;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    if (finish) {
      logits[offset + tid] = (tid == end_id) ? MAX_T_VAL : -MAX_T_VAL;
    } else {
      T bias_val = (bias != nullptr) ? bias[tid] : (T)0.0f;
      logits[offset + tid] += bias_val;
    }
    if (logits_mask) {
      logits[offset + tid] += logits_mask[tid];
    }
    max_val = max(max_val, (float)logits[offset + tid]);
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
    logits[offset + tid] = ((float)logits[offset + tid] / s_sum_val);
  }
}

template <typename T>
void softmax_with_mask_kernelLauncher(T* logits,
                                      const T* bias,
                                      const T* logits_mask,
                                      const int end_id,
                                      const bool* finished,
                                      const int m,
                                      const int n,
                                      cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));
  /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big.
   */
  softmax_with_mask_kernel<<<grid, block, 0, stream>>>(
      logits, bias, logits_mask, end_id, finished, n);
}

template void update_logits_with_mask(float* logits,
                                      const float* bias,
                                      const float* logits_mask,
                                      const int end_id,
                                      const bool* finished,
                                      const int m,
                                      const int n,
                                      cudaStream_t stream);

template void update_logits_with_mask(half* logits,
                                      const half* bias,
                                      const half* logits_mask,
                                      const int end_id,
                                      const bool* finished,
                                      const int m,
                                      const int n,
                                      cudaStream_t stream);

template void softmax_with_mask_kernelLauncher(float* logits,
                                               const float* bias,
                                               const float* logits_mask,
                                               const int end_id,
                                               const bool* finished,
                                               const int m,
                                               const int n,
                                               cudaStream_t stream);

template void softmax_with_mask_kernelLauncher(half* logits,
                                               const half* bias,
                                               const half* logits_mask,
                                               const int end_id,
                                               const bool* finished,
                                               const int m,
                                               const int n,
                                               cudaStream_t stream);
}
