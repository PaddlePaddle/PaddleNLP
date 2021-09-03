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

template <typename T, bool ALIVE = false>
__global__ void init_kernel(bool* finished,
                            int* sequence_length,
                            int* word_ids,
                            T* cum_log_probs,
                            const int sentence_id,
                            const int beam_width,
                            const int batch_size) {
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : 1e20f;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < batch_size * beam_width;
       index += blockDim.x * gridDim.x) {
    finished[index] = false;
    sequence_length[index] = 0;
    if (ALIVE) {
      if (index < batch_size * beam_width / 2) word_ids[index] = sentence_id;
      cum_log_probs[index] =
          (index % beam_width == beam_width / 2) ? (T)0.0f : -MAX_T_VAL;
    } else {
      word_ids[index] = sentence_id;
      cum_log_probs[index] = (index % beam_width == 0) ? (T)0.0f : -MAX_T_VAL;
    }
  }
}

template <typename T>
void init_kernelLauncher_v2(bool* finished,
                            int* sequence_length,
                            int* word_ids,
                            T* cum_log_probs,
                            const int sentence_id,
                            const int batch_size,
                            const int beam_width,
                            cudaStream_t stream) {
  dim3 grid((int)ceil(batch_size * beam_width * 1.0 / 256));
  dim3 block(256);

  init_kernel<T, true><<<grid, block, 0, stream>>>(finished,
                                                   sequence_length,
                                                   word_ids,
                                                   cum_log_probs,
                                                   sentence_id,
                                                   beam_width,
                                                   batch_size);
}

template void init_kernelLauncher_v2(bool* finished,
                                     int* sequence_length,
                                     int* word_ids,
                                     float* cum_log_probs,
                                     const int sentence_id,
                                     const int batch_size,
                                     const int beam_width,
                                     cudaStream_t stream);

template void init_kernelLauncher_v2(bool* finished,
                                     int* sequence_length,
                                     int* word_ids,
                                     half* cum_log_probs,
                                     const int sentence_id,
                                     const int batch_size,
                                     const int beam_width,
                                     cudaStream_t stream);

}  // end of name space fastertransformer
