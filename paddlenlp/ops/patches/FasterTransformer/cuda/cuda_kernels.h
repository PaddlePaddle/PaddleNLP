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


template <typename T>
void init_kernelLauncher_v2(bool* finished,
                            int* sequence_length,
                            int* word_ids,
                            T* cum_log_probs,
                            const int sentence_id,
                            const int batch_size,
                            const int beam_width,
                            cudaStream_t stream);

void update_logits_v2(float* logits,
                      const float* bias,
                      const int end_id,
                      const bool* finished,
                      const int m,
                      const int n,
                      cudaStream_t stream);

// Encoder kernels
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
// End of encoder kernels

template <typename T>
void add_bias_act_kernelLauncher(
    T* out, const T* bias, int m, int n, cudaStream_t stream, bool is_gelu);

template <typename T>
void embedding_position_lookups_bart_kernel_launcher(
    T* from_tensor,
    const T* embedding_table,
    const T* position_encoding_table,
    const int* word_ids,
    const int batch_size,
    const int hidden_units,
    cudaStream_t stream);

}  // namespace fastertransformer
