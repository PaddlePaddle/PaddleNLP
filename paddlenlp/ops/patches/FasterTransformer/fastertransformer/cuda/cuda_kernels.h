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
                            bool* alive_finished,
                            int* sequence_length,
                            int* word_ids,
                            T* cum_log_probs,
                            const int sentence_id,
                            const int batch_size,
                            const int beam_width,
                            cudaStream_t stream);

template <typename T>
void update_logits_v2(T* logits,
                      const T* bias,
                      const int end_id,
                      const bool* finished,
                      const int m,
                      const int n,
                      cudaStream_t stream);

// Encoder kernels
#ifdef WITH_ENCODER
// NOTE: Remove or replace this function.
// template <typename T>
// void add_bias_input_kernelLauncher(T* out,
//                                    const T* input_tensor,
//                                    const T* bias,
//                                    int m,
//                                    int n,
//                                    cudaStream_t stream);

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

template <typename T>
void add_bias_act_kernelLauncher(
    T* out, const T* bias, int m, int n, cudaStream_t stream, bool is_gelu);

// End of encoder kernels
#endif

template <typename T>
void embedding_position_lookups_bart_kernel_launcher(
    T* from_tensor,
    const T* embedding_table,
    const T* position_encoding_table,
    const int* word_ids,
    const int batch_size,
    const int hidden_units,
    cudaStream_t stream);

template <typename T>
void update_with_force_decodingLauncher(const int* trg_word,
                                        const int* trg_length,
                                        bool* finished,
                                        int* word_ids,
                                        int* sequence_length,
                                        int* parent_ids_buf,
                                        int* parent_ids,
                                        int* output_ids,
                                        T* scores,
                                        bool keep_alive_beam,
                                        const int batch_size,
                                        const int beam_width,
                                        const int max_trg_len,
                                        const int step,
                                        cudaStream_t stream);

template <typename T>
void update_KV_cache_kernelLauncher_v2(T** key_cache,
                                       T** value_cache,
                                       const int* beam_ids,
                                       const bool* finished,
                                       const int batch_size,
                                       const int beam_width,
                                       const int head_num,
                                       const int size_per_head,
                                       const int step,
                                       const int decoder_max_seq_len,
                                       const int cache_size,
                                       const int decoder_layers,
                                       cudaStream_t stream,
                                       const int memory_max_seq_len);

template <typename T>
void embeddings_kernel_launcher(T* from_tensor,
                                const T* embedding_table,
                                const T* position_encoding_table,
                                const T* type_table,
                                const int* memory_sequence_length,
                                const int* type_id,
                                const int* word_ids,
                                const int step,
                                const int batch_size,
                                const int hidden_units,
                                const bool pos_bias,
                                cudaStream_t stream);

template <typename T>
void init_cache_kernel_launcher(const float* cache_k,
                                const float* cache_v,
                                const int* memory_sequence_length,
                                T* k_tgt,
                                T* v_tgt,
                                int n_head,
                                int size_per_head,
                                int mem_len,
                                int batch_size,
                                int beam_size,
                                cudaStream_t stream);

}  // namespace fastertransformer
