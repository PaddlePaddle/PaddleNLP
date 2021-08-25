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
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "fastertransformer/arguments.h"
#include "fastertransformer/cuda/topk_kernels.cuh"

namespace fastertransformer {

#define FINAL_MASK 0xffffffff
#define CUDART_PI_F 3.141592654f

/* ********************************** common kernel
 * *********************************** */

template <typename T>
void init_kernelLauncher(bool* finished,
                         int* sequence_length,
                         int* word_ids,
                         T* cum_log_probs,
                         const int sentence_id,
                         const int batch_size,
                         const int beam_width,
                         cudaStream_t stream);


template <typename T>
void init_kernelLauncher_v2(bool* finished,
                            int* sequence_length,
                            int* word_ids,
                            T* cum_log_probs,
                            const int sentence_id,
                            const int batch_size,
                            const int beam_width,
                            cudaStream_t stream);

template <typename T>
void add_bias_act_kernelLauncher(
    T* out, const T* bias, int m, int n, cudaStream_t stream);

template <typename T>
void add_bias_input_layernorm_kernelLauncher(T* out,
                                             const T* input_tensor,
                                             const T* bias,
                                             const T* gamma,
                                             const T* beta,
                                             int m,
                                             int n,
                                             cudaStream_t stream);

template <typename T>
void embedding_lookup_sine_position_encoding_kernel_launcher(
    T* from_tensor,
    const T* embedding_table,
    const T* position_encoding_table,
    const int* word_ids,
    const int batch_size,
    const int hidden_units,
    cudaStream_t stream);

template <typename T>
void remove_sequence_length_padding_kernelLauncher(const T* src,
                                                   T* tgt,
                                                   const int* tmp_mask_offset,
                                                   int* mask_offset,
                                                   const int m,
                                                   const int n,
                                                   cudaStream_t stream);

template <typename T>
void rebuild_sequence_length_padding_kernelLauncher(const T* src,
                                                    T* tgt,
                                                    const int* mask_offset,
                                                    const int m,
                                                    const int n,
                                                    cudaStream_t stream);

template <typename T>
void embedding_position_lookups_kernel_launcher(T* from_tensor,
                                                const T* embedding_table,
                                                const T* pos_table,
                                                const int* word_ids,
                                                const int batch_size,
                                                const int hidden_units,
                                                int step,
                                                cudaStream_t stream);

template <typename T>
void apply_temperature_penalty_kernelLauncher(T* logits,
                                              const T temperature,
                                              const int m,
                                              const int n,
                                              cudaStream_t stream);

template <typename T>
void transpose(T* out,
               const T* in,
               int batch,
               int height,
               int width,
               int stride,
               cudaStream_t stream);
/* *************************** end of common kernel
 * *********************************** */
void build_sequence_length_padding_offset_kernelLauncher(
    const int* sequence_length,
    const int batch_size,
    const int max_seq_len,
    int* valid_word_num,
    int* tmp_mask_offset,
    cudaStream_t stream);

/* *************************** end of common kernel
 * *********************************** */

/* ********************************** BeamSearch kernel
 * *********************************** */

void broadcast_kernelLauncher(float* log_probs,
                              float* cum_log_probs,
                              const int batch_size,
                              const int beam_width,
                              const int vocab_size,
                              cudaStream_t stream);

void update_logits(float* logits,
                   const float* bias,
                   const int end_ids,
                   const bool* finished,
                   const int m,
                   const int n,
                   cudaStream_t stream);

template <bool ALIVE>
void update_logits(float* logits,
                   const float* bias,
                   const int end_ids,
                   const bool* finished,
                   const int m,
                   const int n,
                   cudaStream_t stream);

template <typename T>
void apply_logit_penalties(int step,
                           T* log_probs,
                           int* current_ids,
                           int* previous_ids,
                           int* parent_ids,
                           Gpt2Arguments args,
                           cudaStream_t stream);

void update_kernelLauncher(float* log_probs,
                           float* cum_log_probs,
                           bool* finished,
                           int* parent_ids,
                           int* sequence_length,
                           int* word_ids,
                           int* output_ids,
                           const int batch_size,
                           const int beam_width,
                           const int vocab_size,
                           cudaStream_t stream,
                           const int end_id,
                           int* finished_count);

void update_kernelLauncher_v2(bool* finished,
                              int* parent_ids,
                              int* sequence_length,
                              int* word_ids,
                              int* output_ids,
                              int* finished_count,
                              DecodingBeamsearchArguments args,
                              cudaStream_t stream);

template <typename T>
void update_KV_cache_kernelLauncher(T** key_cache,
                                    T** value_cache,
                                    const int* beam_ids,
                                    const int batch_size,
                                    const int beam_width,
                                    const int hidden_dim,
                                    const int step,
                                    const int cache_size,
                                    const int decoder_layers,
                                    cudaStream_t stream);

void gather_tree_kernel_launcher(int max_time,
                                 int batch_size,
                                 int beam_width,
                                 int* step_ids,
                                 int* parent_ids,
                                 int* max_sequence_lengths,
                                 int end_token,
                                 int* beams,
                                 cudaStream_t stream);

/* *************************** end of BeamSearch kernel
 * *********************************** */

/* ********************************** Sampling kernel
 * *********************************** */

template <typename T>
size_t get_topp_sort_temp_storage_size(const T* log_probs,
                                       const int* id_vals,
                                       T* sorted_log_probs,
                                       int* sorted_id_vals,
                                       int* topp_offset_buf,
                                       const int batch_size,
                                       const int vocab_size);

void sampling_init_kernelLauncher(bool* finished,
                                  int* sequence_length,
                                  int* word_ids,
                                  const int start_id,
                                  const int batch_size,
                                  cudaStream_t stream);

void topp_initialization_kernelLauncher(bool* finished,
                                        int* sequence_length,
                                        int* word_ids,
                                        int* topp_id_val_buf,
                                        int* topp_offset_buf,
                                        const int logits_buf_size,
                                        DecodingSamplingArguments args,
                                        cudaStream_t stream);

void init_topp_id_val_kernel_kernelLauncher(int* topp_id_val_buf,
                                            int* topp_offset_buf,
                                            const int batch_size,
                                            const int vocab_size,
                                            cudaStream_t stream);

template <typename T>
void update_logits_without_softmax(T* logits,
                                   const T* bias,
                                   const int end_ids,
                                   const bool* finished,
                                   const int m,
                                   const int n,
                                   cudaStream_t stream);

template <typename T>
void softmax_kernelLauncher(T* logits,
                            const T* bias,
                            const int end_ids,
                            const bool* finished,
                            const int m,
                            const int n,
                            cudaStream_t stream);

/* *************************** end of Sampling kernel
 * *********************************** */

/* *********************************** Debug tools
 * *********************************** */

template <typename T>
void print_first_k(const T* buf, uint size, cudaStream_t stream);

template <typename T>
void print_abs_mean(const T* buf, uint size, cudaStream_t stream);

/* **************************** end of Debug tools
 * *********************************** */

/* *************************** depreciated kernels
 * *********************************** */

void topK(const float* log_probs,
          int* ids,
          const int batch_size,
          const int beam_width,
          const int vocab_size,
          cudaStream_t stream);

template <typename T>
void embedding_lookup(const T* embedding_table,
                      const int* word_ids,
                      T* from_tensor,
                      const int batch_size,
                      const int beam_width,
                      const int hidden_units,
                      cudaStream_t stream);

template <typename T>
void sine_position_encoder(
    T* output, int step, int m, int n, cudaStream_t stream);

/* ******************** end of depreciated kernels
 * *********************************** */

}  // namespace fastertransformer
