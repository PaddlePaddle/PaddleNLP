/*
 * Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
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
void transpose_cache_batch_major_kernelLauncher(T* k_dst,
                                                T* v_dst,
                                                const T* k_src,
                                                const T* v_src,
                                                const int* memory_seq_len,
                                                const int local_batch_size,
                                                const int memory_max_seq_len,
                                                const int cache_max_seq_len,
                                                const int size_per_head,
                                                const int local_head_num,
                                                cudaStream_t stream);

template <typename T>
void channel_wise_quantize_kernelLauncher(const T* input,
                                        int8_t* ffn_quantize_input_buf_,
                                        T* scale,
                                        const int batch_size,
                                        const int hidden_units,
                                        cudaStream_t stream,
                                        bool use_COL32 = true);

template <typename T>
void channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                          const T* scale,
                                          const T* weight_scale,
                                          T* output,
                                          const int batch_size,
                                          const int hidden_units,
                                          cudaStream_t stream,
                                          bool use_COL32 = true);

template <typename T>
void qkv_channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                          const T* scale,
                                          const T* qw_scale,
                                          const T* kw_scale,
                                          const T* vw_scale,
                                          T* query,
                                          T* key,
                                          T* value,
                                          const int batch_size,
                                          const int hidden_units,
                                          cudaStream_t stream,
                                          bool use_COL32 = true);

template <typename T>
void mem_kv_channel_wise_dequantize_kernelLauncher(const int32_t* input,
                                          const T* scale,
                                          const T* kw_scale,
                                          const T* vw_scale,
                                          T* key,
                                          T* value,
                                          const int m,
                                          const int hidden_units,
                                          cudaStream_t stream,
                                          bool use_COL32 = true);

template <typename T>
void dequant_add_bias_act_quant_COL32_int32I_int8O_kernelLauncher(
                                          int8_t *out,
                                          const int32_t* input,
                                          const T* bias,
                                          T* ffn_inner,
                                          const int batch_size,
                                          const int hidden_units,
                                          const T* weight_scale,
                                          T* scale,
                                          ActivationType activation_type,
                                          cudaStream_t stream);

}
