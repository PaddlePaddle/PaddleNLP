/*
 * Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastertransformer/utils/common_structure.h"


namespace fastertransformer {

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
    int8_t* out,
    const int32_t* input,
    const T* bias,
    T* ffn_inner,
    const int batch_size,
    const int hidden_units,
    const T* weight_scale,
    T* scale,
    ActivationType activation_type,
    cudaStream_t stream);

template <typename T>
void dequant_add_bias_act_quant_kernelLauncher(int32_t* out_quant_buf,
                                               int8_t* input_quant_buf,
                                               T* scale,
                                               T* ffn_inner,
                                               const T* bias,
                                               const T* w_scale,
                                               int m,
                                               int n,
                                               ActivationType activation_type,
                                               cudaStream_t stream);

template <typename T>
void dequant_add_bias_input_layernorm_2_quant_kernelLauncher(
    const int32_t* quant_output_buf,
    const T* input,
    const T* gamma,
    const T* beta,
    const T* bias,
    const T* w_scale,
    T* output,
    T* norm_output,
    int8_t* quantize_input_buf_,
    T* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template <typename T>
void add_bias_input_layernorm_2_quant_kernelLauncher(
    const int32_t* quant_output_buf,
    const T* input,
    const T* gamma,
    const T* beta,
    const T* bias,
    const T* w_scale,
    T* output,
    T* norm_output,
    int8_t* quantize_input_buf_,
    T* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template <typename T>
void dequant_add_bias_input_layernorm_2_kernelLauncher(
    const int32_t* quant_output_buf,
    const T* input,
    const T* gamma,
    const T* beta,
    const T* bias,
    const T* w_scale,
    T* output,
    T* norm_output,
    int8_t* quantize_input_buf_,
    T* scale,
    int m,
    int n,
    cudaStream_t stream,
    bool use_COL32);

template <typename T>
void dequant_add_bias_input_kernelLauncher(const int32_t* quant_in,
                                           const T* scale,
                                           const T* weight_scale,
                                           T* output,
                                           const T* bias,
                                           const T* input,
                                           const int m,
                                           const int n,
                                           cudaStream_t stream,
                                           bool use_COL32);

template <typename T>
void layer_norm_quant(const T* from_tensor,
                      const T* gamma,
                      const T* beta,
                      T* norm_from_tensor_buf_,
                      T* scale,
                      int8_t* quantize_input_buf_,
                      const int m,
                      const int n,
                      cudaStream_t stream,
                      bool use_COL32);

}
