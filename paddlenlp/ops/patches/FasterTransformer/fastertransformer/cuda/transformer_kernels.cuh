/*
 * Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
void t5_layer_norm(const T* from_tensor,
                   const T* gamma,
                   const T* beta,
                   T* norm_from_tensor_buf_,
                   const int m,
                   const int n,
                   cudaStream_t stream);

template <typename T>
void add_bias_input_t5_layernorm_2_kernelLauncher(const T* input,
                                                  const T* gamma,
                                                  const T* beta,
                                                  const T* bias,
                                                  T* output,
                                                  T* norm_output,
                                                  int m,
                                                  int n,
                                                  cudaStream_t stream);

template <typename T>
void gated_add_bias_act_kernelLauncher(T* out,
                                       const T* bias0,
                                       const T* bias1,
                                       int m,
                                       int n,
                                       ActivationType activation_type,
                                       cudaStream_t stream);

}  // namespace fastertransformer
