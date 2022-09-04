/*
 * Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
 
#include "fastertransformer/cuda/masked_multihead_attention_utils.h"
namespace fastertransformer 
{

template<typename T>
struct Vec_t {};
template<>
struct Vec_t<float> {
    using Type = float2;
};
template<>
struct Vec_t<half> {
    using Type = uint32_t;
};

#ifdef ENABLE_BF16
template<>
struct Vec_t<__nv_bfloat16> {
    using Type = __nv_bfloat162;
};
#endif


template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* q_buf,
                                                   T* k_buf,
                                                   T* v_buf,
                                                   const T* __restrict QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int head_num,
                                                   const int size_per_head,
                                                   const int rotary_embedding_dim)
{
    using Vec_t = typename Vec_t<T>::Type;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tidx = threadIdx.x;
    if (tidx * 2 >= size_per_head) {
        return;
    }

    const int batch_time_idx = seq_len * batch_idx + seq_idx;
    const int hidden_idx = head_idx * size_per_head + tidx * 2;
    const int n = head_num * size_per_head;

    // src QKV: [batch, time, 3, head, hidden]
    const int q_idx = batch_time_idx * 3 * n + hidden_idx;
    const int k_idx = batch_time_idx * 3 * n + hidden_idx + n;
    const int v_idx = batch_time_idx * 3 * n + hidden_idx + 2 * n;

    Vec_t q = *reinterpret_cast<const Vec_t*>(&QKV[q_idx]);
    Vec_t k = *reinterpret_cast<const Vec_t*>(&QKV[k_idx]);
    Vec_t v = *reinterpret_cast<const Vec_t*>(&QKV[v_idx]);

    if(qkv_bias != nullptr){
    // qkv_bias: [3, head, hidden]
        Vec_t q_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx]);
        Vec_t k_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + n]);
        Vec_t v_bias = *reinterpret_cast<const Vec_t*>(&qkv_bias[hidden_idx + 2 * n]);

        q = mmha::add(q, q_bias);
        k = mmha::add(k, k_bias);
        v = mmha::add(v, v_bias);
    }

    mmha::apply_rotary_embedding(q, k, tidx, rotary_embedding_dim, seq_idx);

    // q_buf, k_buf, v_buf: [batch, head_num, seq_len, size_per_head]
    const int dest_idx = size_per_head * seq_len * head_num * batch_idx + size_per_head * seq_len * head_idx
                         + size_per_head * seq_idx + tidx * 2;

    *reinterpret_cast<Vec_t*>(&q_buf[dest_idx]) = q;
    *reinterpret_cast<Vec_t*>(&k_buf[dest_idx]) = k;
    *reinterpret_cast<Vec_t*>(&v_buf[dest_idx]) = v;
}

template <typename T>
void add_fusedQKV_bias_transpose_kernelLauncher(
  T* q_buf,
  T* k_buf,
  T* v_buf,
  T* QKV,
  const T* qkv_bias,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const int size_per_head,
  const int rotary_embedding_dim,
  cudaStream_t stream)
{
    if (rotary_embedding_dim == 0) {
        const int m = batch_size * seq_len;
        const int n = head_num * size_per_head;
        dim3 block(384);
        dim3 grid((int)(ceil(1.0 * m * n / 384)));
        add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
            q_buf, k_buf, v_buf, QKV, qkv_bias, batch_size, seq_len, head_num, size_per_head);
    }
    else {
        // To implement rotary embeddings, each thread processes two QKV elems:
        dim3 block((size_per_head / 2 + 31) / 32 * 32);
        dim3 grid(seq_len, head_num, batch_size);
        add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
            q_buf, k_buf, v_buf, QKV, qkv_bias, batch_size, seq_len, head_num, size_per_head, rotary_embedding_dim);
    }
}

template void add_fusedQKV_bias_transpose_kernelLauncher(
    float* q_buf,
    float* k_buf,
    float* v_buf,
    float* QKV,
    const float* qkv_bias,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int rotary_embedding_dim,
    cudaStream_t stream);

template void add_fusedQKV_bias_transpose_kernelLauncher(
    half* q_buf,
    half* k_buf,
    half* v_buf,
    half* QKV,
    const half* qkv_bias,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const int size_per_head,
    const int rotary_embedding_dim,
    cudaStream_t stream);
      
} // namespace fastertransformer