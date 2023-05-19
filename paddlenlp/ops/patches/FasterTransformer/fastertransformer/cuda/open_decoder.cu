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
__global__ void transpose_cache_batch_major(T* k_dst,
                                            T* v_dst,
                                            const T* k_src,
                                            const T* v_src,
                                            const int* memory_seq_len,
                                            const int head_num,
                                            const int size_per_head,
                                            const int memory_max_seq_len,
                                            const int cache_max_len) {
  const int hidden_dim = head_num * size_per_head;
  const int x = (sizeof(T) == 4) ? 4 : 8;
  const int size_per_head_split = size_per_head / x;
  const int batch_id = blockIdx.x;
  const int seq_id = blockIdx.y;

  for (int id = threadIdx.x; id < head_num * size_per_head_split * x;
       id += blockDim.x) {
    int tmp_id = id;
    int x_id = tmp_id % x;
    tmp_id /= x;
    int size_id = tmp_id % size_per_head_split;
    tmp_id /= size_per_head_split;
    int head_id = tmp_id % head_num;

    int src_seq_id =
        (seq_id < memory_seq_len[batch_id])
            ? (seq_id + memory_max_seq_len - memory_seq_len[batch_id])
            : (seq_id - memory_seq_len[batch_id]);

    // key: [B, head_num, L, size_per_head / x, x] ->
    // [B, head_num, size_per_head / x, L, x]
    k_dst[batch_id * hidden_dim * cache_max_len +
          head_id * size_per_head * cache_max_len +
          size_id * cache_max_len * x + seq_id * x + x_id] =
        k_src[batch_id * hidden_dim * memory_max_seq_len +
              head_id * size_per_head * memory_max_seq_len +
              src_seq_id * size_per_head + size_id * x + x_id];

    // value: [B, head_num, L, size_per_head/x, x] ->
    // [B, head_num, L, size_per_head/x, x]
    v_dst[batch_id * hidden_dim * cache_max_len +
          head_id * size_per_head * cache_max_len + seq_id * size_per_head +
          size_id * x + x_id] =
        v_src[batch_id * hidden_dim * memory_max_seq_len +
              head_id * size_per_head * memory_max_seq_len +
              src_seq_id * size_per_head + size_id * x + x_id];
  }
}

template <typename T>
void transpose_cache_batch_major_kernelLauncher(T* k_dst,
                                                T* v_dst,
                                                const T* k_src,
                                                const T* v_src,
                                                const int* memory_seq_len,
                                                const int local_batch_size,
                                                const int memory_max_seq_len,
                                                const int cache_max_len,
                                                const int size_per_head,
                                                const int local_head_num,
                                                cudaStream_t stream) {
  constexpr int block_sz = 128;
  dim3 grid(local_batch_size, memory_max_seq_len);

  transpose_cache_batch_major<<<grid, block_sz, 0, stream>>>(k_dst,
                                                             v_dst,
                                                             k_src,
                                                             v_src,
                                                             memory_seq_len,
                                                             local_head_num,
                                                             size_per_head,
                                                             memory_max_seq_len,
                                                             cache_max_len);
}

template <typename T>
void transpose_general_kernelLauncher(T* dst,
                                      T* src,
                                      const int batch_size,
                                      const int seq_len,
                                      const int head_num,
                                      const int size_per_head,
                                      cudaStream_t stream) {
  dim3 grid, block;
  int grid_size = batch_size * head_num * seq_len;
  if (sizeof(T) == 2) {
    int seq_per_block = grid_size % 4 == 0 ? 4 : 1;
    grid.x = grid_size / seq_per_block;
    block.x = seq_per_block * size_per_head / 2;
    transpose<T><<<grid, block, 0, stream>>>(
        src, dst, batch_size, seq_len, head_num, size_per_head / 2);
  } else {
    const int seq_per_block = 1;
    grid.x = grid_size / seq_per_block;
    block.x = seq_per_block * size_per_head;
    transpose<T><<<grid, block, 0, stream>>>(
        src, dst, batch_size, seq_len, head_num, size_per_head);
  }
}

template void transpose_cache_batch_major_kernelLauncher(
    float* k_dst,
    float* v_dst,
    const float* k_src,
    const float* v_src,
    const int* memory_seq_len,
    const int local_batch_size,
    const int memory_max_seq_len,
    const int cache_max_len,
    const int size_per_head,
    const int local_head_num,
    cudaStream_t stream);

template void transpose_cache_batch_major_kernelLauncher(
    half* k_dst,
    half* v_dst,
    const half* k_src,
    const half* v_src,
    const int* memory_seq_len,
    const int local_batch_size,
    const int memory_max_seq_len,
    const int cache_max_len,
    const int size_per_head,
    const int local_head_num,
    cudaStream_t stream);

template void transpose_general_kernelLauncher(float* dst,
                                               float* src,
                                               const int batch_size,
                                               const int seq_len,
                                               const int head_num,
                                               const int size_per_head,
                                               cudaStream_t stream);

template void transpose_general_kernelLauncher(half* dst,
                                               half* src,
                                               const int batch_size,
                                               const int seq_len,
                                               const int head_num,
                                               const int size_per_head,
                                               cudaStream_t stream);



template <typename T>
void fusedQKV_masked_attention_dispatch_v2(
  const T* qkv_buf, const T* qkv_bias,
  T* key_cache, T* value_cache,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size, 
  int head_num, int size_per_head, const int step, const int max_seq_len, 
  const int max_input_len, const int* input_lengths, const int rotary_embedding_dim, cudaStream_t stream)
{
  using DataType = typename std::conditional<sizeof(T) == 4, float, uint16_t>::type;
  // Prepare the parameters.
  Masked_multihead_attention_params<DataType> params;
  memset(&params, 0, sizeof(params));
  int hidden_units = head_num * size_per_head;
  if (qkv_bias != nullptr) {
      params.q_bias = reinterpret_cast<const DataType*>(qkv_bias);
      params.k_bias = reinterpret_cast<const DataType*>(qkv_bias) + hidden_units;
      params.v_bias = reinterpret_cast<const DataType*>(qkv_bias) + 2 * hidden_units;
  }
  else {
     // gptj/codegen no bias
      params.q_bias = nullptr;
      params.k_bias = nullptr;
      params.v_bias = nullptr;
  }

  // Set the output buffer.
  params.out = reinterpret_cast<DataType *>(context_buf);

  // Set the input buffers.
  params.q = reinterpret_cast<const DataType *>(qkv_buf);
  params.k = reinterpret_cast<const DataType *>(qkv_buf) + hidden_units;
  params.v = reinterpret_cast<const DataType *>(qkv_buf) + 2 * hidden_units;
  params.stride = 3 * hidden_units;
  params.finished = const_cast<bool*>(finished);

  params.k_cache = reinterpret_cast<DataType *>(key_cache);
  params.v_cache = reinterpret_cast<DataType *>(value_cache);
  params.batch_size = inference_batch_size;
  params.seq_length = max_seq_len;
  params.timestep = step-1;
  params.num_heads = head_num;
  params.hidden_size_per_head = size_per_head;
  // GptJ: rotary_embedding
  params.rotary_embedding_dim = rotary_embedding_dim;
  params.inv_sqrt_dh = 1.F / sqrtf((float) params.hidden_size_per_head);

  params.is_mask = true;
  params.input_lengths = input_lengths;
  params.max_input_len = max_input_len;

  masked_multihead_attention(params, stream);
}

template <typename T>
void masked_attention_dispatch_v2(
  T* key_buf, T* value_buf,
  T* query_buf, const T* self_Q_bias, 
  T* key_cache, const T* self_K_bias, T* value_cache, const T* self_V_bias,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size,
  int head_num, int size_per_head, const int step, const int max_seq_len, cudaStream_t stream,
  const T* relative_attention_bias)
{
  if (max_seq_len < 0) {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));
  
    dim3 grid(inference_batch_size * head_num);
  
    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        masked_attention_kernel_opt<32, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache, self_V_bias, context_buf, finished,
          max_batch_size, head_num, step, scalar);
        break;
      case 64:
          masked_attention_kernel_opt<64, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
            key_buf, value_buf,
            query_buf, self_Q_bias,
            key_cache, self_K_bias,
            value_cache, self_V_bias,
            context_buf, 
            finished,
            max_batch_size, head_num, step, scalar);
        break;
      case 128:
          masked_attention_kernel_opt<128, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
            key_buf, value_buf,
            query_buf, self_Q_bias,  key_cache, self_K_bias, value_cache, self_V_bias, context_buf, finished,
            max_batch_size, head_num, step, scalar);
        break;
      default:
        // default path
        int block_size = 128;
        
        //suppose size_per_head <= 128
        if(step <= 64)
          block_size = 64;
        else if(step <= 128 && step > size_per_head)
          block_size = 128;
        else if(step > 128 && step <= 256)
          block_size = 256;
        else if(step > 256 && step <= 512)
          block_size = 512;
        else
          block_size = 1024;
        
        if((int)block_size < size_per_head)
          block_size = size_per_head;
          
        assert(block_size <= 1024);
        dim3 block(block_size);
        T scalar = 1 / sqrtf(size_per_head * 1.0f);
  
        
        int shared_size = sizeof(T) * (size_per_head + step);
        masked_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          key_buf, value_buf,
          query_buf, self_Q_bias,
          key_cache, self_K_bias,
          value_cache, self_V_bias,
          context_buf, finished, max_batch_size,
          head_num, size_per_head, step, scalar);
    }
  } else {
    assert(step > 0);
    assert(size_per_head == 32 || size_per_head == 64 || size_per_head == 128);
    using DataType = typename std::conditional<sizeof(T) == 4, float, uint16_t>::type;
    // Prepare the parameters.
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));
    params.q_bias = reinterpret_cast<const DataType *>(self_Q_bias);
    params.k_bias = reinterpret_cast<const DataType *>(self_K_bias);
    params.v_bias = reinterpret_cast<const DataType *>(self_V_bias);
  
    // Set the output buffer.
    params.out = reinterpret_cast<DataType *>(context_buf);
  
    // Set the input buffers.
    params.q = reinterpret_cast<const DataType *>(query_buf);
    params.k = reinterpret_cast<const DataType *>(key_buf);
    params.v = reinterpret_cast<const DataType *>(value_buf);
    params.stride = 0;
    params.finished = const_cast<bool*>(finished);
  
    params.k_cache = reinterpret_cast<DataType *>(key_cache);
    params.v_cache = reinterpret_cast<DataType *>(value_cache);
    params.batch_size = inference_batch_size;
    params.seq_length = max_seq_len;
    params.timestep = step-1;
    params.num_heads = head_num;
    params.hidden_size_per_head = size_per_head;

    params.is_mask = false;
    params.input_lengths = nullptr;
    params.max_input_len = 0;

    if (relative_attention_bias) {
      params.inv_sqrt_dh = 1.F;

      if (sizeof(T) == 4) {
        params.relative_attention_bias_float = reinterpret_cast<const float*>(relative_attention_bias);
      } else {
        params.relative_attention_bias_half = reinterpret_cast<const half*>(relative_attention_bias);
      }

      params.relative_attention_bias_stride = max_seq_len + 1;
    } else {
      params.inv_sqrt_dh = 1.F / sqrtf((float) params.hidden_size_per_head);
    }

    masked_multihead_attention(params, stream);
  }
}

template <typename T>
void fusedQKV_masked_attention_dispatch_v3(
  const T* qkv_buf, const T* qkv_bias,
  T* key_cache, T* value_cache,
  T* context_buf, const bool* finished, int max_batch_size, int inference_batch_size, 
  int head_num, int size_per_head, const int step, const int max_seq_len, cudaStream_t stream,
  const T* relative_attention_bias)
{
  if (max_seq_len < 0) {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    T scalar = (T)(1.f / sqrtf(size_per_head * 1.0f));
  
    dim3 grid(inference_batch_size * head_num);
  
    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        fusedQKV_masked_attention_kernel_opt<32, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
          qkv_buf, qkv_bias,
          key_cache, value_cache,
          context_buf,
          finished,
          max_batch_size, head_num, step, scalar);
        break;
      case 64:
        fusedQKV_masked_attention_kernel_opt<64, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
          qkv_buf, qkv_bias,
          key_cache,
          value_cache,
          context_buf,
          finished,
          max_batch_size, head_num, step, scalar);
        break;
      case 128:
        fusedQKV_masked_attention_kernel_opt<128, block_sz, T><<<grid, block_sz, sizeof(float)*step, stream>>>(
          qkv_buf, qkv_bias,
          key_cache,
          value_cache,
          context_buf,
          finished,
          max_batch_size, head_num, step, scalar);
        break;
      default:
        assert(false);
    }
  }
  else {
    using DataType = typename std::conditional<sizeof(T) == 4, float, uint16_t>::type;
    // Prepare the parameters.
    Masked_multihead_attention_params<DataType> params;
    memset(&params, 0, sizeof(params));
    int hidden_units = head_num * size_per_head;
    params.q_bias = reinterpret_cast<const DataType *>(qkv_bias);
    params.k_bias = reinterpret_cast<const DataType *>(qkv_bias) + hidden_units;
    params.v_bias = reinterpret_cast<const DataType *>(qkv_bias) + 2 * hidden_units;
  
    // Set the output buffer.
    params.out = reinterpret_cast<DataType *>(context_buf);
  
    // Set the input buffers.
    params.q = reinterpret_cast<const DataType *>(qkv_buf);
    params.k = reinterpret_cast<const DataType *>(qkv_buf) + hidden_units;
    params.v = reinterpret_cast<const DataType *>(qkv_buf) + 2 * hidden_units;
    params.stride = 3 * hidden_units;
    params.finished = const_cast<bool*>(finished);
  
    params.k_cache = reinterpret_cast<DataType *>(key_cache);
    params.v_cache = reinterpret_cast<DataType *>(value_cache);
    params.batch_size = inference_batch_size;
    params.seq_length = max_seq_len;
    params.timestep = step - 1;
    params.num_heads = head_num;
    params.hidden_size_per_head = size_per_head;

    params.is_mask = false;
    params.input_lengths = nullptr;
    params.max_input_len = 0;

    if (relative_attention_bias) {
      params.inv_sqrt_dh = 1.F;

      if (sizeof(T) == 4) {
        params.relative_attention_bias_float = reinterpret_cast<const float*>(relative_attention_bias);
      } else {
        params.relative_attention_bias_half = reinterpret_cast<const half*>(relative_attention_bias);
      }

      params.relative_attention_bias_stride = max_seq_len + 1;
    } else {
      params.inv_sqrt_dh = 1.F / sqrtf((float) params.hidden_size_per_head);
    }

    masked_multihead_attention(params, stream);
  }
}

template void fusedQKV_masked_attention_dispatch_v3(
  const float* qkv_buf, 
  const float* qkv_bias,
  float* key_cache, 
  float* value_cache,
  float* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head, 
  const int step, 
  const int max_seq_len,
  cudaStream_t stream,
  const float* relative_attention_bias);
  
template void fusedQKV_masked_attention_dispatch_v3(
  const half* qkv_buf, 
  const half* qkv_bias,
  half* key_cache, 
  half* value_cache,
  half* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head,
  const int step, 
  const int max_seq_len,
  cudaStream_t stream,
  const half* relative_attention_bias);

template void masked_attention_dispatch_v2(
  float* key_buf, 
  float* value_buf,
  float* query_buf, 
  const float* self_Q_bias, 
  float* key_cache, 
  const float* self_K_bias, 
  float* value_cache, 
  const float* self_V_bias,
  float* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head, 
  const int step,
  const int max_seq_size,
  cudaStream_t stream,
  const float* relative_attention_bias);

template void masked_attention_dispatch_v2(
  half* key_buf, 
  half* value_buf,
  half* query_buf, 
  const half* self_Q_bias, 
  half* key_cache, 
  const half* self_K_bias, 
  half* value_cache, 
  const half* self_V_bias,
  half* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head, 
  const int step,
  const int max_seq_size,
  cudaStream_t stream,
  const half* relative_attention_bias);

template void fusedQKV_masked_attention_dispatch_v2(
  const float* qkv_buf, 
  const float* qkv_bias,
  float* key_cache, 
  float* value_cache,
  float* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head, 
  const int step, 
  const int max_seq_len,
  const int max_input_len, 
  const int* input_lengths,
  const int rotary_embedding_dim,
  cudaStream_t stream);
  
template void fusedQKV_masked_attention_dispatch_v2(
  const half* qkv_buf, 
  const half* qkv_bias,
  half* key_cache, 
  half* value_cache,
  half* context_buf, 
  const bool* finished, 
  int max_batch_size, 
  int inference_batch_size, 
  int head_num, 
  int size_per_head,
  const int step, 
  const int max_seq_len,
  const int max_input_len, 
  const int* input_lengths,
  const int rotary_embedding_dim,
  cudaStream_t stream);

template <typename T>
void cross_attention_dispatch_v2(T* query_buf, const T* Q_bias, 
  T* key_cache, const T* K_bias, T* value_cache, const T* V_bias, const int* length,
  T* context_buf, const bool* finished,
  int batch_size, int head_num, int size_per_head, int step, int seq_len, cudaStream_t stream,
  const T* relative_attention_bias)
  {
    const int block_sz = ATTENTION_BLOCK_SIZE;
    float scalar = (relative_attention_bias) ? 1.0f : 1.f / sqrtf(size_per_head * 1.0f);

    dim3 grid(batch_size * head_num);

    int cond = size_per_head * ((ATTENION_OPT)? 1:0);
    switch (cond)
    {
      case 32:
        cross_attention_kernel_opt<T, 32, block_sz><<<grid, block_sz, sizeof(float)*seq_len, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf, finished,
          batch_size, head_num, step, seq_len, scalar);
        break;
      case 64:
        cross_attention_kernel_opt<T, 64, block_sz><<<grid, block_sz, sizeof(float)*seq_len, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf, finished,
          batch_size, head_num, step, seq_len, scalar);
        break;
      case 128:
        cross_attention_kernel_opt<T, 128, block_sz><<<grid, block_sz, sizeof(float)*seq_len, stream>>>(
          query_buf, Q_bias, key_cache, K_bias, value_cache, V_bias, length, context_buf, finished,
          batch_size, head_num, step, seq_len, scalar);
        break;
      default:
        // default path

        int block_size = 128;

        if(seq_len <= 64)
          block_size = 64;
        else if(seq_len <= 128 && seq_len > size_per_head)
          block_size = 128;
        else if(seq_len > 128 && seq_len <= 256)
          block_size = 256;
        else if(seq_len > 256 && seq_len <= 512)
          block_size = 512;
        else
          block_size = 1024;

        if(block_size < size_per_head)
          block_size = size_per_head;

        assert(block_size <= 1024);
        dim3 block(block_size);
        
        int shared_size = sizeof(T) * (size_per_head + seq_len);
        cross_attention_kernel<T><<<grid, block, shared_size, stream>>>(
          query_buf, Q_bias, 
          key_cache, K_bias,
          value_cache, V_bias,
          length, context_buf, finished,
          batch_size,
          head_num, size_per_head, step, seq_len, scalar);
    }
  }

template void cross_attention_dispatch_v2(
  float* query_buf, 
  const float* Q_bias, 
  float* key_cache, 
  const float* K_bias, 
  float* value_cache, 
  const float* V_bias, 
  const int* length,
  float* context_buf, 
  const bool* finished,
  int batch_size, 
  int head_num, 
  int size_per_head, 
  int step, 
  int seq_len, 
  cudaStream_t stream,
  const float* relative_attention_bias);

template void cross_attention_dispatch_v2(
  half* query_buf, 
  const half* Q_bias, 
  half* key_cache, 
  const half* K_bias, 
  half* value_cache, 
  const half* V_bias, 
  const int* length,
  half* context_buf, 
  const bool* finished,
  int batch_size, 
  int head_num, 
  int size_per_head, 
  int step, 
  int seq_len, 
  cudaStream_t stream,
  const half* relative_attention_bias);

}
