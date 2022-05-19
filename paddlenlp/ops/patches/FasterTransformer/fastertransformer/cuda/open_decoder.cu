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
}
