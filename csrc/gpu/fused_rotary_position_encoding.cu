// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"
#include "paddle/extension.h"

template <typename T, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding_kernel(
    T* __restrict__ arr,
    const T* __restrict__ cos_ptr,
    const T* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  T cos, sin;
  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }

  const T x = arr[x_index];
  const T y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}


template <typename T, bool IS_NEOX>
__global__ void apply_rotary_embedding_kernel(
    T* __restrict__ query,  // [num_tokens, num_heads, head_size]
    T* __restrict__ key,    // [num_tokens, num_kv_heads, head_size]
    const int* __restrict__ position_ids,  // [num_tokens]
    const T* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int pos = position_ids[token_idx];
  const T* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const T* cos_ptr = cache_ptr;
  const T* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding_kernel<T, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding_kernel<T, IS_NEOX>(
        key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
}


void FusedRotaryPositionEncoding(
    paddle::Tensor& query,  // [num_tokens, num_heads, head_size] or
                            // [num_tokens, num_heads * head_size]
    paddle::Tensor& key,
    // [num_tokens, num_kv_heads, head_size] or [num_tokens, num_kv_heads *
    // head_size]
    const paddle::Tensor& position_ids,   // [num_tokens]
    const paddle::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    int head_size,
    bool is_neox) {
  int64_t num_tokens = query.dims()[0];
  int num_heads = query.numel() / num_tokens / head_size;
  int num_kv_heads = key.numel() / num_tokens / head_size;
  int rot_dim = cos_sin_cache.dims()[1];
  int64_t query_stride = num_heads * head_size;
  int64_t key_stride = num_kv_heads * head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      query.dtype(), "apply_rotary_embedding_kernel", [&] {
        if (is_neox) {
          apply_rotary_embedding_kernel<data_t, true>
              <<<grid, block, 0, query.stream()>>>(query.data<data_t>(),
                                                   key.data<data_t>(),
                                                   position_ids.data<int>(),
                                                   cos_sin_cache.data<data_t>(),
                                                   rot_dim,
                                                   query_stride,
                                                   key_stride,
                                                   num_heads,
                                                   num_kv_heads,
                                                   head_size);
        } else {
          apply_rotary_embedding_kernel<data_t, false>
              <<<grid, block, 0, query.stream()>>>(query.data<data_t>(),
                                                   key.data<data_t>(),
                                                   position_ids.data<int>(),
                                                   cos_sin_cache.data<data_t>(),
                                                   rot_dim,
                                                   query_stride,
                                                   key_stride,
                                                   num_heads,
                                                   num_kv_heads,
                                                   head_size);
        }
      });
}

PD_BUILD_OP(fused_rotary_position_encoding)
    .Inputs({"query", "key", "position_ids", "cos_sin_cache"})
    .Outputs({"query_out", "key_out"})
    .Attrs({"head_size: int", "is_neox: bool"})
    .SetInplaceMap({{"query", "query_out"}, {"key", "key_out"}})
    .SetKernelFn(PD_KERNEL(FusedRotaryPositionEncoding));