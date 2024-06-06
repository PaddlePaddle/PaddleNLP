// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

template <typename T>
__global__ void erase_cache_k_kernel(T *cache_k,
                                     const int *seq_lens,
                                     const int num_head,
                                     const int dim_head,
                                     const int max_seq_len,
                                     const int erase_token_num) {
  const int bi = blockIdx.y;
  const int len = seq_lens[bi];
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);


  // [bsz, num_head, dim_head/x, max_seq_len, x]
//   AlignedVector<int32_t, 4> k_dst;
//   Load<int32_t, 4>(k_dst.data(), cache_k + bi * num_head * max_seq_len * dim_head + hi * max_seq_len * dim_head);
  auto k_dst = reinterpret_cast<uint4 *>(
      cache_k + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int dim_head_div_x = dim_head / X_ELEMS;
  if (idx >= dim_head_div_x * erase_token_num) return;

  int max_seq_len_mul_x = max_seq_len * X_ELEMS;
  const int dim_head_div_id = idx / erase_token_num;
//   const int idx_div = idx / dim_head_div_x;
  const int erase_offset = len * X_ELEMS;
//   const int idx_mod = idx % erase_token_num % dim_head_div_x;
//   const int k_vec_id = idx_mod * X_ELEMS;
  const int k_vec_id = idx % erase_token_num % dim_head_div_x;

  uint4 zero;
  zero.x = 0;
  zero.y = 0;
  zero.z = 0;
  zero.w = 0;

  const int dst_id = (dim_head_div_id * max_seq_len_mul_x + erase_offset) / X_ELEMS + k_vec_id;
//   printf("idx: %d, dim_head_div_id: %d, dim_head_div_x: %d, k_vec_id: %d, dst_id: %d\n", idx, dim_head_div_id, dim_head_div_x, k_vec_id, dst_id);
  k_dst[dst_id] = zero;
}

template <typename T>
__global__ void erase_cache_v_kernel(T *cache_v,
                                     const int *seq_lens,
                                     const int num_head,
                                     const int dim_head,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int len = seq_lens[bi];
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;

  // [bsz, num_head, max_seq_len, dim_head/x, x]
  auto v_dst = reinterpret_cast<uint4 *>(
      cache_v + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head + len * dim_head);

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);
  const int dim_head_div_x = dim_head / X_ELEMS;

  if (idx >= dim_head_div_x * len) return;
  uint4 zero;
  zero.x = 0;
  zero.y = 0;
  zero.z = 0;
  zero.w = 0;

  v_dst[idx] = zero;
}

template <paddle::DataType D>
void LaunchEraseCacheKV(const paddle::Tensor& cache_kv,
                        const paddle::Tensor& sequence_lengths,
                        const int erase_token_num) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int64_t bsz = cache_kv.shape()[1]; 
    const int64_t num_head = cache_kv.shape()[2]; 
    const int64_t dim_head = cache_kv.shape()[4]; 

    // [2, bsz, num_head, max_seq_len, head_dim]
    int max_seq_len = cache_kv.shape()[3];
    DataType_ *cache_kv_data = reinterpret_cast<DataType_*>(const_cast<data_t*>(cache_kv.data<data_t>()));

    int64_t cache_k_size = bsz * num_head * max_seq_len * dim_head;

    DataType_ *cache_k_ptr = cache_kv_data;
    DataType_ *cache_v_ptr = cache_kv_data + cache_k_size;

    constexpr int block_sz = 128;
    constexpr int x = VEC_16B / sizeof(DataType_);

    assert(dim_head % x == 0);
    // PD_CHECK((dim_head % x) == 0, "PD_CHECK returns ", false, ", dim_head must be divisible by vec_size.");

    // int max_size = max_seq_len * dim_head / x;
    int size = erase_token_num * dim_head / x;
    dim3 grid(div_up(size, block_sz), bsz, num_head);
    dim3 grid_v(div_up(size, block_sz), bsz, num_head);

    // transpose [bsz, num_head, seq_len, dim_head/x, x]->
    // [bsz, num_head, dim_head/x, max_seq_len, x]
    erase_cache_k_kernel<<<grid, block_sz, 0, cache_kv.stream()>>>(
        cache_k_ptr, sequence_lengths.data<int>(), num_head, dim_head, max_seq_len, erase_token_num);

    // copy [bsz, num_head, seq_len, dim_head/x, x]->
    // [bsz, num_head, max_seq_len, dim_head/x, x]
    erase_cache_v_kernel<<<grid_v, block_sz, 0, cache_kv.stream()>>>(
        cache_v_ptr, sequence_lengths.data<int>(), num_head, dim_head, max_seq_len);
}

void EraseCacheKV(const paddle::Tensor& cache_kv,
                  const paddle::Tensor& sequence_lengths_shape,
                  const int erase_token_num) {
    switch (cache_kv.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchEraseCacheKV<paddle::DataType::BFLOAT16>(
                cache_kv, sequence_lengths_shape, erase_token_num
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchEraseCacheKV<paddle::DataType::FLOAT16>(
                cache_kv, sequence_lengths_shape, erase_token_num
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchEraseCacheKV<paddle::DataType::FLOAT32>(
                cache_kv, sequence_lengths_shape, erase_token_num
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
        }
    }
}

PD_BUILD_OP(erase_cache_kv)
    .Inputs({"cache_kv", "sequence_lengths"})
    .Outputs({"cache_kv_out"})
    .Attrs({"erase_token_num: int"})
    .SetInplaceMap({{"cache_kv", "cache_kv_out"}})
    .SetKernelFn(PD_KERNEL(EraseCacheKV));