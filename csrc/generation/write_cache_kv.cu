// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
__global__ void write_cache_k_kernel(T *cache_k,
                                     const T *k,
                                     const int *seq_lens,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int len = seq_lens ? seq_lens[bi] : seq_len;
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto k_src = reinterpret_cast<const uint4 *>(
      k + bi * num_head * seq_len * dim_head + hi * seq_len * dim_head);
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  auto k_dst = reinterpret_cast<uint4 *>(
      cache_k + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // vec size
  int dim_head_div_x = dim_head / X_ELEMS;

  // FIXME(wangxi): num_head is not need?
  // if (out_idx >= num_head * dim_head_div_x * max_seq_len) return;
  if (out_idx >= dim_head_div_x * max_seq_len) return;

  int idx = out_idx;
  const int k_seq_len_id = idx % max_seq_len;
  // idx = (idx - k_seq_len_id) / max_seq_len;
  idx = idx / max_seq_len;
  const int k_vec_id = idx % dim_head_div_x;

  if (k_seq_len_id < len) {
    k_dst[out_idx] = k_src[k_seq_len_id * dim_head_div_x + k_vec_id];
  }
}

template <typename T>
__global__ void write_cache_v_kernel(T *cache_v,
                                     const T *v,
                                     const int *seq_lens,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int len = seq_lens ? seq_lens[bi] : seq_len;
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto v_src = reinterpret_cast<const uint4 *>(
      v + bi * num_head * seq_len * dim_head + hi * seq_len * dim_head);
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  auto v_dst = reinterpret_cast<uint4 *>(
      cache_v + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);
  const int dim_head_div_x = dim_head / X_ELEMS;

  if (idx >= dim_head_div_x * len) return;

  v_dst[idx] = v_src[idx];
}

template <paddle::DataType D>
void LaunchWriteCacheKV(const paddle::Tensor& input_k, 
                        const paddle::Tensor& input_v, 
                        const paddle::Tensor& cache_kv,
                        const paddle::Tensor& sequence_lengths) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int64_t bsz = input_k.shape()[0];
    const int64_t seq_len = input_k.shape()[2]; 
    const int64_t cache_bsz = cache_kv.shape()[1]; 
    const int64_t num_head = cache_kv.shape()[2]; 
    const int64_t dim_head = cache_kv.shape()[4]; 
    // printf("bsz: %d, cache_bsz: %d, num_head: %d, seq_len: %d, dim_head: %d.\n", bsz, cache_bsz, num_head, seq_len, dim_head);

    const DataType_ *k_ptr = reinterpret_cast<const DataType_*>(input_k.data<data_t>());
    const DataType_ *v_ptr = reinterpret_cast<const DataType_*>(input_v.data<data_t>());

    // [2, bsz, num_head, max_seq_len, head_dim]
    int max_seq_len = cache_kv.shape()[3];
    DataType_ *cache_kv_data = reinterpret_cast<DataType_*>(const_cast<data_t*>(cache_kv.data<data_t>()));

    int64_t cache_k_size = cache_bsz * num_head * max_seq_len * dim_head;

    DataType_ *cache_k_ptr = cache_kv_data;
    DataType_ *cache_v_ptr = cache_kv_data + cache_k_size;

    constexpr int block_sz = 128;
    constexpr int x = VEC_16B / sizeof(DataType_);

    assert(dim_head % x == 0);
    // PD_CHECK((dim_head % x) == 0, "PD_CHECK returns ", false, ", dim_head must be divisible by vec_size.");

    int max_size = max_seq_len * dim_head / x;
    int size = seq_len * dim_head / x;
    dim3 grid(div_up(max_size, block_sz), bsz, num_head);
    dim3 grid_v(div_up(size, block_sz), bsz, num_head);

    // transpose [bsz, num_head, seq_len, dim_head/x, x]->
    // [bsz, num_head, dim_head/x, max_seq_len, x]
    write_cache_k_kernel<<<grid, block_sz, 0, input_k.stream()>>>(
        cache_k_ptr, k_ptr, sequence_lengths.data<int>(), num_head, dim_head, seq_len, max_seq_len);

    // copy [bsz, num_head, seq_len, dim_head/x, x]->
    // [bsz, num_head, max_seq_len, dim_head/x, x]
    write_cache_v_kernel<<<grid_v, block_sz, 0, input_k.stream()>>>(
        cache_v_ptr, v_ptr, sequence_lengths.data<int>(), num_head, dim_head, seq_len, max_seq_len);
}

void WriteCacheKV(const paddle::Tensor& input_k,
                  const paddle::Tensor& input_v,
                  const paddle::Tensor& cache_kv,
                  const paddle::Tensor& sequence_lengths_shape) {
    switch (cache_kv.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchWriteCacheKV<paddle::DataType::BFLOAT16>(
                input_k, input_v, cache_kv, sequence_lengths_shape
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchWriteCacheKV<paddle::DataType::FLOAT16>(
                input_k, input_v, cache_kv, sequence_lengths_shape
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchWriteCacheKV<paddle::DataType::FLOAT32>(
                input_k, input_v, cache_kv, sequence_lengths_shape
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

PD_BUILD_OP(write_cache_kv)
    .Inputs({"input_k", "input_v", "cache_kv", "sequence_lengths"})
    .Outputs({"cache_kv_out"})
    .SetInplaceMap({{"cache_kv", "cache_kv_out"}})
    .SetKernelFn(PD_KERNEL(WriteCacheKV));