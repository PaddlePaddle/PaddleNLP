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

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <type_traits>
#include <vector>
#include "cute/tensor.hpp"
#include "mla_hopper.cuh"
#include "mla_hopper_wg4.cuh"
#include <iostream>
#include <string>
#include <sstream>

#include "batch_mla_with_paged_kv_cache.h"
#include "env.h"

using namespace cute;
using namespace mla_attn;
using namespace std;

template <typename T>
struct cascade_type_traits {
  using type = T;
  using cutlass_type = T;
};
template <>
struct cascade_type_traits<phi::dtype::bfloat16> {
  using type = __nv_bfloat16;
  using cutlass_type = cutlass::bfloat16_t;;
};
template <>
struct cascade_type_traits<phi::dtype::float16> {
  using type = half;
  using cutlass_type = cutlass::half_t;
};
template <>
struct cascade_type_traits<phi::dtype::float8_e4m3fn> {
  using type = __nv_fp8_e4m3;
  using cutlass_type = cutlass::float_e4m3_t;
};

template <typename T>
void BatchMLAWithPagedKVCacheKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& q,  // [token_num, q_head_num, head_dim]
    const paddle::Tensor& latent_cache,  // [max_block_num, q_head_num, block_size, head_dim]
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const paddle::Tensor& num_blocks_x_device,
    const std::string& cache_quant_type_str,
    const int num_blocks_x,
    const int chunk_size,
    const int max_seq_len,
    const int max_dec_len,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int draft_total_token_num,
    const bool causal,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  using NV_TYPE = typename cascade_type_traits<T>::type;
  using CUTLASS_TYPE = typename cascade_type_traits<T>::cutlass_type;
  const auto token_num = meta_data.token_nums;
  const auto block_size = meta_data.block_size;
  const auto bsz = meta_data.batch_size;
  const auto q_head_num = meta_data.q_num_heads;
  const auto max_block_num_per_seq = meta_data.max_blocks_per_seq;
  const auto max_block_num = bsz * max_block_num_per_seq;
  const bool mla_use_wg4 = get_mla_use_wg4();
#if CUDA_VERSION >= 12080
  constexpr bool USE_REG_EALLOC = true;
#else
  constexpr bool USE_REG_EALLOC = false;
#endif
  int q_head_dim = meta_data.head_dims;
  int k_head_dim = meta_data.head_dims;
  int v_head_dim = meta_data.head_dims_v;
  // int num_chunks = max_dec_len / chunk_size;
  int num_chunks = div_up(max_dec_len, chunk_size);

  auto *allocator = paddle::GetAllocator(q.place());
  phi::Allocator::AllocationPtr O_tmp, m_tmp, d_tmp;
  O_tmp = allocator->Allocate(
      phi::SizeOf(q.dtype()) *
      static_cast<size_t>(num_chunks * bsz * draft_total_token_num * q_head_num * v_head_dim));
  m_tmp = allocator->Allocate(
      sizeof(float) *
      static_cast<size_t>(num_chunks * bsz * draft_total_token_num * q_head_num));
  d_tmp = allocator->Allocate(
      sizeof(float) *
      static_cast<size_t>(num_chunks * bsz * draft_total_token_num * q_head_num));

  if (q_head_dim == 576) {
      if (mla_use_wg4 && block_size == 32) {
        if constexpr (!std::is_same<CUTLASS_TYPE, cutlass::half_t>::value) {
          PD_THROW("error!!! mla_wg4 not supports bf16 now!!!\n");
        } else {
          using ParamsType = Params<cutlass::half_t, cutlass::half_t, cutlass::half_t, int> ;
          ParamsType params = {};
          params.Q = reinterpret_cast<cutlass::half_t*>(const_cast<T*>(q.data<T>()));
          params.KV = reinterpret_cast<cutlass::half_t*>(const_cast<T*>(latent_cache.data<T>()));
          params.O = reinterpret_cast<cutlass::half_t*>(const_cast<T*>(out->data<T>()));
          params.O_tmp = reinterpret_cast<cutlass::half_t*>(O_tmp->ptr());
          params.m = reinterpret_cast<float*>(m_tmp->ptr());
          params.d = reinterpret_cast<float*>(d_tmp->ptr());
          params.block_tables = const_cast<int*>(block_tables.data<int>());
          params.seq_lens_this_time = const_cast<int*>(seq_lens_this_time.data<int>());
          params.seq_lens_encoder = const_cast<int*>(seq_lens_encoder.data<int>());
          params.seq_lens_decoder = const_cast<int*>(seq_lens_decoder.data<int>());
          params.cumsum_q_seqlens = const_cast<int*>(cu_seqlens_q.data<int>());
          params.padding_offsets = const_cast<int*>(padding_offsets.data<int>());
          params.batch_ids = const_cast<int*>(batch_ids.data<int>());
          if (q_head_num <=64) {
            params.tile_ids_per_batch = const_cast<int*>(tile_ids_per_batch.data<int>());
          } else {
            params.q_tile_ids_per_batch = const_cast<int*>(tile_ids_per_batch.data<int>());
          }
          params.num_blocks_x = const_cast<int*>(num_blocks_x_device.data<int>());
          params.num_blocks_x_int = num_blocks_x;
          params.q_stride_bsz = q_head_num * q_head_dim;
          params.q_stride_head_num = q_head_dim;
          params.kv_stride_block_num = block_size * k_head_dim;
          params.kv_stride_block_size = k_head_dim;
          params.o_stride_bsz = q_head_num * v_head_dim;
          params.o_stride_head_num = v_head_dim;
          params.bsz = bsz;
          params.token_num = token_num;
          params.max_seq_len = max_seq_len;
          params.max_block_num = max_block_num;
          params.max_block_num_per_seq = max_block_num_per_seq;
          params.q_num_head = q_head_num;
          params.qk_head_dim = q_head_dim;
          params.vo_head_dim = v_head_dim;
          params.block_size = block_size;
          params.draft_total_token_num = draft_total_token_num;
          params.sm_scale = softmax_scale;
          params.chunk_size = chunk_size;
          params.chunk_num = num_chunks;
          BatchMLAWithPagedKVCacheWG4Dispatched<576, 512, half, ParamsType, USE_REG_EALLOC>(
              params, stream
          );
        }
      } else {
        using ParamsType = Params<CUTLASS_TYPE, CUTLASS_TYPE, CUTLASS_TYPE, int>;
        ParamsType params = {};
        params.Q = reinterpret_cast<CUTLASS_TYPE*>(const_cast<T*>(q.data<T>()));
        params.KV = reinterpret_cast<CUTLASS_TYPE*>(const_cast<T*>(latent_cache.data<T>()));
        params.O = reinterpret_cast<CUTLASS_TYPE*>(const_cast<T*>(out->data<T>()));
        params.O_tmp = reinterpret_cast<CUTLASS_TYPE*>(O_tmp->ptr());
        params.m = reinterpret_cast<float*>(m_tmp->ptr());
        params.d = reinterpret_cast<float*>(d_tmp->ptr());
        params.block_tables = const_cast<int*>(block_tables.data<int>());
        params.seq_lens_this_time = const_cast<int*>(seq_lens_this_time.data<int>());
        params.seq_lens_encoder = const_cast<int*>(seq_lens_encoder.data<int>());
        params.seq_lens_decoder = const_cast<int*>(seq_lens_decoder.data<int>());
        params.cumsum_q_seqlens = const_cast<int*>(cu_seqlens_q.data<int>());
        params.padding_offsets = const_cast<int*>(padding_offsets.data<int>());
        params.batch_ids = const_cast<int*>(batch_ids.data<int>());
        if (q_head_num <=64) {
          params.tile_ids_per_batch = const_cast<int*>(tile_ids_per_batch.data<int>());
        } else {
          params.q_tile_ids_per_batch = const_cast<int*>(tile_ids_per_batch.data<int>());
        }
        params.num_blocks_x = const_cast<int*>(num_blocks_x_device.data<int>());
        params.num_blocks_x_int = num_blocks_x;
        params.q_stride_bsz = q_head_num * q_head_dim;
        params.q_stride_head_num = q_head_dim;
        params.kv_stride_block_num = block_size * k_head_dim;
        params.kv_stride_block_size = k_head_dim;
        params.o_stride_bsz = q_head_num * v_head_dim;
        params.o_stride_head_num = v_head_dim;
        params.bsz = bsz;
        params.token_num = token_num;
        params.max_seq_len = max_seq_len;
        params.max_block_num = max_block_num;
        params.max_block_num_per_seq = max_block_num_per_seq;
        params.q_num_head = q_head_num;
        params.qk_head_dim = q_head_dim;
        params.vo_head_dim = v_head_dim;
        params.block_size = block_size;
        params.draft_total_token_num = draft_total_token_num;
        params.sm_scale = softmax_scale;
        params.chunk_size = chunk_size;
        params.chunk_num = num_chunks;
        BatchMLAWithPagedKVCacheDispatched<576, 512, NV_TYPE, ParamsType, USE_REG_EALLOC>(
            params, stream
        );
      }
  } else {
      PD_THROW("error!!! q_head_dim must be 576 !!!\n");
  }
}


template void BatchMLAWithPagedKVCacheKernel<paddle::bfloat16>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& q,  // [token_num, q_head_num, head_dim]
    const paddle::Tensor& latent_cache,  // [max_block_num, q_head_num, block_size, head_dim]
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const paddle::Tensor& num_blocks_x_device,
    const std::string& cache_quant_type_str,
    const int num_blocks_x,
    const int chunk_size,
    const int max_seq_len,
    const int max_dec_len,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int draft_total_token_num,
    const bool causal,
    cudaStream_t& stream,
    paddle::Tensor* out);


template void BatchMLAWithPagedKVCacheKernel<paddle::float16>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& q,  // [token_num, q_head_num, head_dim]
    const paddle::Tensor& latent_cache,  // [max_block_num, q_head_num, block_size, head_dim]
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const paddle::Tensor& num_blocks_x_device,
    const std::string& cache_quant_type_str,
    const int num_blocks_x,
    const int chunk_size,
    const int max_seq_len,
    const int max_dec_len,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int draft_total_token_num,
    const bool causal,
    cudaStream_t& stream,
    paddle::Tensor* out);
