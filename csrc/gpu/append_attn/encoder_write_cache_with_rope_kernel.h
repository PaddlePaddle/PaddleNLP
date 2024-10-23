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
#pragma once

#include "encoder_write_cache_with_rope_impl.cuh"

template <typename T, typename QKV_TYPE>
void EncoderWriteCacheWithRopeKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // kv_num_heads, head_dim] if GQA)
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const int num_blocks,
    const int max_seq_len,
    const bool use_neox_style,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out) {
  auto token_num = meta_data.token_nums;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto head_dim = meta_data.head_dims;

  if (num_heads == kv_num_heads) {
    rotary_qk_variable(
        qkv_out->data<T>(),
        qkv.data<QKV_TYPE>(),
        qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
        qkv_biases ? qkv_biases.get().data<T>() : nullptr,
        rotary_embs.get().data<float>(),
        padding_offsets.data<int>(),
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        token_num,
        num_heads,
        max_seq_len,
        rotary_embs.get().dims()[2],
        head_dim,
        stream,
        use_neox_style);
  } else {
    gqa_rotary_qk_variable(
        qkv_out->data<T>(),
        qkv.data<QKV_TYPE>(),
        qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
        qkv_biases ? qkv_biases.get().data<T>() : nullptr,
        rotary_embs.get().data<float>(),
        padding_offsets.data<int>(),
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        token_num,
        num_heads,
        kv_num_heads,
        max_seq_len,
        rotary_embs.get().dims()[2],
        head_dim,
        stream,
        use_neox_style);
  }
  const uint32_t block_size = meta_data.block_size;
  if (cache_quant_type_str == "none") {
    CascadeAppendWriteCacheKVQKV<T>(meta_data,
                                    *qkv_out,
                                    block_tables,
                                    padding_offsets,
                                    seq_lens_encoder,
                                    seq_lens_decoder,
                                    max_seq_len,
                                    stream,
                                    key_cache_out,
                                    value_cache_out);
  } else if (cache_quant_type_str == "cache_int8") {
    DISPATCH_HEAD_DIM(
        head_dim, HEAD_DIM, {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
          CascadeAppendWriteCacheKVC8QKV<T, HEAD_DIM, BLOCK_SIZE>(
              meta_data,
              *key_cache_out,
              *value_cache_out,
              *qkv_out,
              cache_k_scale.get(),
              cache_v_scale.get(),
              seq_lens_this_time,
              seq_lens_decoder,
              padding_offsets,
              cum_offsets,
              block_tables,
              batch_ids,
              tile_ids,
              num_blocks,
              max_seq_len,
              stream,
              key_cache_out,
              value_cache_out);
        })})
  } else if (cache_quant_type_str == "cache_int4_zp") {
    DISPATCH_HEAD_DIM(
        head_dim, HEAD_DIM, {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
          CascadeAppendWriteCacheKVC4QKV<T, HEAD_DIM, BLOCK_SIZE>(
              meta_data,
              *key_cache_out,
              *value_cache_out,
              *qkv_out,
              cache_k_scale.get(),
              cache_v_scale.get(),
              cache_k_zp.get(),
              cache_v_zp.get(),
              seq_lens_this_time,
              seq_lens_decoder,
              padding_offsets,
              cum_offsets,
              block_tables,
              batch_ids,
              tile_ids,
              num_blocks,
              max_seq_len,
              stream,
              key_cache_out,
              value_cache_out);
        })})
  } else {
    PD_THROW(
        "cache_quant_type_str should be one of [none, cache_int8, "
        "cache_int4_zp]");
  }
}