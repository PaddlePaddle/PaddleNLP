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
#include "append_attention_impl.cuh"
#include "utils.cuh"
// #define DEBUG_DEC_ATTN

template <typename T, typename OutT>
void CascadeAppendAttentionKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,  // [token_num, num_heads, head_dim]
    const paddle::Tensor&
        cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const paddle::Tensor&
        cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>&
        cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const std::string& cache_quant_type_str,
    const int num_blocks,
    const int block_shape_q,
    const int max_seq_len,
    const int max_dec_len,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  const auto token_num = meta_data.token_nums;
  const auto block_size = meta_data.block_size;
  const auto bsz = meta_data.batch_size;
  const auto num_heads = meta_data.q_num_heads;
  const auto group_size = meta_data.q_num_heads / meta_data.kv_num_heads;
  const auto head_dim = meta_data.head_dims;

  if (cache_quant_type_str == "none") {
    DISPATCH_CAUSAL(
        causal,
        CAUSAL,
        {DISPATCH_ENABLE_PREFILL(
            enable_prefill,
            ENABLE_PREFILL,
            {DISPATCH_GQA_GROUP_SIZE(
                group_size,
                GROUP_SIZE,
                {DISPATCH_HEAD_DIM(
                    head_dim,
                    HEAD_DIM,
                    {DISPATCH_BLOCK_SIZE(
                        block_size,
                        BLOCK_SIZE,
                        {DISPATCH_BLOCKSHAPE_Q(
                            block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, {
                              MultiQueryAppendAttention<T,
                                                        GROUP_SIZE,
                                                        HEAD_DIM,
                                                        BLOCK_SIZE,
                                                        CAUSAL,
                                                        BLOCK_SHAPE_Q,
                                                        NUM_WARP_Q,
                                                        OutT,
                                                        ENABLE_PREFILL>(
                                  meta_data,
                                  qkv,
                                  cache_k,
                                  cache_v,
                                  attn_mask,
                                  shift_bias,
                                  smooth_weight,
                                  seq_lens_q,
                                  seq_lens_kv,
                                  seq_lens_encoder,
                                  padding_offsets,
                                  cum_offsets,
                                  block_table,
                                  batch_ids,
                                  tile_ids_per_batch,
                                  num_blocks,
                                  max_seq_len,
                                  max_dec_len,
                                  in_scale,
                                  max_partition_size,
                                  encoder_max_partition_size,
                                  speculate_max_draft_token_num,
                                  is_decoder,
                                  stream,
                                  out);
                            })})})})})})
  } else if (cache_quant_type_str == "cache_int8") {
    DISPATCH_CAUSAL(
        causal,
        CAUSAL,
        {DISPATCH_ENABLE_PREFILL(
            enable_prefill,
            ENABLE_PREFILL,
            {DISPATCH_GQA_GROUP_SIZE(
                group_size,
                GROUP_SIZE,
                {DISPATCH_HEAD_DIM(
                    head_dim,
                    HEAD_DIM,
                    {DISPATCH_BLOCK_SIZE(
                        block_size,
                        BLOCK_SIZE,
                        {DISPATCH_BLOCKSHAPE_Q(
                            block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, {
                              MultiQueryAppendC8Attention<T,
                                                          GROUP_SIZE,
                                                          HEAD_DIM,
                                                          BLOCK_SIZE,
                                                          CAUSAL,
                                                          BLOCK_SHAPE_Q,
                                                          NUM_WARP_Q,
                                                          OutT,
                                                          ENABLE_PREFILL>(
                                  meta_data,
                                  qkv,
                                  cache_k,
                                  cache_v,
                                  attn_mask,
                                  cache_k_scale.get(),
                                  cache_v_scale.get(),
                                  shift_bias,
                                  smooth_weight,
                                  seq_lens_q,
                                  seq_lens_kv,
                                  seq_lens_encoder,
                                  padding_offsets,
                                  cum_offsets,
                                  block_table,
                                  batch_ids,
                                  tile_ids_per_batch,
                                  num_blocks,
                                  max_seq_len,
                                  max_dec_len,
                                  in_scale,
                                  max_partition_size,
                                  encoder_max_partition_size,
                                  speculate_max_draft_token_num,
                                  is_decoder,
                                  stream,
                                  out);
                            })})})})})})
  } else if (cache_quant_type_str == "cache_int4") {
    DISPATCH_CAUSAL(
        causal,
        CAUSAL,
        {DISPATCH_ENABLE_PREFILL(
            enable_prefill,
            ENABLE_PREFILL,
            {DISPATCH_GQA_GROUP_SIZE(
                group_size,
                GROUP_SIZE,
                {DISPATCH_HEAD_DIM(
                    head_dim,
                    HEAD_DIM,
                    {DISPATCH_BLOCK_SIZE(
                        block_size,
                        BLOCK_SIZE,
                        {DISPATCH_BLOCKSHAPE_Q(
                            block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, {
                              MultiQueryAppendC4Attention<T,
                                                          GROUP_SIZE,
                                                          HEAD_DIM,
                                                          BLOCK_SIZE,
                                                          CAUSAL,
                                                          BLOCK_SHAPE_Q,
                                                          NUM_WARP_Q,
                                                          OutT,
                                                          ENABLE_PREFILL>(
                                  meta_data,
                                  qkv,
                                  cache_k,
                                  cache_v,
                                  attn_mask,
                                  cache_k_scale.get(),
                                  cache_v_scale.get(),
                                  cache_k_zp,
                                  cache_v_zp,
                                  shift_bias,
                                  smooth_weight,
                                  seq_lens_q,
                                  seq_lens_kv,
                                  seq_lens_encoder,
                                  padding_offsets,
                                  cum_offsets,
                                  block_table,
                                  batch_ids,
                                  tile_ids_per_batch,
                                  num_blocks,
                                  max_seq_len,
                                  max_dec_len,
                                  in_scale,
                                  max_partition_size,
                                  encoder_max_partition_size,
                                  speculate_max_draft_token_num,
                                  is_decoder,
                                  stream,
                                  out);
                            })})})})})})
  } else {
    PD_THROW("append attention just support C16/C8/C4_zp now!");
  }
}