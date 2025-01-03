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

#include "helper.h"
#include "utils.cuh"

template <typename T, typename OutT>
void CascadeAppendAttentionC16Kernel(
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
    const int num_blocks,
    const int block_shape_q,
    const int max_seq_len,
    const int max_dec_len,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out);

template <typename T, typename OutT>
void CascadeAppendAttentionC8Kernel(
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
    const int num_blocks,
    const int block_shape_q,
    const int max_seq_len,
    const int max_dec_len,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out);

template <typename T, typename OutT>
void CascadeAppendAttentionC4Kernel(
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
    const int num_blocks,
    const int block_shape_q,
    const int max_seq_len,
    const int max_dec_len,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out);

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
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  if (cache_quant_type_str == "none") {
    CascadeAppendAttentionC16Kernel<T, OutT>(meta_data,
                                             qkv,
                                             cache_k,
                                             cache_v,
                                             attn_mask,
                                             cache_k_scale,
                                             cache_v_scale,
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
                                             block_shape_q,
                                             max_seq_len,
                                             max_dec_len,
                                             quant_max_bound,
                                             quant_min_bound,
                                             in_scale,
                                             speculate_max_draft_token_num,
                                             causal,
                                             is_decoder,
                                             enable_prefill,
                                             stream,
                                             out);
  } else if (cache_quant_type_str == "cache_int8") {
    CascadeAppendAttentionC8Kernel<T, OutT>(meta_data,
                                            qkv,
                                            cache_k,
                                            cache_v,
                                            attn_mask,
                                            cache_k_scale,
                                            cache_v_scale,
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
                                            block_shape_q,
                                            max_seq_len,
                                            max_dec_len,
                                            quant_max_bound,
                                            quant_min_bound,
                                            in_scale,
                                            speculate_max_draft_token_num,
                                            causal,
                                            is_decoder,
                                            enable_prefill,
                                            stream,
                                            out);
  } else if (cache_quant_type_str == "cache_int4_zp") {
    CascadeAppendAttentionC4Kernel<T, OutT>(meta_data,
                                            qkv,
                                            cache_k,
                                            cache_v,
                                            attn_mask,
                                            cache_k_scale,
                                            cache_v_scale,
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
                                            block_shape_q,
                                            max_seq_len,
                                            max_dec_len,
                                            quant_max_bound,
                                            quant_min_bound,
                                            in_scale,
                                            speculate_max_draft_token_num,
                                            causal,
                                            is_decoder,
                                            enable_prefill,
                                            stream,
                                            out);
  } else {
    PD_THROW(
        "cache_quant_type_str should be one of [none, cache_int8, "
        "cache_int4_zp]");
  }
}

inline uint32_t get_max_partition_size(int bsz) {
    static const char* max_partition_size_env = std::getenv("FLAGS_cascade_attention_max_partition_size");
    static const uint32_t max_partition_size =
            max_partition_size_env == nullptr ? 0 : std::stoul(std::string(max_partition_size_env));
    return (max_partition_size != 0 ? max_partition_size : (bsz == 1 ? 128 : 512));
}
