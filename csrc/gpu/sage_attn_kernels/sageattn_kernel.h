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

#include "sageattn_fused.cuh"
#include "sageattn_fused_varlen.cuh"

template <typename T, typename OutT>
void SageAttentionKernel(
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
    const paddle::Tensor& cu_seqlen,
    const paddle::Tensor& cu_seqlen_v_padded,
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
    const int max_enc_len_this_time_data,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out) {

    int batch_size = cu_seqlen.shape()[0] - 1;

    const int num_q_head = meta_data.q_num_heads;
    const int head_dim_qk = meta_data.head_dims;

    const int num_kv_head = meta_data.kv_num_heads;
    const int head_dim_v = meta_data.head_dims_v;

    std::vector<paddle::Tensor>&& qkv_with_rope = paddle::split(qkv, {num_q_head * head_dim_qk, num_kv_head * head_dim_qk, num_kv_head * head_dim_v}, 1);
    paddle::Tensor q = paddle::reshape(qkv_with_rope[0], {-1, num_q_head, head_dim_qk});
    paddle::Tensor k = paddle::reshape(qkv_with_rope[1], {-1, num_kv_head, head_dim_qk});
    paddle::Tensor v = paddle::reshape(qkv_with_rope[2], {-1, num_kv_head, head_dim_v});

    paddle::Tensor km = chunked_segment_mean_fwd(k, *const_cast<paddle::Tensor*>(&cu_seqlen), max_enc_len_this_time_data)[0];

    // use varlen API
    paddle::optional<paddle::Tensor> vm = paddle::optional<paddle::Tensor>(paddle::empty({1}, paddle::DataType::FLOAT32, paddle::GPUPlace()));

    *out = sage_attention_varlen_fwd(q, 
                                    k, 
                                    v, 
                                    *const_cast<paddle::Tensor*>(&cu_seqlen),
                                    *const_cast<paddle::Tensor*>(&cu_seqlen_v_padded),
                                    seq_lens_encoder,
                                    km, 
                                    vm, 
                                    max_enc_len_this_time_data, // max_seqlen_q
                                    max_enc_len_this_time_data, // max_seqlen_k
                                    softmax_scale, 
                                    std::string("per_warp"), 
                                    std::string("any"), 
                                    0, 
                                    causal, 
                                    true, 
                                    false, 
                                    false)[0];
    *out = paddle::reshape(*out, {-1, num_q_head * head_dim_qk});
}