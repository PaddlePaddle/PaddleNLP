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
#include "../append_attention_c8_impl.cuh"

template void CascadeAppendAttentionC8Kernel<paddle::bfloat16, int8_t>(
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