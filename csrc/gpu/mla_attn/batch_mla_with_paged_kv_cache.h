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

/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "paddle/extension.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/allocator.h"
#include "append_attn/utils.cuh"

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
    paddle::Tensor* out);
