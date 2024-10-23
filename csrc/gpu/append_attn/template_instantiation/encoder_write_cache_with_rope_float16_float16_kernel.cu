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
#include "../encoder_write_cache_with_rope_kernel.h"

template void EncoderWriteCacheWithRopeKernel<paddle::float16, paddle::float16>(
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
    paddle::Tensor* value_cache_out);