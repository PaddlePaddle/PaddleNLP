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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"
#include <xft/xdnn_plugin.h>

void FusedRotaryPositionEncoding(
    paddle::Tensor& query,  // [num_tokens, num_heads, head_size] or
                            // [num_tokens, num_heads * head_size]
    paddle::Tensor& key,
    // [num_tokens, num_kv_heads, head_size] or [num_tokens, num_kv_heads *
    // head_size]
    const paddle::Tensor& position_ids,   // [num_tokens]
    const paddle::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    int head_size,
    bool is_neox);


std::vector<paddle::Tensor> MlaDeAttn(
    const paddle::Tensor& q,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& decoder_context_len,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& dec_batch_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const float softmax_scale,
    const int block_size,
    const int num_head,
    const int kv_lora_rank, 
    const int rope_head_dim,
    const int dim_qk,
    const int dim_v);


std::vector<paddle::Tensor> MlaEnAttn(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    const paddle::Tensor& v,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& enc_batch_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& max_enc_len_this_time,
    const paddle::Tensor& max_dec_len_this_time,
    const float softmax_scale,
    const int block_size,
    const int num_head,
    const int dim_qk,
    const int dim_v);


std::vector<paddle::Tensor> DecodeMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& decoder_context_len_cache,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& kv_seq_lod_raw,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& kv_seq_lod_raw_cpu,
    const paddle::Tensor& enc_batch_tensor,
    const paddle::Tensor& dec_batch_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len,
    const int kv_num_heads,
    const bool speculate_decoder);

std::vector<paddle::Tensor> PrefillMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& start_token_raw, // encoder cache写时 start token为0 （不算prefix cache）
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& start_token_raw_cpu,
    const paddle::Tensor& enc_batch_tensor,
    const paddle::Tensor& dec_batch_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len,
    const int kv_num_heads);

std::vector<paddle::Tensor> Bmm(const paddle::Tensor& input, 
                                             const paddle::Tensor& weight);
