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

#include "append_attn/append_attention_kernel.h"


template <paddle::DataType D>
void append_attention(
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
    const paddle::optional<paddle::Tensor>& shift_bias,
    const paddle::optional<paddle::Tensor>& smooth_weight,
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const paddle::Tensor& num_blocks_cpu,
    const paddle::Tensor& max_enc_len,
    const paddle::Tensor& max_dec_len,
    paddle::Tensor& out_tmp,
    const std::string& cache_quant_type_str,
    const int max_input_len,
    const int block_shape_q,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const float in_scale,
    const std::string& mode,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  int num_blocks_data = num_blocks_cpu.data<int>()[0];
  int max_enc_len_data = max_enc_len.data<int>()[0];
  int max_dec_len_data = max_dec_len.data<int>()[0];

  auto stream = qkv.stream();
  if (mode == "encoder") {
    if (max_enc_len_data <= 0) return;
    if (in_scale > 0.0) {
      CascadeAppendAttentionKernel<data_t, int8_t>(
          qkv,      // [token_num, num_heads, head_dim]
          cache_k,  // [max_block_num, num_heads, block_size, head_dim]
          cache_v,  // [max_block_num, num_heads, head_dim, block_size]
          attn_mask,
          cache_k_scale,  // [num_kv_heads, head_dim]
          cache_v_scale,  // [num_kv_heads, head_dim]
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
          cache_quant_type_str,  // 可选"none","cache_int8","cache_int4"
          num_blocks_data,
          block_shape_q,
          max_input_len,
          max_enc_len_data,
          num_heads,
          kv_num_heads,
          head_dim,
          in_scale,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          is_decoder,
          enable_prefill,
          stream,
          &out_tmp);
    } else {
      CascadeAppendAttentionKernel<data_t, data_t>(
          qkv,      // [token_num, num_heads, head_dim]
          cache_k,  // [max_block_num, num_heads, block_size, head_dim]
          cache_v,  // [max_block_num, num_heads, head_dim, block_size]
          attn_mask,
          cache_k_scale,  // [num_kv_heads, head_dim]
          cache_v_scale,  // [num_kv_heads, head_dim]
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
          cache_quant_type_str,  // 可选"none","cache_int8","cache_int4"
          num_blocks_data,
          block_shape_q,
          max_input_len,
          max_enc_len_data,
          num_heads,
          kv_num_heads,
          head_dim,
          in_scale,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          is_decoder,
          enable_prefill,
          stream,
          &out_tmp);
    }
  } else if (mode == "decoder") {
    if (max_dec_len_data <= 0) return;
    auto decoder_stream = qkv.stream();
    // cudaEvent_t event;
    // cudaStream_t decoder_stream;

    // cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    // cudaStreamCreate(&decoder_stream);
    if (in_scale > 0.0) {
      CascadeAppendAttentionKernel<data_t, int8_t>(
          qkv,      // [token_num, num_heads, head_dim]
          cache_k,  // [max_block_num, num_heads, block_size, head_dim]
          cache_v,  // [max_block_num, num_heads, head_dim, block_size]
          attn_mask,
          cache_k_scale,  // [num_kv_heads, head_dim]
          cache_v_scale,  // [num_kv_heads, head_dim]
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
          cache_quant_type_str,  // 可选"none","cache_int8","cache_int4"
          num_blocks_data,
          block_shape_q,
          max_input_len,
          max_dec_len_data + 1,
          num_heads,
          kv_num_heads,
          head_dim,
          in_scale,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          is_decoder,
          enable_prefill,
          decoder_stream,
          &out_tmp);
    } else {
      CascadeAppendAttentionKernel<data_t, data_t>(
          qkv,      // [token_num, num_heads, head_dim]
          cache_k,  // [max_block_num, num_heads, block_size, head_dim]
          cache_v,  // [max_block_num, num_heads, head_dim, block_size]
          attn_mask,
          cache_k_scale,  // [num_kv_heads, head_dim]
          cache_v_scale,  // [num_kv_heads, head_dim]
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
          cache_quant_type_str,  // 可选"none","cache_int8","cache_int4"
          num_blocks_data,
          block_shape_q,
          max_input_len,
          max_dec_len_data + 1,
          num_heads,
          kv_num_heads,
          head_dim,
          in_scale,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          is_decoder,
          enable_prefill,
          decoder_stream,
          &out_tmp);
    }
    // cudaEventRecord(event, decoder_stream);
    // cudaStreamWaitEvent(stream, event);
  } else {
    PD_THROW("mode must be encoder or decoder.");
  }
}

void AppendAttention(
    const paddle::Tensor& qkv,  // [token_num, num_heads, head_dim]
    const paddle::Tensor&
        cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const paddle::Tensor&
        cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const paddle::Tensor& num_blocks_cpu,
    const paddle::Tensor& max_enc_len,
    const paddle::Tensor& max_dec_len,
    paddle::Tensor& out_tmp,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>&
        cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& shift_bias,
    const paddle::optional<paddle::Tensor>& smooth_weight,
    const std::string& cache_quant_type_str,
    const int max_input_len,
    const int block_shape_q,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const float in_scale,
    const std::string& mode,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill) {
  switch (qkv.type()) {
    case paddle::DataType::FLOAT16: {
      return append_attention<paddle::DataType::FLOAT16>(
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
          num_blocks_cpu,
          max_enc_len,
          max_dec_len,
          out_tmp,
          cache_quant_type_str,
          max_input_len,
          block_shape_q,
          num_heads,
          kv_num_heads,
          head_dim,
          in_scale,
          mode,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          is_decoder,
          enable_prefill);
    }
    case paddle::DataType::BFLOAT16: {
      return append_attention<paddle::DataType::BFLOAT16>(
          qkv,      // [token_num, num_heads, head_dim]
          cache_k,  // [max_block_num, num_heads, block_size, head_dim]
          cache_v,  // [max_block_num, num_heads, head_dim, block_size]
          attn_mask,
          cache_k_scale,  // [num_kv_heads, head_dim]
          cache_v_scale,  // [num_kv_heads, head_dim]
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
          num_blocks_cpu,
          max_enc_len,
          max_dec_len,
          out_tmp,
          cache_quant_type_str,
          max_input_len,
          block_shape_q,
          num_heads,
          kv_num_heads,
          head_dim,
          in_scale,
          mode,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          is_decoder,
          enable_prefill);
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and bfloat16 are supported. ");
      break;
    }
  }
}

PD_BUILD_OP(append_attention)
    .Inputs({"qkv",
             "cache_k",
             "cache_v",
             "seq_lens_q",
             "seq_lens_kv",
             "seq_lens_encoder",
             "padding_offsets",
             "cum_offsets",
             "block_table",
             "batch_ids",
             "tile_ids_per_batch",
             "num_blocks_cpu",
             "max_enc_len",
             "max_dec_len",
             "out_tmp",
             paddle::Optional("attn_mask"),
             paddle::Optional("cache_k_scale"),
             paddle::Optional("cache_v_scale"),
             paddle::Optional("cache_k_zp"),
             paddle::Optional("cache_v_zp"),
             paddle::Optional("shift_bias"),
             paddle::Optional("smooth_weight")})
    .Outputs({"out"})
    .SetInplaceMap({{"out_tmp", "out"}})
    .Attrs({"cache_quant_type_str: std::string",
            "max_input_len: int",
            "block_shape_q: int",
            "num_heads: int",
            "kv_num_heads: int",
            "head_dim: int",
            "in_scale: float",
            "mode: std::string",
            "max_partition_size: int",
            "encoder_max_partition_size: int",
            "speculate_max_draft_token_num: int",
            "causal: bool",
            "is_decoder: bool",
            "enable_prefill: bool"})
    .SetKernelFn(PD_KERNEL(AppendAttention));
// .SetInferDtypeFn(PD_INFER_DTYPE(AppendAttentionInferDtype));
