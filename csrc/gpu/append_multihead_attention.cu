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
#include "append_attn/decoder_write_cache_with_rope_kernel.h"
#include "append_attn/encoder_write_cache_with_rope_kernel.h"

template <paddle::DataType D>
std::vector<paddle::Tensor> AppendMultiheadAttentionKernel(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& max_enc_len_this_time,
    const paddle::Tensor& max_dec_len_this_time,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_input_length,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const auto& qkv_dims = qkv.dims();
  const auto& key_cache_dims = key_cache.dims();
  const int token_num = qkv_dims[0];
  const int kv_num_heads = key_cache_dims[1];
  const int head_dim = key_cache_dims[3];
  const int total_num_head = qkv_dims[qkv_dims.size() - 1] / head_dim;
  const int num_heads = total_num_head - 2 * kv_num_heads;

  int encoder_num_blocks_data = encoder_num_blocks.data<int>()[0];
  int kv_num_blocks_data = kv_num_blocks.data<int>()[0];
  int decoder_num_blocks_data = decoder_num_blocks.data<int>()[0];
  int max_enc_len_this_time_data = max_enc_len_this_time.data<int>()[0];
  int max_dec_len_this_time_data = max_dec_len_this_time.data<int>()[0];
  auto stream = qkv.stream();

  paddle::Tensor qkv_out;
  if (qkv_out_scales) {
    qkv_out = paddle::empty({token_num, total_num_head, head_dim}, D, qkv.place());
  } else {
    qkv_out = qkv;
  }
  paddle::Tensor fmha_out;
  if (out_linear_in_scale > 0.0) {
    fmha_out = paddle::empty(
        {token_num, num_heads * head_dim}, paddle::DataType::INT8, qkv.place());
  } else {
    fmha_out = paddle::empty({token_num, num_heads * head_dim}, D, qkv.place());
  }

  if (max_enc_len_this_time_data > 0) {
    if (qkv_out_scales) {
      EncoderWriteCacheWithRopeKernel<data_t, int>(
          qkv,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          padding_offsets,
          cum_offsets,
          block_tables,
          kv_batch_ids,
          kv_tile_ids_per_batch,
          rotary_embs,
          qkv_out_scales,
          qkv_bias,
          cache_k_quant_scales,
          cache_v_quant_scales,
          cache_k_zp,
          cache_v_zp,
          cache_quant_type_str,
          kv_num_blocks_data,
          max_input_length,
          num_heads,
          kv_num_heads,
          head_dim,
          use_neox_rotary_style,
          stream,
          &qkv_out,
          const_cast<paddle::Tensor*>(&key_cache),
          const_cast<paddle::Tensor*>(&value_cache));
    } else {
      EncoderWriteCacheWithRopeKernel<data_t, data_t>(
          qkv_out,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          padding_offsets,
          cum_offsets,
          block_tables,
          kv_batch_ids,
          kv_tile_ids_per_batch,
          rotary_embs,
          qkv_out_scales,
          qkv_bias,
          cache_k_quant_scales,
          cache_v_quant_scales,
          cache_k_zp,
          cache_v_zp,
          cache_quant_type_str,
          kv_num_blocks_data,
          max_input_length,
          num_heads,
          kv_num_heads,
          head_dim,
          use_neox_rotary_style,
          stream,
          &qkv_out,
          const_cast<paddle::Tensor*>(&key_cache),
          const_cast<paddle::Tensor*>(&value_cache));
    }
    if (out_linear_in_scale > 0.0) {
      CascadeAppendAttentionKernel<data_t, int8_t>(
          qkv_out,
          key_cache,
          value_cache,
          attn_mask,
          cache_k_dequant_scales,
          cache_v_dequant_scales,
          cache_k_zp,
          cache_v_zp,
          out_linear_shifts,
          out_linear_smooths,
          seq_lens_this_time,
          seq_lens_decoder,
          seq_lens_encoder,
          padding_offsets,
          cum_offsets,
          block_tables,
          encoder_batch_ids,
          encoder_tile_ids_per_batch,
          cache_quant_type_str,
          encoder_num_blocks_data,
          encoder_block_shape_q,
          max_input_length,
          max_enc_len_this_time_data,
          num_heads,
          kv_num_heads,
          head_dim,
          out_linear_in_scale,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          false,
          enable_prefill,
          stream,
          &fmha_out);
    } else {
      CascadeAppendAttentionKernel<data_t, data_t>(
          qkv_out,
          key_cache,
          value_cache,
          attn_mask,
          cache_k_dequant_scales,
          cache_v_dequant_scales,
          cache_k_zp,
          cache_v_zp,
          out_linear_shifts,
          out_linear_smooths,
          seq_lens_this_time,
          seq_lens_decoder,
          seq_lens_encoder,
          padding_offsets,
          cum_offsets,
          block_tables,
          encoder_batch_ids,
          encoder_tile_ids_per_batch,
          cache_quant_type_str,
          encoder_num_blocks_data,
          encoder_block_shape_q,
          max_input_length,
          max_enc_len_this_time_data,
          num_heads,
          kv_num_heads,
          head_dim,
          out_linear_in_scale,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          false,
          enable_prefill,
          stream,
          &fmha_out);
    }
  }

  if (max_dec_len_this_time_data > 0) {
    if (qkv_out_scales) {
      DecoderWriteCacheWithRoPEKernel<data_t, int>(
          qkv,  // [token_num, num_heads, head_dim]
          seq_lens_decoder,
          seq_lens_encoder,
          padding_offsets,
          cum_offsets,
          block_tables,
          rotary_embs,
          qkv_out_scales,
          qkv_bias,
          cache_k_quant_scales,
          cache_v_quant_scales,
          cache_k_zp,
          cache_v_zp,
          cache_quant_type_str,
          use_neox_rotary_style,
          max_input_length,
          num_heads,
          kv_num_heads,
          head_dim,
          stream,
          &qkv_out,
          const_cast<paddle::Tensor*>(&key_cache),
          const_cast<paddle::Tensor*>(&value_cache));
    } else {
      DecoderWriteCacheWithRoPEKernel<data_t, data_t>(
          qkv_out,  // [token_num, num_heads, head_dim]
          seq_lens_decoder,
          seq_lens_encoder,
          padding_offsets,
          cum_offsets,
          block_tables,
          rotary_embs,
          qkv_out_scales,
          qkv_bias,
          cache_k_quant_scales,
          cache_v_quant_scales,
          cache_k_zp,
          cache_v_zp,
          cache_quant_type_str,
          use_neox_rotary_style,
          max_input_length,
          num_heads,
          kv_num_heads,
          head_dim,
          stream,
          &qkv_out,
          const_cast<paddle::Tensor*>(&key_cache),
          const_cast<paddle::Tensor*>(&value_cache));
    }
    auto decoder_stream = qkv.stream();
    // cudaEvent_t event;
    // cudaStream_t decoder_stream;

    // cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    // cudaStreamCreate(&decoder_stream);
    if (out_linear_in_scale > 0.0) {
      CascadeAppendAttentionKernel<data_t, int8_t>(
          qkv_out,
          key_cache,
          value_cache,
          attn_mask,
          cache_k_dequant_scales,
          cache_v_dequant_scales,
          cache_k_zp,
          cache_v_zp,
          out_linear_shifts,
          out_linear_smooths,
          seq_lens_this_time,
          seq_lens_decoder,
          seq_lens_encoder,
          padding_offsets,
          cum_offsets,
          block_tables,
          decoder_batch_ids,
          decoder_tile_ids_per_batch,
          cache_quant_type_str,
          decoder_num_blocks_data,
          decoder_block_shape_q,
          max_input_length,
          max_dec_len_this_time_data + 1,
          num_heads,
          kv_num_heads,
          head_dim,
          out_linear_in_scale,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          false,
          enable_prefill,
          decoder_stream,
          &fmha_out);
    } else {
      CascadeAppendAttentionKernel<data_t, data_t>(
          qkv_out,
          key_cache,
          value_cache,
          attn_mask,
          cache_k_dequant_scales,
          cache_v_dequant_scales,
          cache_k_zp,
          cache_v_zp,
          out_linear_shifts,
          out_linear_smooths,
          seq_lens_this_time,
          seq_lens_decoder,
          seq_lens_encoder,
          padding_offsets,
          cum_offsets,
          block_tables,
          decoder_batch_ids,
          decoder_tile_ids_per_batch,
          cache_quant_type_str,
          decoder_num_blocks_data,
          decoder_block_shape_q,
          max_input_length,
          max_dec_len_this_time_data + 1,
          num_heads,
          kv_num_heads,
          head_dim,
          out_linear_in_scale,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          false,
          enable_prefill,
          decoder_stream,
          &fmha_out);
    }
    // cudaEventRecord(event, decoder_stream);
    // cudaStreamWaitEvent(stream, event);
  }

  return {fmha_out};
}

std::vector<paddle::Tensor> AppendMultiheadAttention(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& max_enc_len_this_time,
    const paddle::Tensor& max_dec_len_this_time,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_input_length,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill) {
  switch (qkv.dtype()) {
    case paddle::DataType::FLOAT16: {
      return AppendMultiheadAttentionKernel<paddle::DataType::FLOAT16>(
          qkv,
          key_cache,
          value_cache,
          seq_lens_encoder,
          seq_lens_decoder,
          seq_lens_this_time,
          padding_offsets,
          cum_offsets,
          block_tables,
          encoder_batch_ids,
          encoder_tile_ids_per_batch,
          encoder_num_blocks,
          kv_batch_ids,
          kv_tile_ids_per_batch,
          kv_num_blocks,
          decoder_batch_ids,
          decoder_tile_ids_per_batch,
          decoder_num_blocks,
          max_enc_len_this_time,
          max_dec_len_this_time,
          rotary_embs,
          attn_mask,
          qkv_bias,
          qkv_out_scales,
          cache_k_quant_scales,
          cache_v_quant_scales,
          cache_k_dequant_scales,
          cache_v_dequant_scales,
          cache_k_zp,
          cache_v_zp,
          out_linear_shifts,
          out_linear_smooths,
          cache_quant_type_str,
          use_neox_rotary_style,
          max_input_length,
          out_linear_in_scale,
          encoder_block_shape_q,
          decoder_block_shape_q,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          is_decoder,
          enable_prefill);
    }
    case paddle::DataType::BFLOAT16: {
      return AppendMultiheadAttentionKernel<paddle::DataType::BFLOAT16>(
          qkv,
          key_cache,
          value_cache,
          seq_lens_encoder,
          seq_lens_decoder,
          seq_lens_this_time,
          padding_offsets,
          cum_offsets,
          block_tables,
          encoder_batch_ids,
          encoder_tile_ids_per_batch,
          encoder_num_blocks,
          kv_batch_ids,
          kv_tile_ids_per_batch,
          kv_num_blocks,
          decoder_batch_ids,
          decoder_tile_ids_per_batch,
          decoder_num_blocks,
          max_enc_len_this_time,
          max_dec_len_this_time,
          rotary_embs,
          attn_mask,
          qkv_bias,
          qkv_out_scales,
          cache_k_quant_scales,
          cache_v_quant_scales,
          cache_k_dequant_scales,
          cache_v_dequant_scales,
          cache_k_zp,
          cache_v_zp,
          out_linear_shifts,
          out_linear_smooths,
          cache_quant_type_str,
          use_neox_rotary_style,
          max_input_length,
          out_linear_in_scale,
          encoder_block_shape_q,
          decoder_block_shape_q,
          max_partition_size,
          encoder_max_partition_size,
          speculate_max_draft_token_num,
          causal,
          is_decoder,
          enable_prefill);
    }
    case paddle::DataType::INT32: {
      if (compute_dtype == "bf16") {
        return AppendMultiheadAttentionKernel<paddle::DataType::BFLOAT16>(
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            padding_offsets,
            cum_offsets,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            max_enc_len_this_time,
            max_dec_len_this_time,
            rotary_embs,
            attn_mask,
            qkv_bias,
            qkv_out_scales,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            cache_k_zp,
            cache_v_zp,
            out_linear_shifts,
            out_linear_smooths,
            cache_quant_type_str,
            use_neox_rotary_style,
            max_input_length,
            out_linear_in_scale,
            encoder_block_shape_q,
            decoder_block_shape_q,
            max_partition_size,
            encoder_max_partition_size,
            speculate_max_draft_token_num,
            causal,
            is_decoder,
            enable_prefill);
      } else if (compute_dtype == "fp16") {
        return AppendMultiheadAttentionKernel<paddle::DataType::FLOAT16>(
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            padding_offsets,
            cum_offsets,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            max_enc_len_this_time,
            max_dec_len_this_time,
            rotary_embs,
            attn_mask,
            qkv_bias,
            qkv_out_scales,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            cache_k_zp,
            cache_v_zp,
            out_linear_shifts,
            out_linear_smooths,
            cache_quant_type_str,
            use_neox_rotary_style,
            max_input_length,
            out_linear_in_scale,
            encoder_block_shape_q,
            decoder_block_shape_q,
            max_partition_size,
            encoder_max_partition_size,
            speculate_max_draft_token_num,
            causal,
            is_decoder,
            enable_prefill);
      } else {
        PD_THROW("Only supported attr of compute_dtype in ['fp16', 'bf16'].");
        break;
      }
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and bfloat16 are supported. ");
      break;
    }
  }
  return {paddle::Tensor{}};
}

std::vector<std::vector<int64_t>> AppendMultiheadAttentionInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& padding_offsets_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& encoder_batch_ids_shape,
    const std::vector<int64_t>& encoder_tile_ids_per_batch_shape,
    const std::vector<int64_t>& encoder_num_blocks_shape,
    const std::vector<int64_t>& kv_batch_ids_shape,
    const std::vector<int64_t>& kv_tile_ids_per_batch_shape,
    const std::vector<int64_t>& kv_num_blocks_shape,
    const std::vector<int64_t>& decoder_batch_ids_shape,
    const std::vector<int64_t>& decoder_tile_ids_per_batch_shape,
    const std::vector<int64_t>& decoder_num_blocks_shape,
    const std::vector<int64_t>& max_enc_len_this_time_shape,
    const std::vector<int64_t>& max_dec_len_this_time_shape,
    const paddle::optional<std::vector<int64_t>>& rotary_embs_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_bias_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_out_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_zp_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_zp_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_shifts_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_smooths_shape) {
  const int token_num = qkv_shape[0];
  const int kv_num_heads = key_cache_shape[1];
  const int head_dim = key_cache_shape[3];
  const int total_num_head = qkv_shape[qkv_shape.size() - 1] / head_dim;
  const int num_heads = total_num_head - 2 * kv_num_heads;
  return {{token_num, num_heads * head_dim}};
}

std::vector<paddle::DataType> AppendMultiheadAttentionInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& padding_offsets_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& encoder_batch_ids_dtype,
    const paddle::DataType& encoder_tile_ids_per_batch_dtype,
    const paddle::DataType& encoder_num_blocks_dtype,
    const paddle::DataType& kv_batch_ids_dtype,
    const paddle::DataType& kv_tile_ids_per_batch_dtype,
    const paddle::DataType& kv_num_blocks_dtype,
    const paddle::DataType& decoder_batch_ids_dtype,
    const paddle::DataType& decoder_tile_ids_per_batch_dtype,
    const paddle::DataType& decoder_num_blocks_dtype,
    const paddle::DataType& max_enc_len_this_time_dtype,
    const paddle::DataType& max_dec_len_this_time_dtype,
    const paddle::optional<paddle::DataType>& rotary_embs_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_dtype,
    const paddle::optional<paddle::DataType>& qkv_bias_dtype,
    const paddle::optional<paddle::DataType>& qkv_out_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_zp_dtype,
    const paddle::optional<paddle::DataType>& cache_v_zp_dtype,
    const paddle::optional<paddle::DataType>& out_linear_shifts_dtype,
    const paddle::optional<paddle::DataType>& out_linear_smooths_dtype,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_input_length,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill) {
  if (out_linear_in_scale > 0.0) {
    return {paddle::DataType::INT8};
  }
  return {qkv_dtype};
}

PD_BUILD_OP(append_multihead_attention)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "padding_offsets",
             "cum_offsets",
             "block_tables",
             "encoder_batch_ids",
             "encoder_tile_ids_per_batch",
             "encoder_num_blocks",
             "kv_batch_ids",
             "kv_tile_ids_per_batch",
             "kv_num_blocks",
             "decoder_batch_ids",
             "decoder_tile_ids_per_batch",
             "decoder_num_blocks",
             "max_enc_len_this_time",
             "max_dec_len_this_time",
             paddle::Optional("rotary_embs"),
             paddle::Optional("attn_mask"),
             paddle::Optional("qkv_bias"),
             paddle::Optional("qkv_out_scales"),
             paddle::Optional("cache_k_quant_scales"),
             paddle::Optional("cache_v_quant_scales"),
             paddle::Optional("cache_k_dequant_scales"),
             paddle::Optional("cache_v_dequant_scales"),
             paddle::Optional("cache_k_zp"),
             paddle::Optional("cache_v_zp"),
             paddle::Optional("out_linear_shifts"),
             paddle::Optional("out_linear_smooths")})
    .Outputs({"fmha_out", "key_cache_out", "value_cache_out"})
    .SetInplaceMap({{"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .Attrs({"compute_type: std::string",
            "cache_quant_type: std::string",
            "use_neox_rotary_style: bool",
            "max_input_length: int",
            "out_linear_in_scale: float",
            "encoder_block_shape_q: int",
            "decoder_block_shape_q: int",
            "max_partition_size: int",
            "encoder_max_partition_size: int",
            "speculate_max_draft_token_num: int",
            "causal: bool",
            "is_decoder: bool",
            "enable_prefill: bool"})
    .SetKernelFn(PD_KERNEL(AppendMultiheadAttention))
    .SetInferShapeFn(PD_INFER_SHAPE(AppendMultiheadAttentionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AppendMultiheadAttentionInferDtype));