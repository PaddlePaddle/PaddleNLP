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

#include "append_attn/multi_head_latent_attention_kernel.h"

template <paddle::DataType D>
std::vector<paddle::Tensor> MultiHeadLatentAttentionKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& query,
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
    const paddle::Tensor& max_len_kv,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& query_bias,
    const paddle::optional<paddle::Tensor>& query_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const std::string& cache_quant_type_str,
    const int max_input_length,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  int encoder_num_blocks_data = encoder_num_blocks.data<int>()[0];
  int kv_num_blocks_data = kv_num_blocks.data<int>()[0];
  int decoder_num_blocks_data = decoder_num_blocks.data<int>()[0];
  int max_enc_len_this_time_data = max_enc_len_this_time.data<int>()[0];
  int max_dec_len_this_time_data = max_dec_len_this_time.data<int>()[0];
  int max_len_kv_data = max_len_kv.data<int>()[0];
  const int encoder_block_shape_q = get_encoder_block_shape_q();
  const int decoder_block_shape_q = get_decoder_block_shape_q();
  auto main_stream = query.stream();
  static cudaEvent_t main_event;
  static cudaEvent_t decoder_event;       
  static cudaStream_t decoder_stream;
  static bool init_flag = false;
  if (max_enc_len_this_time_data > 0 && max_dec_len_this_time_data > 0 &&
      !init_flag) {
    cudaEventCreateWithFlags(&main_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&decoder_event, cudaEventDisableTiming);
    cudaStreamCreateWithFlags(&decoder_stream, cudaStreamNonBlocking);
    init_flag = true;
  }

  paddle::Tensor fmha_out;
  if (out_linear_in_scale > 0.0) {
    if (fabs(quant_max_bound - 127.0f) < 0.000001) {
      fmha_out = GetEmptyTensor(
        {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims_v},
        paddle::DataType::INT8,
        query.place());
    } 
    else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
      fmha_out = GetEmptyTensor(
        {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims_v},
        paddle::DataType::FLOAT8_E4M3FN,
        query.place());
    }else{
      PD_THROW("Only supported attr of quant_max_bound in ['127.0', '448.0'].");
    }
  } else {
    fmha_out = GetEmptyTensor(
        {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims_v},
        D,
        query.place());
  }

  if (max_enc_len_this_time_data > 0) {
    if (max_dec_len_this_time_data > 0) {
      cudaEventRecord(main_event, main_stream);
    }
  }

  if (max_dec_len_this_time_data > 0) {
    cudaStream_t exec_stream;
    if (max_enc_len_this_time_data > 0) {
      cudaStreamWaitEvent(decoder_stream, main_event);
      exec_stream = decoder_stream;
    } else {
      exec_stream = main_stream;
    }
    std::cout << "dec_mla" << std::endl;
    DecodeMLAAttentionKernel<data_t>(
        meta_data,
        query, // [token_num, num_heads, head_dim]
        key_cache,
        value_cache,
        attn_mask,
        out_linear_shifts,
        out_linear_smooths,
        seq_lens_this_time, // q_seq_len is 1
        seq_lens_decoder,
        padding_offsets,
        cum_offsets,
        block_tables,
        max_input_length,
        max_len_kv_data,
        softmax_scale,
        out_linear_in_scale,
        causal,
        exec_stream,
        &fmha_out);
    if (max_enc_len_this_time_data > 0) {
      cudaEventRecord(decoder_event, exec_stream);
      cudaStreamWaitEvent(main_stream, decoder_event);
    }
  }
  return {fmha_out};
}

std::vector<paddle::Tensor> MultiHeadLatentAttention(
    const paddle::Tensor& query,
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
    const paddle::Tensor& max_len_kv,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& query_bias,
    const paddle::optional<paddle::Tensor>& query_out_scales,
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
    const int nope_size,
    const int max_input_length,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder) {
  AppendAttnMetaData meta_data;

  const auto& query_dims = query.dims();
  const auto& key_cache_dims = key_cache.dims();
  const int q_hidden_size = query_dims[query_dims.size() - 1];
  meta_data.token_nums = query_dims[0];
  meta_data.kv_num_heads = key_cache_dims[1];
  meta_data.head_dims = key_cache_dims[3];
  meta_data.head_dims_v = nope_size;
  meta_data.q_num_heads = q_hidden_size / meta_data.head_dims;

  meta_data.max_blocks_per_seq = block_tables.dims()[1];
  meta_data.block_size = key_cache.dims()[2];
  meta_data.batch_size = cum_offsets.dims()[0];

  switch (query.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return MultiHeadLatentAttentionKernel<paddle::DataType::BFLOAT16>(
          meta_data,
          query,
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
          max_len_kv,
          attn_mask,
          query_bias,
          query_out_scales,
          cache_k_quant_scales,
          cache_v_quant_scales,
          cache_k_dequant_scales,
          cache_v_dequant_scales,
          cache_k_zp,
          cache_v_zp,
          out_linear_shifts,
          out_linear_smooths,
          cache_quant_type_str,
          max_input_length,
          softmax_scale,
          quant_max_bound,
          quant_min_bound,
          out_linear_in_scale,
          speculate_max_draft_token_num,
          causal,
          speculate_decoder);
    }
  }
  return {paddle::Tensor{}};
}

std::vector<std::vector<int64_t>> MultiHeadLatentAttentionInferShape(
    const std::vector<int64_t>& query_shape,
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
    const std::vector<int64_t>& max_len_kv_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_shape,
    const paddle::optional<std::vector<int64_t>>& query_bias_shape,
    const paddle::optional<std::vector<int64_t>>& query_out_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_zp_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_zp_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_shifts_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_smooths_shape,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const int nope_size,
    const int max_input_length,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder) {
  const int token_num = query_shape[0];
  const int kv_num_heads = key_cache_shape[1];
  const int head_dim_qk = key_cache_shape[3];
  const int head_dim_v = nope_size;
  const int q_hidden_size = query_shape[query_shape.size() - 1];
  const int num_heads = q_hidden_size / head_dim_qk;
  return {{token_num, num_heads * head_dim_v}};
}

std::vector<paddle::DataType> MultiHeadLatentAttentionInferDtype(
    const paddle::DataType& query_dtype,
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
    const paddle::DataType& max_len_kv_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_dtype,
    const paddle::optional<paddle::DataType>& query_bias_dtype,
    const paddle::optional<paddle::DataType>& query_out_scales_dtype,
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
    const int nope_size,
    const int max_input_length,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder) {
  if (compute_dtype == "bf16") {
    if (out_linear_in_scale > 0.0) {
      if (fabs(quant_max_bound - 127.0f) < 0.000001) {
        return {paddle::DataType::INT8, paddle::DataType::BFLOAT16};
      } else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
        return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::BFLOAT16};
      }else{
        PD_THROW("Only supported attr of quant_max_bound in ['127.0', '448.0'].");
      }
    } else {
      return {paddle::DataType::BFLOAT16, paddle::DataType::BFLOAT16};
    }
  } else if (compute_dtype == "fp16") {
    if (out_linear_in_scale > 0.0) {
      if (fabs(quant_max_bound - 127.0f) < 0.000001) {
        return {paddle::DataType::INT8, paddle::DataType::FLOAT16};
      } else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
        return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT16};
      }else{
        PD_THROW("Only supported attr of quant_max_bound in ['127.0', '448.0'].");
      }
    } else {
      return {paddle::DataType::FLOAT16, paddle::DataType::FLOAT16};
    }
  } else {
    PD_THROW("Only supported attr of compute_dtype in ['fp16', 'bf16'].");
  }
}

PD_BUILD_OP(multi_head_latent_attention)
    .Inputs({"query",
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
             "max_len_kv",
             paddle::Optional("attn_mask"),
             paddle::Optional("query_bias"),
             paddle::Optional("query_out_scales"),
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
            "nope_size: int",
            "max_input_length: int",
            "softmax_scale: float",
            "quant_max_bound: float",
            "quant_min_bound: float",
            "out_linear_in_scale: float",
            "speculate_max_draft_token_num: int",
            "causal: bool",
            "speculate_decoder: bool"})
    .SetKernelFn(PD_KERNEL(MultiHeadLatentAttention))
    .SetInferShapeFn(PD_INFER_SHAPE(MultiHeadLatentAttentionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MultiHeadLatentAttentionInferDtype));