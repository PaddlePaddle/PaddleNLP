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

#include "paddle/extension.h"
#include "all_reduce.h"

namespace py = pybind11;

std::vector<paddle::Tensor> AppendAttention(
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
    const paddle::Tensor& max_len_kv,
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
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder);

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

std::vector<paddle::Tensor> MultiHeadLatentAttention(
    const paddle::Tensor& query,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& cu_seqlens_q,
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
    const paddle::Tensor& decoder_num_blocks_cpu,
    const paddle::Tensor& decoder_chunk_size_cpu,
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
    const int speculate_draft_total_token_num,
    const bool causal,
    const bool speculate_decoder);

std::vector<paddle::Tensor> GetBlockShapeAndSplitKVBlock(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& max_enc_len_this_time,
    const paddle::Tensor& max_dec_len_this_time,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& cum_offsets,
    const int group_size,
    const int block_size,
    const int decoder_step_token_num);

std::vector<paddle::Tensor> NoauxTc(paddle::Tensor& scores,
                                    paddle::Tensor& scores_with_bias,
                                    int n_group,
                                    int topk_group,
                                    int topk,
                                    float routed_scaling_factor);

std::vector<paddle::Tensor> PrefillMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len);

std::vector<paddle::Tensor> DecodeMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len,
    const bool speculate_decoder);

paddle::Tensor cutlass_fp8_fp8_half_block_gemm_fused_func(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::Tensor& x_scale,
    const paddle::Tensor& y_scale,
    const paddle::optional<paddle::Tensor>& bias,
    bool trans_x,
    bool trans_y,
    std::string output_dtype,
    std::string activation_type);

void GetPositionIdsAndMaskEncoderBatch(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& position_ids,
    const paddle::Tensor& mask_encoder_batch);

void SetPreidsTokenPenaltyMultiScores(const paddle::Tensor& pre_ids,
                                      const paddle::Tensor& input_ids,
                                      const paddle::Tensor& seq_lens_encoder,
                                      const paddle::Tensor& seq_lens_decoder,
                                      const paddle::Tensor& step_idx,
                                      const paddle::Tensor& stop_flags,
                                      const paddle::Tensor& logits,
                                      const paddle::Tensor& penalty_scores,
                                      const paddle::Tensor& frequency_scores,
                                      const paddle::Tensor& presence_scores,
                                      const paddle::Tensor& temperatures,
                                      const paddle::Tensor& bad_tokens,
                                      const paddle::Tensor& cur_len,
                                      const paddle::Tensor& min_len,
                                      const paddle::Tensor& eos_token_id);

void UpdateInputesV2(const paddle::Tensor& stop_flags,
               const paddle::Tensor& step_idx,
               const paddle::Tensor& not_need_stop, // cpu
               const paddle::Tensor& seq_lens_this_time,
               const paddle::Tensor& seq_lens_encoder,
               const paddle::Tensor& seq_lens_decoder,
               const paddle::Tensor& max_dec_len,
               const paddle::Tensor& input_ids,
               const paddle::Tensor& stop_nums,
               const paddle::Tensor& next_tokens,
               const paddle::Tensor& is_block_step,
               const paddle::Tensor& end_ids,
               const paddle::Tensor& kwargs_next_tokens);

paddle::Tensor RebuildPaddingV2Func(const paddle::Tensor& tmp_out, // [token_num, dim_embed]
                                    const paddle::Tensor& cum_offsets, // [bsz, 1]
                                    const paddle::Tensor& seq_lens_decoder,
                                    const paddle::Tensor& seq_lens_encoder,
                                    const paddle::optional<paddle::Tensor>& output_padding_offset,
                                    int max_input_length);

std::vector<paddle::Tensor> PerTokenGroupQuant(const paddle::Tensor& x,
                                        const int group_size,
                                        const bool transpose_scale,
                                        const float quant_max_bound,
                                        const float quant_min_bound);

std::vector<paddle::Tensor> PerTensorQuantFp8(const paddle::Tensor& x, const paddle::optional<paddle::Tensor>& scale);

std::vector<paddle::Tensor> GetPaddingOffsetV2(const paddle::Tensor& input_ids,
                                               const paddle::Tensor& cum_offsets,
                                               const paddle::Tensor& token_num,
                                               const paddle::Tensor& seq_len,
                                               const paddle::optional<paddle::Tensor>& draft_tokens,
                                               const paddle::optional<paddle::Tensor>& seq_lens_encoder);

void SaveOutMmsg(const paddle::Tensor& x,
                 const paddle::Tensor& not_need_stop, // cpu
                 const paddle::Tensor& msg_queue_id,      // cpu
                 int64_t rank_id);

void GetOutput(const paddle::Tensor& x,
               const paddle::Tensor& msg_queue_id, // cpu
               int64_t rank_id,
               bool wait_flag);

void StepPaddle(const paddle::Tensor &stop_flags,
                const paddle::Tensor &seq_lens_this_time,
                const paddle::Tensor &ori_seq_lens_encoder,
                const paddle::Tensor &seq_lens_encoder,
                const paddle::Tensor &seq_lens_decoder,
                const paddle::Tensor &block_tables,  // [bsz, block_num_per_seq]
                const paddle::Tensor &encoder_block_lens,
                const paddle::Tensor &is_block_step,
                const paddle::Tensor &step_block_list,
                const paddle::Tensor &step_lens,
                const paddle::Tensor &recover_block_list,
                const paddle::Tensor &recover_lens,
                const paddle::Tensor &need_block_list,
                const paddle::Tensor &need_block_len,
                const paddle::Tensor &used_list_len,
                const paddle::Tensor &free_list,
                const paddle::Tensor &free_list_len,
                const paddle::Tensor &input_ids,
                const paddle::Tensor &pre_ids,
                const paddle::Tensor &step_idx,
                const paddle::Tensor &next_tokens,
                const paddle::Tensor &first_token_ids,
                const int block_size,
                const int encoder_decoder_block_num);

void SaveOutputDygraph(
    const paddle::Tensor& all_token_ids,
    const paddle::Tensor& tokens,
    const paddle::Tensor& result_ids,
    const paddle::Tensor& step_idx
);

PYBIND11_MODULE(paddlenlp_ops, m) {
  /**
   * all_reduce.cu
   */
  m.def("init_custom_all_reduce", &init_custom_all_reduce, "init all reduce class function");
  m.def("all_reduce", &all_reduce, "all reduce function");
  m.def("dispose", &dispose, "del function for python");
  m.def("meta_size", &meta_size, "meta_size function for Signal struct");
  m.def("register_buffer", &register_buffer, "register ipc buffer");
  m.def("f_append_attention", &AppendAttention, "AppendAttention");
  m.def("f_fused_rotary_position_encoding", &FusedRotaryPositionEncoding, "FusedRotaryPositionEncoding");
  m.def("f_multi_head_latent_attention", &MultiHeadLatentAttention, "MultiHeadLatentAttention");
  m.def("f_noaux_tc", &NoauxTc, "NoauxTc");
  m.def("f_get_block_shape_and_split_kv_block", &GetBlockShapeAndSplitKVBlock, "GetBlockShapeAndSplitKVBlock");
  m.def("f_prefill_mla_write_cache", &PrefillMLAWriteCacheKernel, "PrefillMLAWriteCacheKernel");
  m.def("f_decode_mla_write_cache", &DecodeMLAWriteCacheKernel, "DecodeMLAWriteCacheKernel");
  m.def("f_get_position_ids_and_mask_encoder_batch", &GetPositionIdsAndMaskEncoderBatch, "GetPositionIdsAndMaskEncoderBatch");
  m.def("f_set_preids_token_penalty_multi_scores", &SetPreidsTokenPenaltyMultiScores, "SetPreidsTokenPenaltyMultiScores");
  m.def("f_update_inputs_v2", &UpdateInputesV2, "UpdateInputesV2");
  m.def("f_rebuild_padding_v2", &RebuildPaddingV2Func, "RebuildPaddingV2Func");
  m.def("f_per_token_group_quant", &PerTokenGroupQuant, "PerTokenGroupQuant");
  m.def("f_per_tensor_quant_fp8", &PerTensorQuantFp8, "PerTensorQuantFp8");
  m.def("f_get_padding_offset_v2", &GetPaddingOffsetV2, "GetPaddingOffsetV2");
  m.def("f_save_output", &SaveOutMmsg, "SaveOutMmsg");
  m.def("f_get_output", &GetOutput, "GetOutput");
  m.def("f_step_paddle", &StepPaddle, "StepPaddle");
  m.def("f_save_output_dygraph", &SaveOutputDygraph, "SaveOutputDygraph");
//   m.def("f_cutlass_fp8_fp8_half_block_gemm_fused", &cutlass_fp8_fp8_half_block_gemm_fused_func, "cutlass_fp8_fp8_half_block_gemm_fused_func");
}

PYBIND11_MODULE(paddlenlp_ops_80, m) {
  /**
   * all_reduce.cu
   */
  m.def("init_custom_all_reduce", &init_custom_all_reduce, "init all reduce class function");
  m.def("all_reduce", &all_reduce, "all reduce function");
  m.def("dispose", &dispose, "del function for python");
  m.def("meta_size", &meta_size, "meta_size function for Signal struct");
  m.def("register_buffer", &register_buffer, "register ipc buffer");
  m.def("f_append_attention", &AppendAttention, "AppendAttention");
  m.def("f_fused_rotary_position_encoding", &FusedRotaryPositionEncoding, "FusedRotaryPositionEncoding");
  m.def("f_multi_head_latent_attention", &MultiHeadLatentAttention, "MultiHeadLatentAttention");
  m.def("f_noaux_tc", &NoauxTc, "NoauxTc");
  m.def("f_get_block_shape_and_split_kv_block", &GetBlockShapeAndSplitKVBlock, "GetBlockShapeAndSplitKVBlock");
  m.def("f_prefill_mla_write_cache", &PrefillMLAWriteCacheKernel, "PrefillMLAWriteCacheKernel");
  m.def("f_decode_mla_write_cache", &DecodeMLAWriteCacheKernel, "DecodeMLAWriteCacheKernel");
  m.def("f_get_position_ids_and_mask_encoder_batch", &GetPositionIdsAndMaskEncoderBatch, "GetPositionIdsAndMaskEncoderBatch");
  m.def("f_set_preids_token_penalty_multi_scores", &SetPreidsTokenPenaltyMultiScores, "SetPreidsTokenPenaltyMultiScores");
  m.def("f_update_inputs_v2", &UpdateInputesV2, "UpdateInputesV2");
  m.def("f_rebuild_padding_v2", &RebuildPaddingV2Func, "RebuildPaddingV2Func");
  m.def("f_per_token_group_quant", &PerTokenGroupQuant, "PerTokenGroupQuant");
  m.def("f_per_tensor_quant_fp8", &PerTensorQuantFp8, "PerTensorQuantFp8");
  m.def("f_get_padding_offset_v2", &GetPaddingOffsetV2, "GetPaddingOffsetV2");
  m.def("f_save_output", &SaveOutMmsg, "SaveOutMmsg");
  m.def("f_get_output", &GetOutput, "GetOutput");
  m.def("f_step_paddle", &StepPaddle, "StepPaddle");
  m.def("f_save_output_dygraph", &SaveOutputDygraph, "SaveOutputDygraph");
}

PYBIND11_MODULE(paddlenlp_ops_90, m) {
  /**
   * all_reduce.cu
   */
  m.def("init_custom_all_reduce", &init_custom_all_reduce, "init all reduce class function");
  m.def("all_reduce", &all_reduce, "all reduce function");
  m.def("dispose", &dispose, "del function for python");
  m.def("meta_size", &meta_size, "meta_size function for Signal struct");
  m.def("register_buffer", &register_buffer, "register ipc buffer");
  m.def("f_append_attention", &AppendAttention, "AppendAttention");
  m.def("f_fused_rotary_position_encoding", &FusedRotaryPositionEncoding, "FusedRotaryPositionEncoding");
  m.def("f_multi_head_latent_attention", &MultiHeadLatentAttention, "MultiHeadLatentAttention");
  m.def("f_noaux_tc", &NoauxTc, "NoauxTc");
  m.def("f_get_block_shape_and_split_kv_block", &GetBlockShapeAndSplitKVBlock, "GetBlockShapeAndSplitKVBlock");
  m.def("f_prefill_mla_write_cache", &PrefillMLAWriteCacheKernel, "PrefillMLAWriteCacheKernel");
  m.def("f_decode_mla_write_cache", &DecodeMLAWriteCacheKernel, "DecodeMLAWriteCacheKernel");
  m.def("f_get_position_ids_and_mask_encoder_batch", &GetPositionIdsAndMaskEncoderBatch, "GetPositionIdsAndMaskEncoderBatch");
  m.def("f_set_preids_token_penalty_multi_scores", &SetPreidsTokenPenaltyMultiScores, "SetPreidsTokenPenaltyMultiScores");
  m.def("f_update_inputs_v2", &UpdateInputesV2, "UpdateInputesV2");
  m.def("f_rebuild_padding_v2", &RebuildPaddingV2Func, "RebuildPaddingV2Func");
  m.def("f_per_token_group_quant", &PerTokenGroupQuant, "PerTokenGroupQuant");
  m.def("f_per_tensor_quant_fp8", &PerTensorQuantFp8, "PerTensorQuantFp8");
  m.def("f_get_padding_offset_v2", &GetPaddingOffsetV2, "GetPaddingOffsetV2");
  m.def("f_save_output", &SaveOutMmsg, "SaveOutMmsg");
  m.def("f_get_output", &GetOutput, "GetOutput");
  m.def("f_step_paddle", &StepPaddle, "StepPaddle");
  m.def("f_save_output_dygraph", &SaveOutputDygraph, "SaveOutputDygraph");
}