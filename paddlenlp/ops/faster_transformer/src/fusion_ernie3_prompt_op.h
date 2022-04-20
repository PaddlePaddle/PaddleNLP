#pragma once

#include <string>
#include <vector>

#include "fastertransformer/decoding_beamsearch.h"
#include "fastertransformer/decoding_sampling.h"
#include "fastertransformer/open_decoder.h"
#include "fastertransformer/utils/common.h"

#ifdef PADDLE_ON_INFERENCE
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif

std::vector<paddle::Tensor> Ernie3PromptCUDAForward(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_q_bias,
    const std::vector<paddle::Tensor>& self_k_weight,
    const std::vector<paddle::Tensor>& self_k_bias,
    const std::vector<paddle::Tensor>& self_v_weight,
    const std::vector<paddle::Tensor>& self_v_bias,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& self_out_bias,
    const std::vector<paddle::Tensor>& ffn_ln_weight,
    const std::vector<paddle::Tensor>& ffn_ln_bias,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& trans_weight,
    const paddle::Tensor& trans_bias,
    const paddle::Tensor& lm_ln_weight,
    const paddle::Tensor& lm_ln_bias,
    const paddle::Tensor& lm_out_weight, // embedding_weight
    const paddle::Tensor& lm_out_bias, // embedding_bias, gpt have no
    const paddle::Tensor& position_ids,
    const paddle::Tensor& positional_embedding_weight,
    const paddle::Tensor& pos_ids_extra,
    const paddle::Tensor& positional_extra_embedding_weight,
    const paddle::Tensor& logits_mask,
    const paddle::Tensor& decoder_position_ids,
    paddle::Tensor& output_ids,
    paddle::Tensor& parent_ids,
    paddle::Tensor& sequence_length,
    paddle::Tensor& output_scores,
    const std::string& decoding_strategy,
    const int beam_size,
    const int topk,
    const float topp,
    const int max_len,
    const int n_head,
    const int size_per_head,
    const int num_layer,
    const int bos_id,
    const int eos_id,
    const float temperature,
    const float repetition_penalty,
    const float len_penalty,
    const float beam_search_diversity_rate_,
    const bool early_stopping,
    const int min_length,
    const bool normalize_before,
    const std::string& hidden_act);
