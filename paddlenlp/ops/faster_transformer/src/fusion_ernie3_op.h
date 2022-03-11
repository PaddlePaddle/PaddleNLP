#pragma once

#include <string>
#include <vector>

#include "fastertransformer/ernie3.h"
#include "fastertransformer/open_decoder.h"
#include "fastertransformer/utils/common.h"

#ifdef PADDLE_ON_INFERENCE
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif


std::vector<paddle::Tensor> Ernie3CUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& sharing_self_ln_weight,
    const std::vector<paddle::Tensor>& sharing_self_ln_bias,
    const std::vector<paddle::Tensor>& sharing_self_q_weight,
    const std::vector<paddle::Tensor>& sharing_self_q_bias,
    const std::vector<paddle::Tensor>& sharing_self_k_weight,
    const std::vector<paddle::Tensor>& sharing_self_k_bias,
    const std::vector<paddle::Tensor>& sharing_self_v_weight,
    const std::vector<paddle::Tensor>& sharing_self_v_bias,
    const std::vector<paddle::Tensor>& sharing_self_out_weight,
    const std::vector<paddle::Tensor>& sharing_self_out_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_ln_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_ln_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_inter_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_inter_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_out_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_out_bias,
    const paddle::Tensor& sharing_decoder_ln_weight,
    const paddle::Tensor& sharing_decoder_ln_bias,
    const paddle::Tensor& sharing_to_nlg_weight,
    const std::vector<paddle::Tensor>& nlg_self_ln_weight,
    const std::vector<paddle::Tensor>& nlg_self_ln_bias,
    const std::vector<paddle::Tensor>& nlg_self_q_weight,
    const std::vector<paddle::Tensor>& nlg_self_q_bias,
    const std::vector<paddle::Tensor>& nlg_self_k_weight,
    const std::vector<paddle::Tensor>& nlg_self_k_bias,
    const std::vector<paddle::Tensor>& nlg_self_v_weight,
    const std::vector<paddle::Tensor>& nlg_self_v_bias,
    const std::vector<paddle::Tensor>& nlg_self_out_weight,
    const std::vector<paddle::Tensor>& nlg_self_out_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_ln_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_ln_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_inter_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_inter_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_out_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_out_bias,
    const paddle::Tensor& nlg_decoder_ln_weight,
    const paddle::Tensor& nlg_decoder_ln_bias,
    const paddle::Tensor& trans_weight,
    const paddle::Tensor& trans_bias,
    const paddle::Tensor& lm_ln_weight,
    const paddle::Tensor& lm_ln_bias,
    const paddle::Tensor& lm_out_weight, // embedding_weight
    const paddle::Tensor& lm_out_bias, // embedding_bias, gpt have no
    const paddle::Tensor& positional_embedding_weight,
    paddle::Tensor& output_ids,
    const int topk,
    const float topp,
    const int max_len,
    const int sharing_n_head,
    const int sharing_size_per_head,
    const int sharing_num_layer,
    const int nlg_n_head,
    const int nlg_size_per_head,
    const int nlg_num_layer,
    const int bos_id,
    const int eos_id,
    const float temperature,
    const float repetition_penalty,
    const int min_length,
    const bool use_fp16);
