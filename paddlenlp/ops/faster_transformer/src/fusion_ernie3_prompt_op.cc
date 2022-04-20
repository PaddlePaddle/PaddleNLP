#include <string>
#include <vector>

#include "fusion_ernie3_prompt_op.h"
#include "pd_traits.h"

std::vector<paddle::Tensor> Ernie3PromptForward(
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
    const bool rel_len,
    const bool normalize_before,
    const std::string& hidden_act) {
  int batch_size = input_ids.shape()[0];
  int max_out_len = rel_len ? max_len + input_ids.shape()[1] : max_len;

  std::vector<int64_t> output_ids_dims;
  std::vector<int64_t> output_scores_dims;
  std::vector<int64_t> parent_ids_dims;
  std::vector<int64_t> sequence_length_dims({batch_size});
  if (decoding_strategy == "beam_search") {
    if (batch_size != -1) {
      batch_size /= beam_size;
    }
    output_ids_dims = {max_out_len, batch_size, beam_size};
    output_scores_dims = {batch_size, beam_size};
    parent_ids_dims = output_ids_dims;
  } else if (decoding_strategy == "beam_search_v2" ||
             decoding_strategy == "beam_search_v3") {
    // Use separated alive and finish beam queues to avoid the decrease of alive
    // beams. The outputs must include both the finish and alive to trace full
    // path.
    if (batch_size != -1) {
      sequence_length_dims = {batch_size * 2};
      batch_size /= beam_size;
    } else {
      sequence_length_dims = {batch_size};
    }
    output_ids_dims = {max_out_len, batch_size, beam_size * 2};
    output_scores_dims = {batch_size, beam_size * 2};
    parent_ids_dims = output_ids_dims;
  } else if (decoding_strategy == "topk_sampling" ||
             decoding_strategy == "topp_sampling" ||
             decoding_strategy == "sampling") {
    output_ids_dims = {max_out_len, batch_size};
    output_scores_dims = {batch_size};
    parent_ids_dims = {1};
  } else {
    PD_THROW("Not supported decoding strategy. ");
  }
  auto output_ids = paddle::Tensor(input_ids.place(), output_ids_dims);
  auto parent_ids = paddle::Tensor(input_ids.place(), parent_ids_dims);
  auto sequence_length =
      paddle::Tensor(input_ids.place(), sequence_length_dims);
  auto output_scores = paddle::Tensor(input_ids.place(), output_scores_dims);

  if (input_ids.place() == paddle::PlaceType::kGPU) {
    auto start_len = paddle::Tensor(paddle::PlaceType::kGPU);
    if (start_length.place() != paddle::PlaceType::kGPU) {
      start_len = start_length.copy_to<int>(paddle::PlaceType::kGPU);
    } else {
      start_len = start_length;
    }
    return Ernie3PromptCUDAForward(input_ids,
                            attn_mask,
                            start_len,
                            word_embedding,
                            self_ln_weight,
                            self_ln_bias,
                            self_q_weight,
                            self_q_bias,
                            self_k_weight,
                            self_k_bias,
                            self_v_weight,
                            self_v_bias,
                            self_out_weight,
                            self_out_bias,
                            ffn_ln_weight,
                            ffn_ln_bias,
                            ffn_inter_weight,
                            ffn_inter_bias,
                            ffn_out_weight,
                            ffn_out_bias,
                            decoder_ln_weight,
                            decoder_ln_bias,
                            trans_weight,
                            trans_bias,
                            lm_ln_weight,
                            lm_ln_bias,
                            lm_out_weight,
                            lm_out_bias,
                            position_ids,
                            positional_embedding_weight,
                            pos_ids_extra,
                            positional_extra_embedding_weight,
                            logits_mask,
                            decoder_position_ids,
                            output_ids,
                            parent_ids,
                            sequence_length,
                            output_scores,
                            decoding_strategy,
                            beam_size,
                            topk,
                            topp,
                            max_out_len,
                            n_head,
                            size_per_head,
                            num_layer,
                            bos_id,
                            eos_id,
                            temperature,
                            repetition_penalty,
                            len_penalty,
                            beam_search_diversity_rate_,
                            early_stopping,
                            min_length,
                            normalize_before,
                            hidden_act);
  } else {
    PD_THROW("Not implemented place. Only GPU is supported. ");
  }
}

std::vector<std::vector<int64_t>> Ernie3PromptInferShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& attn_mask_shape,
    const std::vector<int64_t>& start_length,
    const std::vector<int64_t>& word_embedding_shape,
    const std::vector<std::vector<int64_t>>& self_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_q_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_q_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_k_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_k_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_v_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_v_bias_shapes,
    const std::vector<std::vector<int64_t>>& self_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& self_out_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_inter_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_inter_bias_shapes,
    const std::vector<std::vector<int64_t>>& ffn_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& ffn_out_bias_shapes,
    const std::vector<int64_t>& decoder_ln_weight_shape,
    const std::vector<int64_t>& decoder_ln_bias_shape,
    const std::vector<int64_t>& trans_weight_shape,
    const std::vector<int64_t>& trans_bias_shape,
    const std::vector<int64_t>& lm_ln_weight_shape,
    const std::vector<int64_t>& lm_ln_bias_shape,
    const std::vector<int64_t>& lm_out_weight_shape,
    const std::vector<int64_t>& lm_out_bias_shape,
    const std::vector<int64_t>& position_ids_shape,
    const std::vector<int64_t>& positional_embedding_weight_shape,
    const std::vector<int64_t>& pos_ids_extra_shape,
    const std::vector<int64_t>& positional_extra_embedding_weight_shape,
    const std::vector<int64_t>& logits_mask_shape,
    const std::vector<int64_t>& decoder_position_ids_shape,
    const std::string& decoding_strategy,
    const int& beam_size,
    const int& topk,
    const float& topp,
    const int& max_len,
    const int& n_head,
    const int& size_per_head,
    const int& num_layer,
    const int& bos_id,
    const int& eos_id,
    const float& temperature,
    const float& repetition_penalty,
    const float& len_penalty,
    const float& beam_search_diversity_rate_,
    const bool& early_stopping,
    const int& min_length,
    const bool& rel_len,
    const bool& normalize_before,
    const std::string& hidden_act){
    int batch_size = input_ids_shape[0];

    std::vector<int64_t> output_ids_dims;
    std::vector<int64_t> output_scores_dims;
    std::vector<int64_t> sequence_length_dims({batch_size});
    if (decoding_strategy == "beam_search") {
      if (batch_size != -1) {
        batch_size /= beam_size;
      }
      output_ids_dims = {max_len, batch_size, beam_size};
      output_scores_dims = {batch_size, beam_size};
      return {output_ids_dims, output_ids_dims, sequence_length_dims, output_scores_dims};
    } else if (decoding_strategy == "beam_search_v2" ||
              decoding_strategy == "beam_search_v3") {
      // Use separated alive and finish beam queues to avoid the decrease of alive
      // beams. The outputs must include both the finish and alive to trace full
      // path.
      if (batch_size != -1) {
        sequence_length_dims = {batch_size * 2};
        batch_size /= beam_size;
      } else {
        sequence_length_dims = {batch_size};
      }
      output_ids_dims = {max_len, batch_size, beam_size * 2};
      output_scores_dims = {batch_size, beam_size * 2};
      return {output_ids_dims, output_ids_dims, sequence_length_dims, output_scores_dims};
    } else if (decoding_strategy == "topk_sampling" ||
              decoding_strategy == "topp_sampling" ||
              decoding_strategy == "sampling") {
      output_ids_dims = {max_len, batch_size};
      output_scores_dims = {batch_size};
      return {output_ids_dims, {1}, sequence_length_dims, output_scores_dims};
    } else {
      PD_THROW("Not supported decoding strategy. ");
    }
}

std::vector<paddle::DataType> Ernie3PromptInferDtype(
    const paddle::DataType& input_ids,
    const paddle::DataType& attn_mask,
    const paddle::DataType& start_length,
    const paddle::DataType& word_embedding,
    const std::vector<paddle::DataType>& self_ln_weight,
    const std::vector<paddle::DataType>& self_ln_bias,
    const std::vector<paddle::DataType>& self_q_weight,
    const std::vector<paddle::DataType>& self_q_bias,
    const std::vector<paddle::DataType>& self_k_weight,
    const std::vector<paddle::DataType>& self_k_bias,
    const std::vector<paddle::DataType>& self_v_weight,
    const std::vector<paddle::DataType>& self_v_bias,
    const std::vector<paddle::DataType>& self_out_weight,
    const std::vector<paddle::DataType>& self_out_bias,
    const std::vector<paddle::DataType>& ffn_ln_weight,
    const std::vector<paddle::DataType>& ffn_ln_bias,
    const std::vector<paddle::DataType>& ffn_inter_weight,
    const std::vector<paddle::DataType>& ffn_inter_bias,
    const std::vector<paddle::DataType>& ffn_out_weight,
    const std::vector<paddle::DataType>& ffn_out_bias,
    const paddle::DataType& decoder_ln_weight,
    const paddle::DataType& decoder_ln_bias,
    const paddle::DataType& trans_weight,
    const paddle::DataType& trans_bias,
    const paddle::DataType& lm_ln_weight,
    const paddle::DataType& lm_ln_bias,
    const paddle::DataType& lm_out_weight,
    const paddle::DataType& lm_out_bias,
    const paddle::DataType& position_ids,
    const paddle::DataType& positional_embedding_weight,
    const paddle::DataType& pos_ids_extra,
    const paddle::DataType& positional_extra_embedding_weight,
    const paddle::DataType& logits_mask,
    const paddle::DataType& decoder_position_ids){
  return {paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::FLOAT32};
}

PD_BUILD_OP(fusion_ernie3_prompt)
    .Inputs({"Input",
             "AttentionMask",
             "StartLength",
             "WordEmbedding",
             paddle::Vec("SelfLayernormWeight"),
             paddle::Vec("SelfLayernormBias"),
             paddle::Vec("SelfQueryWeight"),
             paddle::Vec("SelfQueryBias"),
             paddle::Vec("SelfKeyWeight"),
             paddle::Vec("SelfKeyBias"),
             paddle::Vec("SelfValueWeight"),
             paddle::Vec("SelfValueBias"),
             paddle::Vec("SelfOutWeight"),
             paddle::Vec("SelfOutBias"),
             paddle::Vec("FFNLayernormWeight"),
             paddle::Vec("FFNLayernormBias"),
             paddle::Vec("FFNInterWeight"),
             paddle::Vec("FFNInterBias"),
             paddle::Vec("FFNOutWeight"),
             paddle::Vec("FFNOutBias"),
             "DecoderLayernormWeight",
             "DecoderLayernormBias",
             "TransWeight",
             "TransBias",
             "LmLayernormWeight",
             "LmLayernormBias",
             "LmOutWeight",
             "LmOutBias",
             "PositionIds",
             "PositionEncEmb",
             "PositionExtraIds",
             "PositionExtraEncEmb",
             "LogitsMask",
             "DecPositionIds"})
    .Outputs({"OutputIds", "ParentIds", "SequenceLength", "OutputScores"})
    .Attrs({"decoding_strategy: std::string",
            "beam_size: int",
            "topk: int",
            "topp: float",
            "max_len: int",
            "n_head: int",
            "size_per_head: int",
            "num_layer: int",
            "bos_id: int",
            "eos_id: int",
            "temperature: float",
            "repetition_penalty: float",
            "len_penalty: float",
            "beam_search_diversity_rate: float",
            "early_stopping: bool",
            "min_length: int",
            "rel_len: bool",
            "normalize_before: bool",
            "hidden_act: std::string"})
    .SetKernelFn(PD_KERNEL(Ernie3PromptForward))
    .SetInferShapeFn(PD_INFER_SHAPE(Ernie3PromptInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Ernie3PromptInferDtype));
