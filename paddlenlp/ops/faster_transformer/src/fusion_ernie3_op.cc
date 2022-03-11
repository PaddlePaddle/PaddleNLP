#include <string>
#include <vector>

#include "fusion_ernie3_op.h"
#include "pd_traits.h"


std::vector<paddle::Tensor> Ernie3Forward(
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
    const bool use_fp16 = false) {
  int batch_size = input.shape()[0];
  int start_len = input.shape()[1];
  int total_len = max_len + start_len;
  std::vector<int64_t> output_dims({total_len, batch_size});
  auto output_ids = paddle::Tensor(input.place(), output_dims);

  if (word_embedding.place() == paddle::PlaceType::kGPU) {
    return Ernie3CUDAForward(input,
                           attn_mask,
                           start_length,
                           word_embedding,
                           sharing_self_ln_weight,
                           sharing_self_ln_bias,
                           sharing_self_q_weight,
                           sharing_self_q_bias,
                           sharing_self_k_weight,
                           sharing_self_k_bias,
                           sharing_self_v_weight,
                           sharing_self_v_bias,
                           sharing_self_out_weight,
                           sharing_self_out_bias,
                           sharing_ffn_ln_weight,
                           sharing_ffn_ln_bias,
                           sharing_ffn_inter_weight,
                           sharing_ffn_inter_bias,
                           sharing_ffn_out_weight,
                           sharing_ffn_out_bias,
                           sharing_decoder_ln_weight,
                           sharing_decoder_ln_bias,
                           sharing_to_nlg_weight,
                           nlg_self_ln_weight,
                           nlg_self_ln_bias,
                           nlg_self_q_weight,
                           nlg_self_q_bias,
                           nlg_self_k_weight,
                           nlg_self_k_bias,
                           nlg_self_v_weight,
                           nlg_self_v_bias,
                           nlg_self_out_weight,
                           nlg_self_out_bias,
                           nlg_ffn_ln_weight,
                           nlg_ffn_ln_bias,
                           nlg_ffn_inter_weight,
                           nlg_ffn_inter_bias,
                           nlg_ffn_out_weight,
                           nlg_ffn_out_bias,
                           nlg_decoder_ln_weight,
                           nlg_decoder_ln_bias,
                           trans_weight,
                           trans_bias,
                           lm_ln_weight,
                           lm_ln_bias,
                           lm_out_weight,
                           lm_out_bias,
                           positional_embedding_weight,
                           output_ids,
                           topk,
                           topp,
                           total_len,
                           sharing_n_head,
                           sharing_size_per_head,
                           sharing_num_layer,
                           nlg_n_head,
                           nlg_size_per_head,
                           nlg_num_layer,
                           bos_id,
                           eos_id,
                           temperature,
                           repetition_penalty,
                           min_length,
                           use_fp16);
  } else {
    PD_THROW("Not implemented place. Only GPU is supported. ");
  }
}

std::vector<std::vector<int64_t>> Ernie3InferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& attn_mask_shape,
    const std::vector<int64_t>& start_length,
    const std::vector<int64_t>& word_embedding_shape,
    const std::vector<std::vector<int64_t>>& sharing_self_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_q_weight_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_q_bias_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_k_weight_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_k_bias_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_v_weight_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_v_bias_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& sharing_self_out_bias_shapes,
    const std::vector<std::vector<int64_t>>& sharing_ffn_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& sharing_ffn_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& sharing_ffn_inter_weight_shapes,
    const std::vector<std::vector<int64_t>>& sharing_ffn_inter_bias_shapes,
    const std::vector<std::vector<int64_t>>& sharing_ffn_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& sharing_ffn_out_bias_shapes,
    const std::vector<int64_t>& sharing_decoder_ln_weight_shape,
    const std::vector<int64_t>& sharing_decoder_ln_bias_shape,
    const std::vector<int64_t>& sharing_to_nlg_weight_shape,
    const std::vector<std::vector<int64_t>>& nlg_self_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_q_weight_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_q_bias_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_k_weight_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_k_bias_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_v_weight_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_v_bias_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& nlg_self_out_bias_shapes,
    const std::vector<std::vector<int64_t>>& nlg_ffn_ln_weight_shapes,
    const std::vector<std::vector<int64_t>>& nlg_ffn_ln_bias_shapes,
    const std::vector<std::vector<int64_t>>& nlg_ffn_inter_weight_shapes,
    const std::vector<std::vector<int64_t>>& nlg_ffn_inter_bias_shapes,
    const std::vector<std::vector<int64_t>>& nlg_ffn_out_weight_shapes,
    const std::vector<std::vector<int64_t>>& nlg_ffn_out_bias_shapes,
    const std::vector<int64_t>& nlg_decoder_ln_weight_shape,
    const std::vector<int64_t>& nlg_decoder_ln_bias_shape,
    const std::vector<int64_t>& trans_weight_shape,
    const std::vector<int64_t>& trans_bias_shape,
    const std::vector<int64_t>& lm_ln_weight_shape,
    const std::vector<int64_t>& lm_ln_bias_shape,
    const std::vector<int64_t>& lm_out_weight_shape,
    const std::vector<int64_t>& lm_out_bias_shape,
    const std::vector<int64_t>& positional_embedding_weight_shape,
    const int& topk,
    const float& topp,
    const int& max_len,
    const int& sharing_n_head,
    const int& sharing_size_per_head,
    const int& sharing_num_layer,
    const int& nlg_n_head,
    const int& nlg_size_per_head,
    const int& nlg_num_layer,
    const int& bos_id,
    const int& eos_id,
    const float& temperature,
    const float& repetition_penalty,
    const int& min_length,
    const bool& use_fp16 = false) {
  int64_t batch_size = input_shape[0];
  int64_t start_len = input_shape[1];
  std::vector<int64_t> output_dims({max_len + start_len, batch_size});
  return {output_dims};
}

std::vector<paddle::DataType> Ernie3InferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& attn_mask_dtype,
    const paddle::DataType& start_length_dtype,
    const paddle::DataType& word_embedding_dtype,
    const std::vector<paddle::DataType>& sharing_self_ln_weight_dtype,
    const std::vector<paddle::DataType>& sharing_self_ln_bias_dtype,
    const std::vector<paddle::DataType>& sharing_self_q_weight_dtype,
    const std::vector<paddle::DataType>& sharing_self_q_bias_dtype,
    const std::vector<paddle::DataType>& sharing_self_k_weight_dtype,
    const std::vector<paddle::DataType>& sharing_self_k_bias_dtype,
    const std::vector<paddle::DataType>& sharing_self_v_weight_dtype,
    const std::vector<paddle::DataType>& sharing_self_v_bias_dtype,
    const std::vector<paddle::DataType>& sharing_self_out_weight_dtype,
    const std::vector<paddle::DataType>& sharing_self_out_bias_dtype,
    const std::vector<paddle::DataType>& sharing_ffn_ln_weight_dtype,
    const std::vector<paddle::DataType>& sharing_ffn_ln_bias_dtype,
    const std::vector<paddle::DataType>& sharing_ffn_inter_weight_dtype,
    const std::vector<paddle::DataType>& sharing_ffn_inter_bias_dtype,
    const std::vector<paddle::DataType>& sharing_ffn_out_weight_dtype,
    const std::vector<paddle::DataType>& sharing_ffn_out_bias_dtype,
    const paddle::DataType& sharing_decoder_ln_weight_dtype,
    const paddle::DataType& sharing_decoder_ln_bias_dtype,
    const paddle::DataType& sharing_to_nlg_weight_dtype,
    const std::vector<paddle::DataType>& nlg_self_ln_weight_dtype,
    const std::vector<paddle::DataType>& nlg_self_ln_bias_dtype,
    const std::vector<paddle::DataType>& nlg_self_q_weight_dtype,
    const std::vector<paddle::DataType>& nlg_self_q_bias_dtype,
    const std::vector<paddle::DataType>& nlg_self_k_weight_dtype,
    const std::vector<paddle::DataType>& nlg_self_k_bias_dtype,
    const std::vector<paddle::DataType>& nlg_self_v_weight_dtype,
    const std::vector<paddle::DataType>& nlg_self_v_bias_dtype,
    const std::vector<paddle::DataType>& nlg_self_out_weight_dtype,
    const std::vector<paddle::DataType>& nlg_self_out_bias_dtype,
    const std::vector<paddle::DataType>& nlg_ffn_ln_weight_dtype,
    const std::vector<paddle::DataType>& nlg_ffn_ln_bias_dtype,
    const std::vector<paddle::DataType>& nlg_ffn_inter_weight_dtype,
    const std::vector<paddle::DataType>& nlg_ffn_inter_bias_dtype,
    const std::vector<paddle::DataType>& nlg_ffn_out_weight_dtype,
    const std::vector<paddle::DataType>& nlg_ffn_out_bias_dtype,
    const paddle::DataType& nlg_decoder_ln_weight_dtype,
    const paddle::DataType& nlg_decoder_ln_bias_dtype,
    const paddle::DataType& trans_weight_dtype,
    const paddle::DataType& trans_bias_dtype,
    const paddle::DataType& lm_ln_weight_dtype,
    const paddle::DataType& lm_ln_bias_dtype,
    const paddle::DataType& lm_out_weight_dtype,
    const paddle::DataType& lm_out_bias_dtype,
    const paddle::DataType& positional_embedding_weight_dtype) {
  return {paddle::DataType::INT32};
}

PD_BUILD_OP(fusion_ernie3)
    .Inputs({"Input",
             "AttentionMask",
             "StartLength",
             "WordEmbedding",
             paddle::Vec("SharingSelfLayernormWeight"),
             paddle::Vec("SharingSelfLayernormBias"),
             paddle::Vec("SharingSelfQueryWeight"),
             paddle::Vec("SharingSelfQueryBias"),
             paddle::Vec("SharingSelfKeyWeight"),
             paddle::Vec("SharingSelfKeyBias"),
             paddle::Vec("SharingSelfValueWeight"),
             paddle::Vec("SharingSelfValueBias"),
             paddle::Vec("SharingSelfOutWeight"),
             paddle::Vec("SharingSelfOutBias"),
             paddle::Vec("SharingFFNLayernormWeight"),
             paddle::Vec("SharingFFNLayernormBias"),
             paddle::Vec("SharingFFNInterWeight"),
             paddle::Vec("SharingFFNInterBias"),
             paddle::Vec("SharingFFNOutWeight"),
             paddle::Vec("SharingFFNOutBias"),
             "SharingDecoderLayernormWeight",
             "SharingDecoderLayernormBias",
             "SharingToNlgWeight",
             paddle::Vec("NlgSelfLayernormWeight"),
             paddle::Vec("NlgSelfLayernormBias"),
             paddle::Vec("NlgSelfQueryWeight"),
             paddle::Vec("NlgSelfQueryBias"),
             paddle::Vec("NlgSelfKeyWeight"),
             paddle::Vec("NlgSelfKeyBias"),
             paddle::Vec("NlgSelfValueWeight"),
             paddle::Vec("NlgSelfValueBias"),
             paddle::Vec("NlgSelfOutWeight"),
             paddle::Vec("NlgSelfOutBias"),
             paddle::Vec("NlgFFNLayernormWeight"),
             paddle::Vec("NlgFFNLayernormBias"),
             paddle::Vec("NlgFFNInterWeight"),
             paddle::Vec("NlgFFNInterBias"),
             paddle::Vec("NlgFFNOutWeight"),
             paddle::Vec("NlgFFNOutBias"),
             "NlgDecoderLayernormWeight",
             "NlgDecoderLayernormBias",
             "TransWeight",
             "TransBias",
             "LmLayernormWeight",
             "LmLayernormBias",
             "LmOutWeight",
             "LmOutBias",
             "PositionEncEmb"})
    .Outputs({"OutputIds"})
    .Attrs({"topk: int",
            "topp: float",
            "max_len: int",
            "sharing_n_head: int",
            "sharing_size_per_head: int",
            "sharing_num_layer: int",
            "nlg_n_head: int",
            "nlg_size_per_head: int",
            "nlg_num_layer: int",
            "bos_id: int",
            "eos_id: int",
            "temperature: float",
            "repetition_penalty: float",
            "min_length: int",
            "use_fp16: bool"})
    .SetKernelFn(PD_KERNEL(Ernie3Forward))
    .SetInferShapeFn(PD_INFER_SHAPE(Ernie3InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Ernie3InferDtype));
