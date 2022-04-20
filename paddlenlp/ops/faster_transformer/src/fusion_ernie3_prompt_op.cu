#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#include "cublas_handle.h"
#include "fastertransformer/cuda/cub/cub.cuh"
#include "fusion_ernie3_prompt_op.h"
#include "pd_traits.h"

template <paddle::DataType D>
std::vector<paddle::Tensor> ernie3_prompt_kernel(
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
    const std::string& hidden_act,
    cudaStream_t stream) {
    
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  DecodingInitParam<DataType_> decoding_params;
  decoding_params.cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
  decoding_params.cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

  typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
  decoding_params.stream = stream;
  fastertransformer::Allocator<AllocatorType::PD> allocator_(stream);
int beam_width_ = (decoding_strategy == "beam_search" ||
                decoding_strategy == "beam_search_v2" ||
                decoding_strategy == "beam_search_v3")
                    ? beam_size
                    : 1;
int candidate_num_ =
    ("topk_sampling" == decoding_strategy ||
    "topp_sampling" == decoding_strategy || "sampling" == decoding_strategy)
        ? topk
        : 1;
float probability_threshold_ =
    ("topk_sampling" == decoding_strategy ||
    "topp_sampling" == decoding_strategy || "sampling" == decoding_strategy)
        ? topp
        : 0.0;

auto input_ids_dims = input_ids.shape();
int batch_size_ = (decoding_strategy == "beam_search" ||
                    decoding_strategy == "beam_search_v2" ||
                    decoding_strategy == "beam_search_v3")
                        ? input_ids_dims[0] / beam_width_
                        : input_ids_dims[0];
const int memory_max_seq_len = input_ids_dims[1];
const int memory_hidden_dim = n_head * size_per_head;
const int vocab_size = lm_out_bias.shape()[0];
decoding_params.output_ids = output_ids.mutable_data<int>(input_ids.place());
decoding_params.parent_ids = parent_ids.mutable_data<int>(input_ids.place());
decoding_params.sequence_length =
    sequence_length.mutable_data<int>(input_ids.place());
decoding_params.output_scores = output_scores.mutable_data<float>(input_ids.place());

decoding_params.d_start_ids = const_cast<int *>(input_ids.data<int>());
decoding_params.d_attn_mask =
    reinterpret_cast<DataType_*>(const_cast<data_t_ *>(attn_mask.data<data_t_>()));
decoding_params.d_start_lengths = start_length.data<int>();
decoding_params.memory_sequence_length = start_length.data<int>();
decoding_params.pos_ids_extra = pos_ids_extra.data<int>();

if (decoding_strategy == "beam_search" ||
    decoding_strategy == "beam_search_v2" ||
    decoding_strategy == "beam_search_v3") {
    decoding_params.request_batch_size = batch_size_ * beam_width_;
} else if (decoding_strategy == "sampling" ||
            decoding_strategy == "topk_sampling" ||
            decoding_strategy == "topp_sampling") {
    decoding_params.request_batch_size = batch_size_;
}
decoding_params.max_input_len = memory_max_seq_len;
decoding_params.request_input_len = memory_max_seq_len;
decoding_params.request_output_len = max_len;

DecoderInitParam<DataType_>* params =
    new DecoderInitParam<DataType_>[num_layer];
for (int i = 0; i < num_layer; i++) {
    params[i].stream = stream;
    params[i].cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
    params[i].cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

    if (decoding_strategy == "beam_search" ||
        decoding_strategy == "beam_search_v2" ||
        decoding_strategy == "beam_search_v3") {
    params[i].request_batch_size = batch_size_ * beam_width_;
    params[i].request_max_mem_seq_len = memory_max_seq_len;
    } else if (decoding_strategy == "sampling" ||
            decoding_strategy == "topk_sampling" ||
            decoding_strategy == "topp_sampling") {
    params[i].request_batch_size = batch_size_;
    params[i].request_max_mem_seq_len = memory_max_seq_len;
    }

    // self attn
    params[i].self_layernorm.gamma = reinterpret_cast<const DataType_*>(
        self_ln_weight[i].data<data_t_>());
    params[i].self_layernorm.beta = reinterpret_cast<const DataType_*>(
        self_ln_bias[i].data<data_t_>());
    // query
    params[i].self_attention.query_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_q_weight[i].data<data_t_>());
    params[i].self_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_q_bias[i].data<data_t_>());
    // key
    params[i].self_attention.key_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_k_weight[i].data<data_t_>());
    params[i].self_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_k_bias[i].data<data_t_>());
    // value
    params[i].self_attention.value_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_v_weight[i].data<data_t_>());
    params[i].self_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_v_bias[i].data<data_t_>());
    // out proj
    params[i].self_attention.attention_output_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_out_weight[i].data<data_t_>());
    params[i].self_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_out_bias[i].data<data_t_>());

    // ffn
    params[i].ffn_layernorm.gamma = reinterpret_cast<const DataType_*>(
        ffn_ln_weight[i].data<data_t_>());
    params[i].ffn_layernorm.beta = reinterpret_cast<const DataType_*>(
        ffn_ln_bias[i].data<data_t_>());
    // intermediate proj
    params[i].ffn.intermediate_weight.kernel =
        reinterpret_cast<const DataType_*>(
            ffn_inter_weight[i].data<data_t_>());
    params[i].ffn.intermediate_weight.bias = reinterpret_cast<const DataType_*>(
        ffn_inter_bias[i].data<data_t_>());
    // out proj
    params[i].ffn.output_weight.kernel = reinterpret_cast<const DataType_*>(
        ffn_out_weight[i].data<data_t_>());
    params[i].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_out_bias[i].data<data_t_>());
}  
decoding_params.layernorm.gamma = reinterpret_cast<const DataType_*>(
    decoder_ln_weight.data<data_t_>());
decoding_params.layernorm.beta = reinterpret_cast<const DataType_*>(
    decoder_ln_bias.data<data_t_>());
decoding_params.trans_kernel =
    reinterpret_cast<const DataType_*>(trans_weight.data<data_t_>());
decoding_params.trans_bias =
    reinterpret_cast<const DataType_*>(trans_bias.data<data_t_>());

decoding_params.lm_layernorm.gamma =
    reinterpret_cast<const DataType_*>(lm_ln_weight.data<data_t_>());
decoding_params.lm_layernorm.beta =
    reinterpret_cast<const DataType_*>(lm_ln_bias.data<data_t_>());

// For embedding
decoding_params.embedding_table =
    reinterpret_cast<const DataType_*>(word_embedding.data<data_t_>());
// For weight sharing matmul
decoding_params.embedding_kernel =
    reinterpret_cast<const DataType_*>(lm_out_weight.data<data_t_>());
// For matmul bias
decoding_params.embedding_bias =
    reinterpret_cast<const DataType_*>(lm_out_bias.data<data_t_>());
decoding_params.position_encoding_table = reinterpret_cast<const DataType_*>(
    positional_embedding_weight.data<data_t_>());

// For masking some id during gen.
decoding_params.logits_mask =
    reinterpret_cast<const DataType_*>(logits_mask.data<data_t_>());

decoding_params.pos_extra_table =
    reinterpret_cast<const DataType_*>(positional_extra_embedding_weight.data<data_t_>());

decoding_params.position_ids = position_ids.data<int>();
decoding_params.decoder_position_ids = decoder_position_ids.data<int>();

  ActivationType activate =
      (hidden_act == "gelu") ? ActivationType::GELU : ActivationType::RELU;

int finished_candidate_num_ =
    ("beam_search_v3" == decoding_strategy) ? beam_width_ : beam_width_ * 2;

if ("beam_search" == decoding_strategy) {
    DecodingBeamsearch<DecodingTraits_::OpType>* ernie3_decoding_beam_search_;

    ernie3_decoding_beam_search_ =
        new DecodingBeamsearch<DecodingTraits_::OpType>(
            allocator_,
            batch_size_,
            beam_width_,
            max_len,
            n_head,
            size_per_head,
            vocab_size,
            num_layer,
            memory_hidden_dim,
            memory_max_seq_len,
            bos_id,
            eos_id,
            beam_search_diversity_rate_,
            true,        /*is_fuse_topk_softMax*/
            true,        /*is_fuse_qkv*/
            false,       /*keep_alive_beam*/
            len_penalty, /*alpha not used for this case*/
            normalize_before,
            0, /*pos_offset BART only for now*/
            activate,
            false, /*pos_bias*/
            true, /*prefix_lm*/
            -1,  /*finished_candidate_num*/
            false,  /*early_stopping*/
            false,  /*is_mbart*/
            min_length,
            4, /*inner_coeff*/
            true); /*is_ernie3_prompt_*/
    ernie3_decoding_beam_search_->forward_context(params, decoding_params);
    ernie3_decoding_beam_search_->forward(params, decoding_params);

    delete ernie3_decoding_beam_search_;
} else if ("beam_search_v2" == decoding_strategy ||
            "beam_search_v3" == decoding_strategy) {
    DecodingBeamsearch<DecodingTraits_::OpType>* ernie3_decoding_beam_search_;

    ernie3_decoding_beam_search_ =
        new DecodingBeamsearch<DecodingTraits_::OpType>(
            allocator_,
            batch_size_,
            beam_width_,
            max_len,
            n_head,
            size_per_head,
            vocab_size,
            num_layer,
            memory_hidden_dim,
            memory_max_seq_len,
            bos_id,
            eos_id,
            beam_search_diversity_rate_,
            true, /*is_fuse_topk_softMax*/
            true, /*is_fuse_qkv*/
            true, /*keep_alive_beam*/
            len_penalty,
            normalize_before,
            0, /*pos_offset BART only for now*/
            activate,
            false, /*pos_bias*/
            true, /*prefix_lm*/
            finished_candidate_num_,
            early_stopping,
            false,  /*is_mbart*/
            min_length,
            4,  /*inner_coeff*/
            true);  /*is_ernie3_prompt_*/
    ernie3_decoding_beam_search_->forward_context(params, decoding_params);
    ernie3_decoding_beam_search_->forward(params, decoding_params);

    delete ernie3_decoding_beam_search_;
} else if ("topk_sampling" == decoding_strategy ||
            "topp_sampling" == decoding_strategy ||
            "sampling" == decoding_strategy) {
    DecodingSampling<DecodingTraits_::OpType>* ernie3_decoding_sampling_;


    ernie3_decoding_sampling_ = new DecodingSampling<DecodingTraits_::OpType>(
        allocator_,
        batch_size_,
        max_len,
        n_head,
        size_per_head,
        vocab_size,
        num_layer,
        memory_hidden_dim,
        memory_max_seq_len,
        bos_id,
        eos_id,
        candidate_num_,
        probability_threshold_,
        true, /*is_fuse_qkv*/
        normalize_before,
        0, /*pos_offset BART only for now*/
        activate,
        false, /*pos_bias*/
        temperature,
        repetition_penalty,  /*repeat_penalty*/
        true, /*prefix_lm*/
        false,  /*is_mbart*/
        min_length,
        4,  /*inner_coeff*/
        true);
    ernie3_decoding_sampling_->forward_context(params, decoding_params);
    ernie3_decoding_sampling_->forward(params, decoding_params);
    delete ernie3_decoding_sampling_;
} else {
    PD_THROW(
        "Only beam_search, beam_search_v2, topk_sampling and topp_sampling are "
        "supported for "
        "FasterTransformer. ");
}
delete[] params;

return {output_ids, parent_ids, sequence_length, output_scores};      
}

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
    const std::string& hidden_act) {
  auto stream = word_embedding.stream();
  cublasSetStream(CublasHandle::GetInstance()->cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  switch (self_ln_weight[0].type()) {
      case paddle::DataType::FLOAT16:{
        ret = ernie3_prompt_kernel<paddle::DataType::FLOAT16>(input_ids,
                                                    attn_mask,
                                                    start_length,
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
                                                    max_len,
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
                                                    hidden_act,
                                                    stream);
        break;
  } 
  case paddle::DataType::FLOAT32: {
        ret = ernie3_prompt_kernel<paddle::DataType::FLOAT32>(input_ids,
                                                    attn_mask,
                                                    start_length,
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
                                                    max_len,
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
                                                    hidden_act,
                                                    stream);
        break;
  }
default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and float32 are supported. ");
      break;
    }
  }
  return ret;
}
