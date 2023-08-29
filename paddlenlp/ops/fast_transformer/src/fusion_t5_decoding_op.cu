/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>


#include "fusion_t5_decoding_op.h"
#include "pd_traits.h"

template <paddle::DataType D>
std::vector<paddle::Tensor> t5_decoding_kernel(
    const paddle::Tensor& input,
    const paddle::Tensor& memory_sequence_length,
    const paddle::Tensor& word_emb,
    const std::vector<paddle::Tensor>& self_layernorm_weight,
    const std::vector<paddle::Tensor>& self_layernorm_bias,
    const std::vector<paddle::Tensor>& self_attn_query_weight,
    const std::vector<paddle::Tensor>& self_attn_query_bias,
    const std::vector<paddle::Tensor>& self_attn_key_weight,
    const std::vector<paddle::Tensor>& self_attn_key_bias,
    const std::vector<paddle::Tensor>& self_attn_value_weight,
    const std::vector<paddle::Tensor>& self_attn_value_bias,
    const std::vector<paddle::Tensor>& self_attn_output_weight,
    const std::vector<paddle::Tensor>& self_attn_output_bias,
    const std::vector<paddle::Tensor>& cross_layernorm_weight,
    const std::vector<paddle::Tensor>& cross_layernorm_bias,
    const std::vector<paddle::Tensor>& cross_attn_query_weight,
    const std::vector<paddle::Tensor>& cross_attn_query_bias,
    const std::vector<paddle::Tensor>& cross_attn_key_weight,
    const std::vector<paddle::Tensor>& cross_attn_key_bias,
    const std::vector<paddle::Tensor>& cross_attn_value_weight,
    const std::vector<paddle::Tensor>& cross_attn_value_bias,
    const std::vector<paddle::Tensor>& cross_attn_output_weight,
    const std::vector<paddle::Tensor>& cross_attn_output_bias,
    const std::vector<paddle::Tensor>& ffn_layernorm_weight,
    const std::vector<paddle::Tensor>& ffn_layernorm_bias,
    const std::vector<paddle::Tensor>& ffn_intermediate_weight_0,
    const std::vector<paddle::Tensor>& ffn_intermediate_bias_0,
    const std::vector<paddle::Tensor>& ffn_intermediate_weight_1,
    const std::vector<paddle::Tensor>& ffn_intermediate_bias_1,
    const std::vector<paddle::Tensor>& ffn_output_weight,
    const std::vector<paddle::Tensor>& ffn_output_bias,
    const paddle::Tensor& self_relative_attention_bias_weight,
    const paddle::Tensor& decoder_layernorm_weight,
    const paddle::Tensor& decoder_layernorm_bias,
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    paddle::Tensor& output_ids,
    paddle::Tensor& parent_ids,
    paddle::Tensor& sequence_length,
    const std::string& decoding_strategy,
    const int& beam_size,
    const int& topk,
    const float& topp,
    const int& head_num_,
    const int& size_per_head_,
    const int& num_layer_,
    const int& start_id_,
    const int& end_id_,
    const int64_t& max_seq_len_,
    const float& beam_search_diversity_rate_,
    const float& alpha,
    const float& temperature,
    const bool& early_stopping,
    const int& max_distance,
    const int& num_buckets,
    const bool& tie_word_embeddings,
    const std::string& act,
    cudaStream_t stream) {
  int beam_width_ = (decoding_strategy == "beam_search" ||
                     decoding_strategy == "beam_search_v2" ||
                     decoding_strategy == "beam_search_v3")
                        ? beam_size
                        : 1;
  int candidate_num_ =
      (decoding_strategy == "topk_sampling" ||
       decoding_strategy == "topp_sampling" || decoding_strategy == "sampling")
          ? topk
          : 1;
  float probability_threshold_ =
      (decoding_strategy == "topk_sampling" ||
       decoding_strategy == "topp_sampling" || decoding_strategy == "sampling")
          ? topp
          : 0.0;

  auto input_dims = input.shape();
  int batch_size_ = (decoding_strategy == "beam_search" ||
                     decoding_strategy == "beam_search_v2" ||
                     decoding_strategy == "beam_search_v3")
                        ? input_dims[0] / beam_width_
                        : input_dims[0];
  const int memory_max_seq_len = input_dims[1];
  const int memory_hidden_dim = input_dims[2];
  const int vocab_size = word_emb.shape()[0];

  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  DecodingInitParam<DataType_> decoding_params;
  decoding_params.cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
  decoding_params.cublaslt_handle =
      CublasHandle::GetInstance()->cublaslt_handle_;

  decoding_params.output_ids = output_ids.mutable_data<int>(input.place());
  decoding_params.parent_ids = parent_ids.mutable_data<int>(input.place());
  decoding_params.sequence_length =
      sequence_length.mutable_data<int>(input.place());

  typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
  decoding_params.stream = stream;
  fastertransformer::Allocator<AllocatorType::PD> allocator_(stream);

  decoding_params.memory_tensor =
      reinterpret_cast<const DataType_*>(input.data<data_t_>());
  decoding_params.memory_sequence_length = memory_sequence_length.data<int>();

  DecoderInitParam<DataType_>* params =
      new DecoderInitParam<DataType_>[num_layer_];

  int inner_coeff = ffn_intermediate_weight_0[0].shape()[1] / memory_hidden_dim;
  int inner_size = ffn_intermediate_weight_0[0].shape()[1];

  auto q_weight_shape = self_attn_query_weight[0].shape();
  auto k_weight_shape = self_attn_key_weight[0].shape();

  if (decoding_strategy == "beam_search" ||
      decoding_strategy == "beam_search_v2" ||
      decoding_strategy == "beam_search_v3") {
    decoding_params.request_batch_size = batch_size_ * beam_width_;
  } else if (decoding_strategy == "sampling" ||
             decoding_strategy == "topk_sampling" ||
             decoding_strategy == "topp_sampling") {
    decoding_params.request_batch_size = batch_size_;
  }

  bool use_gated = false;

  for (int i = 0; i < num_layer_; i++) {
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
        self_layernorm_weight[i].data<data_t_>());

    if (self_layernorm_bias[i].shape()[0] != 1) {
      params[i].self_layernorm.beta = reinterpret_cast<const DataType_*>(
          self_layernorm_bias[i].data<data_t_>());
    } else {
      params[i].self_layernorm.beta = nullptr;
    }

    // query
    params[i].self_attention.query_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_attn_query_weight[i].data<data_t_>());
    params[i].self_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_query_bias[i].data<data_t_>());
    // key
    params[i].self_attention.key_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_attn_key_weight[i].data<data_t_>());
    params[i].self_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_key_bias[i].data<data_t_>());
    // value
    params[i].self_attention.value_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_attn_value_weight[i].data<data_t_>());
    params[i].self_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_value_bias[i].data<data_t_>());
    // out proj
    params[i].self_attention.attention_output_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_attn_output_weight[i].data<data_t_>());
    params[i].self_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_output_bias[i].data<data_t_>());

    // cross
    params[i].cross_layernorm.gamma = reinterpret_cast<const DataType_*>(
        cross_layernorm_weight[i].data<data_t_>());
    if (cross_layernorm_bias[i].shape()[0] != 1) {
      params[i].cross_layernorm.beta = reinterpret_cast<const DataType_*>(
          cross_layernorm_bias[i].data<data_t_>());
    } else {
      params[i].cross_layernorm.beta = nullptr;
    }
    // query
    params[i].cross_attention.query_weight.kernel =
        reinterpret_cast<const DataType_*>(
            cross_attn_query_weight[i].data<data_t_>());
    params[i].cross_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(
            cross_attn_query_bias[i].data<data_t_>());
    // key
    params[i].cross_attention.key_weight.kernel =
        reinterpret_cast<const DataType_*>(
            cross_attn_key_weight[i].data<data_t_>());
    params[i].cross_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(
            cross_attn_key_bias[i].data<data_t_>());
    // value
    params[i].cross_attention.value_weight.kernel =
        reinterpret_cast<const DataType_*>(
            cross_attn_value_weight[i].data<data_t_>());
    params[i].cross_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(
            cross_attn_value_bias[i].data<data_t_>());
    // out proj
    params[i].cross_attention.attention_output_weight.kernel =
        reinterpret_cast<const DataType_*>(
            cross_attn_output_weight[i].data<data_t_>());
    params[i].cross_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(
            cross_attn_output_bias[i].data<data_t_>());

    // ffn
    params[i].ffn_layernorm.gamma = reinterpret_cast<const DataType_*>(
        ffn_layernorm_weight[i].data<data_t_>());
    if (ffn_layernorm_bias[i].shape()[0] != 1) {
      params[i].ffn_layernorm.beta = reinterpret_cast<const DataType_*>(
          ffn_layernorm_bias[i].data<data_t_>());
    } else {
      params[i].ffn_layernorm.beta = nullptr;
    }
    // intermediate proj
    params[i].ffn.intermediate_weight.kernel =
        reinterpret_cast<const DataType_*>(
            ffn_intermediate_weight_0[i].data<data_t_>());
    params[i].ffn.intermediate_weight.bias = reinterpret_cast<const DataType_*>(
        ffn_intermediate_bias_0[i].data<data_t_>());

    if (ffn_intermediate_weight_1[i].shape()[0] != 1) {
        use_gated = true;
        params[i].ffn.intermediate_weight_1.kernel =
            reinterpret_cast<const DataType_*>(
                ffn_intermediate_weight_1[i].data<data_t_>());
        params[i].ffn.intermediate_weight_1.bias = reinterpret_cast<const DataType_*>(
            ffn_intermediate_bias_1[i].data<data_t_>());
    } else {
        params[i].ffn.intermediate_weight_1.kernel = nullptr;
        params[i].ffn.intermediate_weight_1.bias = nullptr;
    }
    
    // out proj
    params[i].ffn.output_weight.kernel = reinterpret_cast<const DataType_*>(
        ffn_output_weight[i].data<data_t_>());
    params[i].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_output_bias[i].data<data_t_>());
  }

  // relative bias
  decoding_params.self_relative_attention_bias_weight =
      reinterpret_cast<const DataType_*>(
          self_relative_attention_bias_weight.data<data_t_>());

  decoding_params.layernorm.gamma = reinterpret_cast<const DataType_*>(
      decoder_layernorm_weight.data<data_t_>());
  if (decoder_layernorm_bias.shape()[0] != 1) {
    decoding_params.layernorm.beta = reinterpret_cast<const DataType_*>(
        decoder_layernorm_bias.data<data_t_>());
  } else {
    decoding_params.layernorm.beta = nullptr;
  }
  // for embedding
  decoding_params.embedding_table =
      reinterpret_cast<const DataType_*>(word_emb.data<data_t_>());

  // for weight sharing matmul
  decoding_params.embedding_kernel =
      reinterpret_cast<const DataType_*>(embedding_weight.data<data_t_>());
  // for matmul bias
  decoding_params.embedding_bias =
      reinterpret_cast<const DataType_*>(embedding_bias.data<data_t_>());

  int finished_candidate_num_ =
      ("beam_search_v3" == decoding_strategy) ? beam_width_ : beam_width_ * 2;

  ActivationType activate =
      (act == "gelu") ? ActivationType::GELU : ActivationType::RELU;

  if ("beam_search" == decoding_strategy) {
    T5DecodingBeamsearch<DecodingTraits_::OpType>* decoding_beam_search_;
    decoding_beam_search_ = new T5DecodingBeamsearch<DecodingTraits_::OpType>(
        allocator_,
        batch_size_,
        beam_width_,
        max_seq_len_,
        head_num_,
        size_per_head_,
        vocab_size,
        num_layer_,
        memory_hidden_dim,
        memory_max_seq_len,
        start_id_,
        end_id_,
        beam_search_diversity_rate_,
        true,  // is_fuse_topk_softMax
        true,  // fuse_qkv
        false,                 // keep_alive_beam
        0.6,                   // alpha
        true,                  // normalization_before
        activate,
        -1,                    // finished_candidate_num
        false,                 // early_stopping
        0,                     // min_length
        inner_coeff,
        inner_size,
        num_buckets,
        max_distance,
        tie_word_embeddings,
        use_gated);

    decoding_beam_search_->forward(params, decoding_params);

    delete decoding_beam_search_;
  } else if ("beam_search_v2" == decoding_strategy ||
             "beam_search_v3" == decoding_strategy) {
    T5DecodingBeamsearch<DecodingTraits_::OpType>* decoding_beam_search_;
    decoding_beam_search_ = new T5DecodingBeamsearch<DecodingTraits_::OpType>(
        allocator_,
        batch_size_,
        beam_width_,
        max_seq_len_,
        head_num_,
        size_per_head_,
        vocab_size,
        num_layer_,
        memory_hidden_dim,
        memory_max_seq_len,
        start_id_,
        end_id_,
        beam_search_diversity_rate_,
        true,  // is_fuse_topk_softMax
        true,  // fuse_qkv
        true,  // keep_alive_beam
        alpha,
        true,                     // normalization_before
        activate,
        finished_candidate_num_,
        early_stopping,
        0,                        // min_length
        inner_coeff,
        inner_size,
        num_buckets,
        max_distance,
        tie_word_embeddings,
        use_gated);

    decoding_beam_search_->forward(params, decoding_params);

    delete decoding_beam_search_;
  } else if ("topk_sampling" == decoding_strategy ||
             "topp_sampling" == decoding_strategy ||
             "sampling" == decoding_strategy) {

    T5DecodingSampling<DecodingTraits_::OpType>* decoding_sampling_;
    decoding_sampling_ = new T5DecodingSampling<DecodingTraits_::OpType>(
        allocator_,
        batch_size_,
        max_seq_len_,
        head_num_,
        size_per_head_,
        vocab_size,
        num_layer_,
        memory_hidden_dim,
        memory_max_seq_len,
        start_id_,
        end_id_,
        candidate_num_,
        probability_threshold_,
        true,  // fuse_qkv
        true,                  // normalization_before
        activate,
        1.0,                   // temperature
        1.0,                   // repeat_penalty
        0,                     // min_length
        inner_coeff,
        inner_size,
        -1,  // seed
        1,  // tensor_para_size
        1,  // layer_para_size
        num_buckets,
        max_distance,
        tie_word_embeddings,
        use_gated);

    decoding_sampling_->forward(params, decoding_params);

    delete decoding_sampling_;

  } else {
    PD_THROW(
        "Only beam_search, topk_sampling and topp_sampling are supported for "
        "FastGeneration. ");
  }
  delete[] params;

  return {output_ids, parent_ids, sequence_length};
}

std::vector<paddle::Tensor> T5DecodingCUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& mem_seq_len,
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
    const std::vector<paddle::Tensor>& cross_ln_weight,
    const std::vector<paddle::Tensor>& cross_ln_bias,
    const std::vector<paddle::Tensor>& cross_q_weight,
    const std::vector<paddle::Tensor>& cross_q_bias,
    const std::vector<paddle::Tensor>& cross_k_weight,
    const std::vector<paddle::Tensor>& cross_k_bias,
    const std::vector<paddle::Tensor>& cross_v_weight,
    const std::vector<paddle::Tensor>& cross_v_bias,
    const std::vector<paddle::Tensor>& cross_out_weight,
    const std::vector<paddle::Tensor>& cross_out_bias,
    const std::vector<paddle::Tensor>& ffn_ln_weight,
    const std::vector<paddle::Tensor>& ffn_ln_bias,
    const std::vector<paddle::Tensor>& ffn_inter_weight_0,
    const std::vector<paddle::Tensor>& ffn_inter_bias_0,
    const std::vector<paddle::Tensor>& ffn_inter_weight_1,
    const std::vector<paddle::Tensor>& ffn_inter_bias_1,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& self_relative_attention_bias_weight,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    paddle::Tensor& output_ids,
    paddle::Tensor& parent_ids,
    paddle::Tensor& sequence_length,
    const std::string& decoding_strategy,
    const int& beam_size,
    const int& topk,
    const float& topp,
    const int& n_head,
    const int& size_per_head,
    const int& num_layer,
    const int& bos_id,
    const int& eos_id,
    const int64_t& max_len,
    const float& beam_search_diversity_rate,
    const float& alpha,
    const float& temperature,
    const bool& early_stopping,
    const int& max_distance,
    const int& num_buckets,
    const bool& tie_word_embeddings,
    const std::string& act) {
  auto stream = input.stream();

  cublasSetStream(CublasHandle::GetInstance()->cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  switch (input.type()) {
    case paddle::DataType::FLOAT16: {
      ret = t5_decoding_kernel<paddle::DataType::FLOAT16>(
          input,
          mem_seq_len,
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
          cross_ln_weight,
          cross_ln_bias,
          cross_q_weight,
          cross_q_bias,
          cross_k_weight,
          cross_k_bias,
          cross_v_weight,
          cross_v_bias,
          cross_out_weight,
          cross_out_bias,
          ffn_ln_weight,
          ffn_ln_bias,
          ffn_inter_weight_0,
          ffn_inter_bias_0,
          ffn_inter_weight_1,
          ffn_inter_bias_1,
          ffn_out_weight,
          ffn_out_bias,
          self_relative_attention_bias_weight,
          decoder_ln_weight,
          decoder_ln_bias,
          embedding_weight,
          embedding_bias,
          output_ids,
          parent_ids,
          sequence_length,
          decoding_strategy,
          beam_size,
          topk,
          topp,
          n_head,
          size_per_head,
          num_layer,
          bos_id,
          eos_id,
          max_len,
          beam_search_diversity_rate,
          alpha,
          temperature,
          early_stopping,
          max_distance,
          num_buckets,
          tie_word_embeddings,
          act,
          stream);
      break;
    }
    case paddle::DataType::FLOAT32: {
      ret = t5_decoding_kernel<paddle::DataType::FLOAT32>(
          input,
          mem_seq_len,
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
          cross_ln_weight,
          cross_ln_bias,
          cross_q_weight,
          cross_q_bias,
          cross_k_weight,
          cross_k_bias,
          cross_v_weight,
          cross_v_bias,
          cross_out_weight,
          cross_out_bias,
          ffn_ln_weight,
          ffn_ln_bias,
          ffn_inter_weight_0,
          ffn_inter_bias_0,
          ffn_inter_weight_1,
          ffn_inter_bias_1,
          ffn_out_weight,
          ffn_out_bias,
          self_relative_attention_bias_weight,
          decoder_ln_weight,
          decoder_ln_bias,
          embedding_weight,
          embedding_bias,
          output_ids,
          parent_ids,
          sequence_length,
          decoding_strategy,
          beam_size,
          topk,
          topp,
          n_head,
          size_per_head,
          num_layer,
          bos_id,
          eos_id,
          max_len,
          beam_search_diversity_rate,
          alpha,
          temperature,
          early_stopping,
          max_distance,
          num_buckets,
          tie_word_embeddings,
          act,
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
