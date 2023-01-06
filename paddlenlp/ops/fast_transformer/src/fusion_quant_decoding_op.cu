/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "fastertransformer/cuda/cub/cub.cuh"
#include "fusion_quant_decoding_op.h"
#include "pd_traits.h"

template <paddle::DataType D>
std::vector<paddle::Tensor> decoding_kernel(
    const paddle::Tensor& input,
    const paddle::Tensor& memory_sequence_length,
    const paddle::Tensor& word_emb,
    const std::vector<paddle::Tensor>& self_layernorm_weight,
    const std::vector<paddle::Tensor>& self_layernorm_bias,
    const std::vector<paddle::Tensor>& self_attn_query_weight,
    const std::vector<paddle::Tensor>& self_attn_query_weight_scale,
    const std::vector<paddle::Tensor>& self_attn_query_bias,
    const std::vector<paddle::Tensor>& self_attn_key_weight,
    const std::vector<paddle::Tensor>& self_attn_key_weight_scale,
    const std::vector<paddle::Tensor>& self_attn_key_bias,
    const std::vector<paddle::Tensor>& self_attn_value_weight,
    const std::vector<paddle::Tensor>& self_attn_value_weight_scale,
    const std::vector<paddle::Tensor>& self_attn_value_bias,
    const std::vector<paddle::Tensor>& self_attn_output_weight,
    const std::vector<paddle::Tensor>& self_attn_output_weight_scale,
    const std::vector<paddle::Tensor>& self_attn_output_bias,
    const std::vector<paddle::Tensor>& cross_layernorm_weight,
    const std::vector<paddle::Tensor>& cross_layernorm_bias,
    const std::vector<paddle::Tensor>& cross_attn_query_weight,
    const std::vector<paddle::Tensor>& cross_attn_query_weight_scale,
    const std::vector<paddle::Tensor>& cross_attn_query_bias,
    const std::vector<paddle::Tensor>& cross_attn_key_weight,
    const std::vector<paddle::Tensor>& cross_attn_key_weight_scale,
    const std::vector<paddle::Tensor>& cross_attn_key_bias,
    const std::vector<paddle::Tensor>& cross_attn_value_weight,
    const std::vector<paddle::Tensor>& cross_attn_value_weight_scale,
    const std::vector<paddle::Tensor>& cross_attn_value_bias,
    const std::vector<paddle::Tensor>& cross_attn_output_weight,
    const std::vector<paddle::Tensor>& cross_attn_output_weight_scale,
    const std::vector<paddle::Tensor>& cross_attn_output_bias,
    const std::vector<paddle::Tensor>& ffn_layernorm_weight,
    const std::vector<paddle::Tensor>& ffn_layernorm_bias,
    const std::vector<paddle::Tensor>& ffn_intermediate_weight,
    const std::vector<paddle::Tensor>& ffn_intermediate_weight_scale,
    const std::vector<paddle::Tensor>& ffn_intermediate_bias,
    const std::vector<paddle::Tensor>& ffn_output_weight,
    const std::vector<paddle::Tensor>& ffn_output_weight_scale,
    const std::vector<paddle::Tensor>& ffn_output_bias,
    const paddle::Tensor& decoder_layernorm_weight,
    const paddle::Tensor& decoder_layernorm_bias,
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    const paddle::Tensor& position_encoding_table,
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
    cudaStream_t stream) {
  int beam_width_ = (decoding_strategy == "beam_search" ||
                     decoding_strategy == "beam_search_v2")
                        ? beam_size
                        : 1;
  int candidate_num_ = (decoding_strategy == "topk_sampling" ||
                        decoding_strategy == "topp_sampling")
                           ? topk
                           : 1;
  float probability_threshold_ = (decoding_strategy == "topk_sampling" ||
                                  decoding_strategy == "topp_sampling")
                                     ? topp
                                     : 0.0;

  auto input_dims = input.shape();
  int batch_size_ = (decoding_strategy == "beam_search" ||
                     decoding_strategy == "beam_search_v2")
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
  int sm = SMVersion::GetInstance()->sm_;

  //   decoding_params.output_ids = output_ids.data<int>();
  //   decoding_params.parent_ids = parent_ids.data<int>();
  //   decoding_params.sequence_length = sequence_length.data<int>();
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

  auto q_weight_shape = self_attn_query_weight[0].shape();
  auto k_weight_shape = self_attn_key_weight[0].shape();
  bool fuse_qkv = (q_weight_shape[1] == k_weight_shape[1]) ? false : true;

  bool use_int8 = (ffn_intermediate_weight[0].type() == paddle::DataType::INT8);

  for (int i = 0; i < num_layer_; i++) {
    params[i].stream = stream;
    params[i].cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
    params[i].cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

    if (decoding_strategy == "beam_search" ||
        decoding_strategy == "beam_search_v2") {
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
    params[i].self_layernorm.beta = reinterpret_cast<const DataType_*>(
        self_layernorm_bias[i].data<data_t_>());
    // query
    if (use_int8) {
      params[i].self_attention.query_weight.int8_kernel =
          self_attn_query_weight[i].data<int8_t>();
      params[i].self_attention.query_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              self_attn_query_weight_scale[i].data<data_t_>());
    } else {
      params[i].self_attention.query_weight.kernel =
          reinterpret_cast<const DataType_*>(
              self_attn_query_weight[i].data<data_t_>());
    }

    params[i].self_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_query_bias[i].data<data_t_>());
    // key
    if (use_int8) {
      params[i].self_attention.key_weight.int8_kernel =
          self_attn_key_weight[i].data<int8_t>();
      params[i].self_attention.key_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              self_attn_key_weight_scale[i].data<data_t_>());
    } else {
      params[i].self_attention.key_weight.kernel =
          reinterpret_cast<const DataType_*>(
              self_attn_key_weight[i].data<data_t_>());
    }

    params[i].self_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_key_bias[i].data<data_t_>());
    // value
    if (use_int8) {
      params[i].self_attention.value_weight.int8_kernel =
          self_attn_value_weight[i].data<int8_t>();
      params[i].self_attention.value_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              self_attn_value_weight_scale[i].data<data_t_>());
    } else {
      params[i].self_attention.value_weight.kernel =
          reinterpret_cast<const DataType_*>(
              self_attn_value_weight[i].data<data_t_>());
    }

    params[i].self_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_value_bias[i].data<data_t_>());
    // out proj
    if (use_int8) {
      params[i].self_attention.attention_output_weight.int8_kernel =
          self_attn_output_weight[i].data<int8_t>();
      params[i].self_attention.attention_output_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              self_attn_output_weight_scale[i].data<data_t_>());
    } else {
      params[i].self_attention.attention_output_weight.kernel =
          reinterpret_cast<const DataType_*>(
              self_attn_output_weight[i].data<data_t_>());
    }

    params[i].self_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_output_bias[i].data<data_t_>());

    // cross
    params[i].cross_layernorm.gamma = reinterpret_cast<const DataType_*>(
        cross_layernorm_weight[i].data<data_t_>());
    params[i].cross_layernorm.beta = reinterpret_cast<const DataType_*>(
        cross_layernorm_bias[i].data<data_t_>());
    // query
    if (use_int8) {
      params[i].cross_attention.query_weight.int8_kernel =
          cross_attn_query_weight[i].data<int8_t>();
      params[i].cross_attention.query_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              cross_attn_query_weight_scale[i].data<data_t_>());
    } else {
      params[i].cross_attention.query_weight.kernel =
          reinterpret_cast<const DataType_*>(
              cross_attn_query_weight[i].data<data_t_>());
    }

    params[i].cross_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(
            cross_attn_query_bias[i].data<data_t_>());
    // key
    if (cross_attn_key_weight[i].type() == paddle::DataType::INT8) {
      params[i].cross_attention.key_weight.int8_kernel =
          cross_attn_key_weight[i].data<int8_t>();
      params[i].cross_attention.key_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              cross_attn_key_weight_scale[i].data<data_t_>());
    } else {
      params[i].cross_attention.key_weight.kernel =
          reinterpret_cast<const DataType_*>(
              cross_attn_key_weight[i].data<data_t_>());
    }

    params[i].cross_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(
            cross_attn_key_bias[i].data<data_t_>());
    // value
    if (cross_attn_value_weight[i].type() == paddle::DataType::INT8) {
      params[i].cross_attention.value_weight.int8_kernel =
          cross_attn_value_weight[i].data<int8_t>();
      params[i].cross_attention.value_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              cross_attn_value_weight_scale[i].data<data_t_>());
    } else {
      params[i].cross_attention.value_weight.kernel =
          reinterpret_cast<const DataType_*>(
              cross_attn_value_weight[i].data<data_t_>());
    }

    params[i].cross_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(
            cross_attn_value_bias[i].data<data_t_>());
    // out proj
    if (use_int8) {
      params[i].cross_attention.attention_output_weight.int8_kernel =
          cross_attn_output_weight[i].data<int8_t>();
      params[i].cross_attention.attention_output_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              cross_attn_output_weight_scale[i].data<data_t_>());
    } else {
      params[i].cross_attention.attention_output_weight.kernel =
          reinterpret_cast<const DataType_*>(
              cross_attn_output_weight[i].data<data_t_>());
    }

    params[i].cross_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(
            cross_attn_output_bias[i].data<data_t_>());

    // ffn
    params[i].ffn_layernorm.gamma = reinterpret_cast<const DataType_*>(
        ffn_layernorm_weight[i].data<data_t_>());
    params[i].ffn_layernorm.beta = reinterpret_cast<const DataType_*>(
        ffn_layernorm_bias[i].data<data_t_>());
    // intermediate proj
    if (use_int8) {
      params[i].ffn.intermediate_weight.int8_kernel =
          ffn_intermediate_weight[i].data<int8_t>();
      params[i].ffn.intermediate_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              ffn_intermediate_weight_scale[i].data<data_t_>());
    } else {
      params[i].ffn.intermediate_weight.kernel =
          reinterpret_cast<const DataType_*>(
              ffn_intermediate_weight[i].data<data_t_>());
    }

    params[i].ffn.intermediate_weight.bias = reinterpret_cast<const DataType_*>(
        ffn_intermediate_bias[i].data<data_t_>());
    // out proj
    if (use_int8) {
      params[i].ffn.output_weight.int8_kernel =
          ffn_output_weight[i].data<int8_t>();
      params[i].ffn.output_weight.kernel_scale =
          reinterpret_cast<const DataType_*>(
              ffn_output_weight_scale[i].data<data_t_>());
    } else {
      params[i].ffn.output_weight.kernel = reinterpret_cast<const DataType_*>(
          ffn_output_weight[i].data<data_t_>());
    }

    params[i].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_output_bias[i].data<data_t_>());
  }

  decoding_params.layernorm.gamma = reinterpret_cast<const DataType_*>(
      decoder_layernorm_weight.data<data_t_>());
  decoding_params.layernorm.beta = reinterpret_cast<const DataType_*>(
      decoder_layernorm_bias.data<data_t_>());
  // for embedding
  decoding_params.embedding_table =
      reinterpret_cast<const DataType_*>(word_emb.data<data_t_>());

  // for weight sharing matmul
  decoding_params.embedding_kernel =
      reinterpret_cast<const DataType_*>(embedding_weight.data<data_t_>());
  // for matmul bias
  decoding_params.embedding_bias =
      reinterpret_cast<const DataType_*>(embedding_bias.data<data_t_>());

  decoding_params.position_encoding_table = reinterpret_cast<const DataType_*>(
      position_encoding_table.data<data_t_>());

  if ("beam_search" == decoding_strategy) {
    DecodingBeamsearch<DecodingTraits_::OpType>* decoding_beam_search_;
    decoding_beam_search_ = new DecodingBeamsearch<DecodingTraits_::OpType>(
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
        fuse_qkv,
        false,                 // keep_alive_beam
        0.6,                   // alpha
        true,                  // normalization_before
        0,                     // pos_offset
        ActivationType::RELU,  // act
        false,                 // pos_bias
        false,                 // prefix_lm
        -1,                    // finished_candidate_num
        false,                 // early_stopping
        false,                 // is_mbart
        0,                     // min_length
        4,                     // inner_coeff
        false, // is_miro
        use_int8,
        sm);

    decoding_beam_search_->forward(params, decoding_params);

    delete decoding_beam_search_;
  } else if ("beam_search_v2" == decoding_strategy) {
    DecodingBeamsearch<DecodingTraits_::OpType>* decoding_beam_search_;
    decoding_beam_search_ = new DecodingBeamsearch<DecodingTraits_::OpType>(
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
        fuse_qkv,
        true,  // keep_alive_beam
        alpha,
        true,                  // normalization_before
        0,                     // pos_offset
        ActivationType::RELU,  // act
        false,                 // pos_bias
        false,                 // prefix_lm
        -1,                    // finished_candidate_num
        false,                 // early_stopping
        false,                 // is_mbart
        0,                     // min_length
        4,                     // inner_coeff
        false, // is_miro
        use_int8,
        sm);

    decoding_beam_search_->forward(params, decoding_params);

    delete decoding_beam_search_;
  } else if ("topk_sampling" == decoding_strategy ||
             "topp_sampling" == decoding_strategy ||
             "sampling" == decoding_strategy) {
    DecodingSampling<DecodingTraits_::OpType>* decoding_sampling_;
    decoding_sampling_ = new DecodingSampling<DecodingTraits_::OpType>(
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
        fuse_qkv,
        true,                  // normalization_before
        0,                     // pos_offset
        ActivationType::RELU,  // act
        false,                 // pos_bias
        1.0,                   // temperature
        1.0,                   // repeat_penalty
        false,                 // prefix_lm
        false,                 // is_mbart
        0,                     // min_length
        4,                     // inner_coeff
        -1,                    // seed
        1,                     // tensor_para_size
        1,                     // layer_para_size
        false, // is_miro
        use_int8,
        sm);

    decoding_sampling_->forward(params, decoding_params);

    delete decoding_sampling_;
  } else {
    PD_THROW(
        "Only beam_search, topk_sampling and topp_sampling are supported for "
        "FasterTransformer. ");
  }
  delete[] params;

  return {output_ids, parent_ids, sequence_length};
}

std::vector<paddle::Tensor> QuantDecodingCUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& mem_seq_len,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_q_weight_scale,
    const std::vector<paddle::Tensor>& self_q_bias,
    const std::vector<paddle::Tensor>& self_k_weight,
    const std::vector<paddle::Tensor>& self_k_weight_scale,
    const std::vector<paddle::Tensor>& self_k_bias,
    const std::vector<paddle::Tensor>& self_v_weight,
    const std::vector<paddle::Tensor>& self_v_weight_scale,
    const std::vector<paddle::Tensor>& self_v_bias,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& self_out_weight_scale,
    const std::vector<paddle::Tensor>& self_out_bias,
    const std::vector<paddle::Tensor>& cross_ln_weight,
    const std::vector<paddle::Tensor>& cross_ln_bias,
    const std::vector<paddle::Tensor>& cross_q_weight,
    const std::vector<paddle::Tensor>& cross_q_weight_scale,
    const std::vector<paddle::Tensor>& cross_q_bias,
    const std::vector<paddle::Tensor>& cross_k_weight,
    const std::vector<paddle::Tensor>& cross_k_weight_scale,
    const std::vector<paddle::Tensor>& cross_k_bias,
    const std::vector<paddle::Tensor>& cross_v_weight,
    const std::vector<paddle::Tensor>& cross_v_weight_scale,
    const std::vector<paddle::Tensor>& cross_v_bias,
    const std::vector<paddle::Tensor>& cross_out_weight,
    const std::vector<paddle::Tensor>& cross_out_weight_scale,
    const std::vector<paddle::Tensor>& cross_out_bias,
    const std::vector<paddle::Tensor>& ffn_ln_weight,
    const std::vector<paddle::Tensor>& ffn_ln_bias,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_weight_scale,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_weight_scale,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    const paddle::Tensor& positional_embedding_weight,
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
    const float& alpha) {
  auto stream = input.stream();

  cublasSetStream(CublasHandle::GetInstance()->cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  switch (input.type()) {
    case paddle::DataType::FLOAT16: {
      ret = decoding_kernel<paddle::DataType::FLOAT16>(
          input,
          mem_seq_len,
          word_embedding,
          self_ln_weight,
          self_ln_bias,
          self_q_weight,
          self_q_weight_scale,
          self_q_bias,
          self_k_weight,
          self_k_weight_scale,
          self_k_bias,
          self_v_weight,
          self_v_weight_scale,
          self_v_bias,
          self_out_weight,
          self_out_weight_scale,
          self_out_bias,
          cross_ln_weight,
          cross_ln_bias,
          cross_q_weight,
          cross_q_weight_scale,
          cross_q_bias,
          cross_k_weight,
          cross_k_weight_scale,
          cross_k_bias,
          cross_v_weight,
          cross_v_weight_scale,
          cross_v_bias,
          cross_out_weight,
          cross_out_weight_scale,
          cross_out_bias,
          ffn_ln_weight,
          ffn_ln_bias,
          ffn_inter_weight,
          ffn_inter_weight_scale,
          ffn_inter_bias,
          ffn_out_weight,
          ffn_out_weight_scale,
          ffn_out_bias,
          decoder_ln_weight,
          decoder_ln_bias,
          embedding_weight,
          embedding_bias,
          positional_embedding_weight,
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
          stream);
      break;
    }
    case paddle::DataType::FLOAT32: {
      ret = decoding_kernel<paddle::DataType::FLOAT32>(
          input,
          mem_seq_len,
          word_embedding,
          self_ln_weight,
          self_ln_bias,
          self_q_weight,
          self_q_weight_scale,
          self_q_bias,
          self_k_weight,
          self_k_weight_scale,
          self_k_bias,
          self_v_weight,
          self_v_weight_scale,
          self_v_bias,
          self_out_weight,
          self_out_weight_scale,
          self_out_bias,
          cross_ln_weight,
          cross_ln_bias,
          cross_q_weight,
          cross_q_weight_scale,
          cross_q_bias,
          cross_k_weight,
          cross_k_weight_scale,
          cross_k_bias,
          cross_v_weight,
          cross_v_weight_scale,
          cross_v_bias,
          cross_out_weight,
          cross_out_weight_scale,
          cross_out_bias,
          ffn_ln_weight,
          ffn_ln_bias,
          ffn_inter_weight,
          ffn_inter_weight_scale,
          ffn_inter_bias,
          ffn_out_weight,
          ffn_out_weight_scale,
          ffn_out_bias,
          decoder_ln_weight,
          decoder_ln_bias,
          embedding_weight,
          embedding_bias,
          positional_embedding_weight,
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
