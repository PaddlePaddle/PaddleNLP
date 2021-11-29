/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "fusion_mbart_decoding_op.h"
#include "pd_traits.h"


__global__ void get_trg_length_mbart(const int* trg_word,
                                     int* trg_length,
                                     const int seq_len,
                                     const int pad_id) {
  int bid = threadIdx.x;

  int cnt_nonpads = 0;
  for (int i = 0; i < seq_len; ++i) {
    if (pad_id != trg_word[bid * seq_len + i]) {
      cnt_nonpads++;
    } else {
      break;
    }
  }
  trg_length[bid] = cnt_nonpads;
}

template <paddle::DataType D>
std::vector<paddle::Tensor> mbart_decoding_kernel(
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
    const std::vector<paddle::Tensor>& ffn_intermediate_weight,
    const std::vector<paddle::Tensor>& ffn_intermediate_bias,
    const std::vector<paddle::Tensor>& ffn_output_weight,
    const std::vector<paddle::Tensor>& ffn_output_bias,
    const paddle::Tensor& decoder_layernorm_weight,
    const paddle::Tensor& decoder_layernorm_bias,
    const paddle::Tensor& mbart_layernorm_weight,
    const paddle::Tensor& mbart_layernorm_bias,
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    const paddle::Tensor& position_encoding_table,
    const paddle::Tensor& trg_word,
    paddle::Tensor& output_ids,
    paddle::Tensor& parent_ids,
    paddle::Tensor& sequence_length,
    std::string decoding_strategy,
    int beam_size,
    int topk,
    float topp,
    int head_num_,
    int size_per_head_,
    int num_layer_,
    int start_id_,
    int end_id_,
    int64_t max_seq_len_,
    float beam_search_diversity_rate_,
    float alpha,
    const std::string& hidden_act,
    cublasHandle_t cublas_handle_,
    cublasLtHandle_t cublaslt_handle_,
    cudaStream_t stream) {
  int beam_width_ = (decoding_strategy == "beam_search" ||
                     decoding_strategy == "beam_search_v2")
                        ? beam_size
                        : 1;
  int candidate_num_ = (decoding_strategy == "sampling") ? topk : 1;
  float probability_threshold_ = (decoding_strategy == "sampling") ? topp : 0.0;

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
  decoding_params.cublas_handle = cublas_handle_;
  decoding_params.cublaslt_handle = cublaslt_handle_;

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

  auto trg_word_shape = trg_word.shape();
  int trg_max_len =
      (trg_word_shape.size() == 2) ? static_cast<int>(trg_word_shape[1]) : 0;

  paddle::Tensor trg_length =
      (trg_word_shape.size() == 2 && trg_word_shape[0] != 0)
          ? paddle::Tensor(paddle::PlaceType::kGPU, {trg_word_shape[0]})
          : paddle::Tensor(paddle::PlaceType::kGPU, {1});
  auto trg_length_ptr = trg_length.mutable_data<int>(input.place());

  if (trg_word_shape.size() == 2 && trg_word_shape[0] != 0 &&
      trg_word_shape[0] != 0) {
    decoding_params.trg_word = trg_word.data<int>();

    get_trg_length_mbart<<<1, trg_word_shape[0], 0, stream>>>(
        decoding_params.trg_word, trg_length_ptr, trg_max_len, start_id_);
    decoding_params.trg_length = trg_length_ptr;
  }

  DecoderInitParam<DataType_>* params =
      new DecoderInitParam<DataType_>[num_layer_];

  for (int i = 0; i < num_layer_; i++) {
    params[i].stream = stream;
    params[i].cublas_handle = cublas_handle_;
    params[i].cublaslt_handle = cublaslt_handle_;

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
    params[i].cross_layernorm.beta = reinterpret_cast<const DataType_*>(
        cross_layernorm_bias[i].data<data_t_>());
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
    params[i].ffn_layernorm.beta = reinterpret_cast<const DataType_*>(
        ffn_layernorm_bias[i].data<data_t_>());
    // intermediate proj
    params[i].ffn.intermediate_weight.kernel =
        reinterpret_cast<const DataType_*>(
            ffn_intermediate_weight[i].data<data_t_>());
    params[i].ffn.intermediate_weight.bias = reinterpret_cast<const DataType_*>(
        ffn_intermediate_bias[i].data<data_t_>());
    // out proj
    params[i].ffn.output_weight.kernel = reinterpret_cast<const DataType_*>(
        ffn_output_weight[i].data<data_t_>());
    params[i].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_output_bias[i].data<data_t_>());
  }

  decoding_params.layernorm.gamma = reinterpret_cast<const DataType_*>(
      decoder_layernorm_weight.data<data_t_>());
  decoding_params.layernorm.beta = reinterpret_cast<const DataType_*>(
      decoder_layernorm_bias.data<data_t_>());

  // for mbart embedding layernorm
  decoding_params.mbart_layernorm.gamma = reinterpret_cast<const DataType_*>(
      mbart_layernorm_weight.data<data_t_>());
  decoding_params.mbart_layernorm.beta =
      reinterpret_cast<const DataType_*>(mbart_layernorm_bias.data<data_t_>());

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

  ActivationType activate =
      (hidden_act == "gelu") ? ActivationType::GELU : ActivationType::RELU;

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
        true,  /*is_fuse_topk_softMax*/
        false, /*is_fuse_qkv*/
        false, /*keep_alive_beam*/
        alpha, /*alpha not used for this case*/
        true,
        2, /*pos_offset BART and MBART only for now*/
        activate,
        false,  // pos_bias
        false /*prefix_lm*/,
        true /*is_mbart */);

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
        true,   // is_fuse_topk_softMax
        false,  // is_fuse_qkv
        true,   // keep_alive_beam
        alpha,
        true,
        2, /*pos_offset BART and MBART only for now*/
        activate,
        false,  // pos_bias
        false /*prefix_lm*/,
        true /*is_mbart */);

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
        false, /*is_fuse_qkv*/
        true,
        2, /*pos_offset BART and MBART only for now*/
        activate,
        false,  // pos_bias
        1.0,    // temperature
        1.0,    // repeat_penalty
        false,  // prefix_lm
        true /*is_mbart */);

    decoding_sampling_->forward(params, decoding_params);

    delete decoding_sampling_;
  } else {
    PD_THROW(
        "Only beam_search, beam_search_v2 and sampling are supported for "
        "FasterTransformer. ");
  }
  delete[] params;

  return {output_ids, parent_ids, sequence_length};
}

std::vector<paddle::Tensor> MBartDecodingCUDAForward(
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
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& mbart_ln_weight,
    const paddle::Tensor& mbart_ln_bias,
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    const paddle::Tensor& positional_embedding_weight,
    const paddle::Tensor& trg_word,
    paddle::Tensor& output_ids,
    paddle::Tensor& parent_ids,
    paddle::Tensor& sequence_length,
    std::string decoding_strategy,
    int beam_size,
    int topk,
    float topp,
    int n_head,
    int size_per_head,
    int num_layer,
    int bos_id,
    int eos_id,
    int64_t max_len,
    float beam_search_diversity_rate,
    float alpha,
    const std::string& hidden_act) {
  auto stream = input.stream();
  cublasHandle_t cublas_handle_;
  cublasCreate(&cublas_handle_);
  cublasLtHandle_t cublaslt_handle_;
  cublasLtCreate(&cublaslt_handle_);
  cublasSetStream(cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  switch (input.type()) {
    case paddle::DataType::FLOAT16: {
      ret = mbart_decoding_kernel<paddle::DataType::FLOAT16>(
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
          ffn_inter_weight,
          ffn_inter_bias,
          ffn_out_weight,
          ffn_out_bias,
          decoder_ln_weight,
          decoder_ln_bias,
          mbart_ln_weight,
          mbart_ln_bias,
          embedding_weight,
          embedding_bias,
          positional_embedding_weight,
          trg_word,
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
          hidden_act,
          cublas_handle_,
          cublaslt_handle_,
          stream);
      break;
    }
    case paddle::DataType::FLOAT32: {
      ret = mbart_decoding_kernel<paddle::DataType::FLOAT32>(
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
          ffn_inter_weight,
          ffn_inter_bias,
          ffn_out_weight,
          ffn_out_bias,
          decoder_ln_weight,
          decoder_ln_bias,
          mbart_ln_weight,
          mbart_ln_bias,
          embedding_weight,
          embedding_bias,
          positional_embedding_weight,
          trg_word,
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
          hidden_act,
          cublas_handle_,
          cublaslt_handle_,
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

  cublasDestroy(cublas_handle_);
  cublasLtDestroy(cublaslt_handle_);
  return ret;
}
