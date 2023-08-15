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

// TODO(guosheng): `HOST` conflict exists in float.h of paddle and mpi.h of mpi
#include "fusion_unified_decoding_op.h"
#include "pd_traits.h"
#ifdef HOST
#undef HOST
#endif

#include "fastertransformer/decoding_beamsearch.h"
#include "fastertransformer/decoding_sampling.h"
#include "fastertransformer/utils/common.h"

#ifdef BUILD_GPT  // consistent with FasterTransformer
#include "parallel_utils.h"
#endif


template <paddle::DataType D>
std::vector<paddle::Tensor> unified_decoding_kernel(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& memory_sequence_length,
    const paddle::Tensor& type_id,
    const paddle::Tensor& decoder_type_id,
    const paddle::Tensor& logits_mask,
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
    const std::vector<paddle::Tensor>& ffn_layernorm_weight,
    const std::vector<paddle::Tensor>& ffn_layernorm_bias,
    const std::vector<paddle::Tensor>& ffn_intermediate_weight,
    const std::vector<paddle::Tensor>& ffn_intermediate_bias,
    const std::vector<paddle::Tensor>& ffn_output_weight,
    const std::vector<paddle::Tensor>& ffn_output_bias,
    const paddle::Tensor& decoder_layernorm_weight,
    const paddle::Tensor& decoder_layernorm_bias,
    const paddle::Tensor& trans_weight,
    const paddle::Tensor& trans_bias,
    const paddle::Tensor& lm_layernorm_weight,
    const paddle::Tensor& lm_layernorm_bias,
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    const paddle::Tensor& position_encoding_table,
    const paddle::Tensor& type_embedding_weight,
    const paddle::Tensor& role_id,
    const paddle::Tensor& decoder_role_id,
    const paddle::Tensor& role_embedding_table,
    const paddle::Tensor& position_ids,
    const paddle::Tensor& decoder_position_ids,
    paddle::Tensor& output_ids,
    paddle::Tensor& parent_ids,
    paddle::Tensor& sequence_length,
    paddle::Tensor& output_scores,
    const std::string& decoding_strategy,
    const int beam_size,
    const int topk,
    const float topp,
    const int head_num_,
    const int size_per_head_,
    const int num_layer_,
    const int start_id_,
    const int end_id_,
    const int64_t max_seq_len_,
    const float beam_search_diversity_rate_,
    const int unk_id,
    const int mask_id,
    const float temperature,
    const float len_penalty,
    const bool normalize_before,
    const bool pos_bias,
    const std::string& hidden_act,
    const bool early_stopping,
    const int min_length,
    cudaStream_t stream,
    const int tensor_para_size = 1,
    const int layer_para_size = 1,
    const int layer_para_batch_size = 1) {
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
  const int memory_hidden_dim = head_num_ * size_per_head_;
  const int vocab_size = word_emb.shape()[0];

  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  DecodingInitParam<DataType_> decoding_params;
  decoding_params.cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
  decoding_params.cublaslt_handle =
      CublasHandle::GetInstance()->cublaslt_handle_;

  decoding_params.output_ids = output_ids.mutable_data<int>(input_ids.place());
  decoding_params.parent_ids = parent_ids.mutable_data<int>(input_ids.place());
  decoding_params.sequence_length =
      sequence_length.mutable_data<int>(input_ids.place());
  decoding_params.output_scores = output_scores.mutable_data<float>(input_ids.place());

  typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
  decoding_params.stream = stream;
  fastertransformer::Allocator<AllocatorType::PD> allocator_(stream);

  decoding_params.d_start_ids = const_cast<int *>(input_ids.data<int>());
  decoding_params.d_attn_mask =
      reinterpret_cast<DataType_*>(const_cast<data_t_ *>(attn_mask.data<data_t_>()));
  decoding_params.d_start_lengths = memory_sequence_length.data<int>();

  decoding_params.memory_sequence_length = memory_sequence_length.data<int>();
  decoding_params.type_id = type_id.data<int>();
  decoding_params.decoder_type_id = decoder_type_id.data<int>();

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
  decoding_params.request_output_len = max_seq_len_;

#ifdef BUILD_GPT
  auto* model_para_desc = ModelParaDescFactory::CreateModelParaDesc(
      head_num_,
      size_per_head_,
      num_layer_,
      tensor_para_size,
      layer_para_size,
      layer_para_batch_size,
      const_cast<data_t_*>(word_emb.data<data_t_>()));
  auto& tensor_parallel_param = model_para_desc->tensor_parallel_param;
  auto& layer_parallel_param = model_para_desc->layer_parallel_param;
  auto seed = model_para_desc->dist(model_para_desc->gen);
#else
  TensorParallelParam tensor_parallel_param;
  LayerParallelParam layer_parallel_param;
  tensor_parallel_param.rank = 0;
  tensor_parallel_param.world_size = 1;
  tensor_parallel_param.local_head_num_ = head_num_;
  tensor_parallel_param.local_hidden_units_ = memory_hidden_dim;

  layer_parallel_param.rank = 0;
  layer_parallel_param.world_size = 1;
  layer_parallel_param.layers_per_group = num_layer_;
  layer_parallel_param.local_batch_size = batch_size_;
  int seed = -1;
#endif

  DecoderInitParam<DataType_>* params =
      new DecoderInitParam<DataType_>[num_layer_];

  // Allow python passing partial weights for model parallel.
  int inner_coeff =
      (memory_hidden_dim == self_attn_output_weight[0].shape()[0])
          ? ffn_intermediate_weight[0].shape()[1] / memory_hidden_dim
          : (ffn_intermediate_weight[0].shape()[1] * tensor_para_size /
             memory_hidden_dim);

  for (int i = 0; i < self_layernorm_weight.size(); i++) {
    // Allow python passing weights of all layers or only passing the
    // corresponding layers to save memory.
    int layer_idx = self_layernorm_weight.size() != num_layer_
                        ? layer_parallel_param.rank *
                                  layer_parallel_param.layers_per_group +
                              i
                        : i;
    params[layer_idx].stream = stream;
    params[layer_idx].cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
    params[layer_idx].cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

    if (decoding_strategy == "beam_search" ||
        decoding_strategy == "beam_search_v2" ||
        decoding_strategy == "beam_search_v3") {
      params[layer_idx].request_batch_size = batch_size_ * beam_width_;
      params[layer_idx].request_max_mem_seq_len = memory_max_seq_len;
    } else if (decoding_strategy == "sampling" ||
               decoding_strategy == "topk_sampling" ||
               decoding_strategy == "topp_sampling") {
      params[layer_idx].request_batch_size = batch_size_;
      params[layer_idx].request_max_mem_seq_len = memory_max_seq_len;
    }

    // self attn
    params[layer_idx].self_layernorm.gamma = reinterpret_cast<const DataType_*>(
        self_layernorm_weight[i].data<data_t_>());
    params[layer_idx].self_layernorm.beta = reinterpret_cast<const DataType_*>(
        self_layernorm_bias[i].data<data_t_>());
    // query
    params[layer_idx].self_attention.query_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_attn_query_weight[i].data<data_t_>());
    params[layer_idx].self_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_query_bias[i].data<data_t_>());
    // key
    params[layer_idx].self_attention.key_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_attn_key_weight[i].data<data_t_>());
    params[layer_idx].self_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_key_bias[i].data<data_t_>());
    // value
    params[layer_idx].self_attention.value_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_attn_value_weight[i].data<data_t_>());
    params[layer_idx].self_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_value_bias[i].data<data_t_>());
    // out proj
    params[layer_idx].self_attention.attention_output_weight.kernel =
        reinterpret_cast<const DataType_*>(
            self_attn_output_weight[i].data<data_t_>());

    params[layer_idx].self_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(
            self_attn_output_bias[i].data<data_t_>());

    // ffn
    params[layer_idx].ffn_layernorm.gamma = reinterpret_cast<const DataType_*>(
        ffn_layernorm_weight[i].data<data_t_>());
    params[layer_idx].ffn_layernorm.beta = reinterpret_cast<const DataType_*>(
        ffn_layernorm_bias[i].data<data_t_>());
    // intermediate proj
    params[layer_idx].ffn.intermediate_weight.kernel =
        reinterpret_cast<const DataType_*>(
            ffn_intermediate_weight[i].data<data_t_>());
    params[layer_idx].ffn.intermediate_weight.bias = reinterpret_cast<const DataType_*>(
        ffn_intermediate_bias[i].data<data_t_>());
    // out proj
    params[layer_idx].ffn.output_weight.kernel = reinterpret_cast<const DataType_*>(
        ffn_output_weight[i].data<data_t_>());
    params[layer_idx].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_output_bias[i].data<data_t_>());
  }

  decoding_params.layernorm.gamma = reinterpret_cast<const DataType_*>(
      decoder_layernorm_weight.data<data_t_>());
  decoding_params.layernorm.beta = reinterpret_cast<const DataType_*>(
      decoder_layernorm_bias.data<data_t_>());
  decoding_params.trans_kernel =
      reinterpret_cast<const DataType_*>(trans_weight.data<data_t_>());
  decoding_params.trans_bias =
      reinterpret_cast<const DataType_*>(trans_bias.data<data_t_>());

  decoding_params.lm_layernorm.gamma =
      reinterpret_cast<const DataType_*>(lm_layernorm_weight.data<data_t_>());
  decoding_params.lm_layernorm.beta =
      reinterpret_cast<const DataType_*>(lm_layernorm_bias.data<data_t_>());

  // For embedding
  decoding_params.embedding_table =
      reinterpret_cast<const DataType_*>(word_emb.data<data_t_>());
  // For weight sharing matmul
  decoding_params.embedding_kernel =
      reinterpret_cast<const DataType_*>(embedding_weight.data<data_t_>());
  // For matmul bias
  decoding_params.embedding_bias =
      reinterpret_cast<const DataType_*>(embedding_bias.data<data_t_>());
  decoding_params.position_encoding_table = reinterpret_cast<const DataType_*>(
      position_encoding_table.data<data_t_>());

  // For masking some id during gen.
  decoding_params.logits_mask =
      reinterpret_cast<const DataType_*>(logits_mask.data<data_t_>());

  decoding_params.type_table =
      reinterpret_cast<const DataType_*>(type_embedding_weight.data<data_t_>());

  // For role embedding.
  auto role_id_shape = role_id.shape();
  if (role_id_shape.size() > 0 && numel(role_id_shape) > 0) {
    decoding_params.role_id = role_id.data<int>();
    decoding_params.decoder_role_id = decoder_role_id.data<int>();
    decoding_params.role_embedding_table =
        reinterpret_cast<const DataType_*>(role_embedding_table.data<data_t_>());
  }

  auto position_id_shape = position_ids.shape();
  if (position_id_shape.size() > 0 && numel(position_id_shape) > 0) {
      decoding_params.position_ids = position_ids.data<int>();
      decoding_params.decoder_position_ids = decoder_position_ids.data<int>();
  }

  ActivationType activate =
      (hidden_act == "gelu") ? ActivationType::GELU : ActivationType::RELU;

  int finished_candidate_num_ =
      ("beam_search_v3" == decoding_strategy) ? beam_width_ : beam_width_ * 2;

  if ("beam_search" == decoding_strategy) {
    DecodingBeamsearch<DecodingTraits_::OpType>* unified_decoding_beam_search_;

    unified_decoding_beam_search_ =
        new DecodingBeamsearch<DecodingTraits_::OpType>(
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
            true,        /*is_fuse_topk_softMax*/
            true,        /*is_fuse_qkv*/
            false,       /*keep_alive_beam*/
            len_penalty, /*alpha not used for this case*/
            normalize_before,
            0, /*pos_offset BART only for now*/
            activate,
            pos_bias,
            true, /*prefix_lm*/
            -1,  /*finished_candidate_num*/
            false,  /*early_stopping*/
            false,  /*is_mbart*/
            min_length,
            inner_coeff);
    unified_decoding_beam_search_->set_tensor_parallel_param(
        tensor_parallel_param);
    unified_decoding_beam_search_->set_layer_parallel_param(
        layer_parallel_param);
    unified_decoding_beam_search_->forward_context(params, decoding_params);
    unified_decoding_beam_search_->forward(params, decoding_params);

    delete unified_decoding_beam_search_;
  } else if ("beam_search_v2" == decoding_strategy ||
             "beam_search_v3" == decoding_strategy) {
    DecodingBeamsearch<DecodingTraits_::OpType>* unified_decoding_beam_search_;

    unified_decoding_beam_search_ =
        new DecodingBeamsearch<DecodingTraits_::OpType>(
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
            true, /*is_fuse_topk_softMax*/
            true, /*is_fuse_qkv*/
            true, /*keep_alive_beam*/
            len_penalty,
            normalize_before,
            0, /*pos_offset BART only for now*/
            activate,
            pos_bias,
            true, /*prefix_lm*/
            finished_candidate_num_,
            early_stopping,
            false,  /*is_mbart*/
            min_length,
            inner_coeff);
    unified_decoding_beam_search_->forward_context(params, decoding_params);
    unified_decoding_beam_search_->forward(params, decoding_params);

    delete unified_decoding_beam_search_;
  } else if ("topk_sampling" == decoding_strategy ||
             "topp_sampling" == decoding_strategy ||
             "sampling" == decoding_strategy) {
    DecodingSampling<DecodingTraits_::OpType>* unified_decoding_sampling_;

    unified_decoding_sampling_ = new DecodingSampling<DecodingTraits_::OpType>(
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
        true, /*is_fuse_qkv*/
        normalize_before,
        0, /*pos_offset BART only for now*/
        activate,
        pos_bias,
        temperature,
        1.0,   /*repeat_penalty*/
        true,  /*prefix_lm*/
        false, /*is_mbart*/
        min_length,
        inner_coeff,
        seed,
        tensor_para_size,
        layer_para_size);
    unified_decoding_sampling_->set_tensor_parallel_param(
        tensor_parallel_param);
    unified_decoding_sampling_->set_layer_parallel_param(layer_parallel_param);
    unified_decoding_sampling_->forward_context(params, decoding_params);
    unified_decoding_sampling_->forward(params, decoding_params);

    delete unified_decoding_sampling_;
  } else {
    PD_THROW(
        "Only beam_search, beam_search_v2, topk_sampling and topp_sampling are "
        "supported for "
        "FastGeneration. ");
  }
  delete[] params;

  return {output_ids, parent_ids, sequence_length, output_scores};
}

std::vector<paddle::Tensor> UnifiedDecodingCUDAForward(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& mem_seq_len,
    const paddle::Tensor& type_id,
    const paddle::Tensor& decoder_type_id,
    const paddle::Tensor& logits_mask,
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
    const paddle::Tensor& embedding_weight,
    const paddle::Tensor& embedding_bias,
    const paddle::Tensor& positional_embedding_weight,
    const paddle::Tensor& type_embedding_weight,
    const paddle::Tensor& role_id,
    const paddle::Tensor& decoder_role_id,
    const paddle::Tensor& role_embedding_table,
    const paddle::Tensor& position_ids,
    const paddle::Tensor& decoder_position_ids,
    paddle::Tensor& output_ids,
    paddle::Tensor& parent_ids,
    paddle::Tensor& sequence_length,
    paddle::Tensor& output_scores,
    const std::string& decoding_strategy,
    const int beam_size,
    const int topk,
    const float topp,
    const int n_head,
    const int size_per_head,
    const int num_layer,
    const int bos_id,
    const int eos_id,
    const int64_t max_len,
    const float beam_search_diversity_rate,
    const int unk_id,
    const int mask_id,
    const float temperature,
    const float len_penalty,
    const bool normalize_before,
    const bool pos_bias,
    const std::string& hidden_act,
    const bool early_stopping,
    const int min_length,
    const int tensor_para_size = 1,
    const int layer_para_size = 1,
    const int layer_para_batch_size = 1) {
  auto stream = input_ids.stream();

  cublasSetStream(CublasHandle::GetInstance()->cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  switch (self_ln_weight[0].type()) {
    case paddle::DataType::FLOAT16: {
      ret = unified_decoding_kernel<paddle::DataType::FLOAT16>(
          input_ids,
          attn_mask,
          mem_seq_len,
          type_id,
          decoder_type_id,
          logits_mask,
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
          embedding_weight,
          embedding_bias,
          positional_embedding_weight,
          type_embedding_weight,
          role_id,
          decoder_role_id,
          role_embedding_table,
          position_ids,
          decoder_position_ids,
          output_ids,
          parent_ids,
          sequence_length,
          output_scores,
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
          unk_id,
          mask_id,
          temperature,
          len_penalty,
          normalize_before,
          pos_bias,
          hidden_act,
          early_stopping,
          min_length,
          stream,
          tensor_para_size,
          layer_para_size,
          layer_para_batch_size);
      break;
    }
    case paddle::DataType::FLOAT32: {
      ret = unified_decoding_kernel<paddle::DataType::FLOAT32>(
          input_ids,
          attn_mask,
          mem_seq_len,
          type_id,
          decoder_type_id,
          logits_mask,
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
          embedding_weight,
          embedding_bias,
          positional_embedding_weight,
          type_embedding_weight,
          role_id,
          decoder_role_id,
          role_embedding_table,
          position_ids,
          decoder_position_ids,
          output_ids,
          parent_ids,
          sequence_length,
          output_scores,
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
          unk_id,
          mask_id,
          temperature,
          len_penalty,
          normalize_before,
          pos_bias,
          hidden_act,
          early_stopping,
          min_length,
          stream,
          tensor_para_size,
          layer_para_size,
          layer_para_batch_size);
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
