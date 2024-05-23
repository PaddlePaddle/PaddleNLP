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

// Use the global cublas handle
#include "cublas_handle.h"

// TODO(guosheng): `HOST` conflict exists in float.h of paddle and mpi.h of mpi
#include "fusion_gptj_op.h"
#include "pd_traits.h"
#ifdef HOST
#undef HOST
#endif

#include "fastertransformer/utils/common.h"

#ifdef BUILD_GPT  // consistent with FasterTransformer
#include "parallel_utils.h"
#endif

template <paddle::DataType D>
std::vector<paddle::Tensor> gptj_kernel(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_emb,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& emb_weight,
    const paddle::Tensor& emb_bias,
    paddle::Tensor& output_ids,
    const int topk,
    const float topp,
    const int max_len,
    const int n_head,
    const int size_per_head,
    const int num_layer,
    const int bos_id,
    const int eos_id,
    const float temperature,
    const int rotary_embedding_dim,
    const float repetition_penalty,
    const int min_length,
    cudaStream_t stream,
    const int tensor_para_size = 1,
    const int layer_para_size = 1,
    const int layer_para_batch_size = 1) {
  auto input_dims = input.shape();
  int batch_size_ = input_dims[0];
  int start_len = input_dims[1];
  const int vocab_size = emb_bias.shape()[0];

  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  DecodingInitParam<DataType_> decoding_params;
  decoding_params.cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
  decoding_params.cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

#ifdef PADDLE_NEW_ALLOCATOR
  // For PaddlePaddle>=2.3.0
  decoding_params.output_ids = output_ids.data<int>();
#else
  decoding_params.output_ids = output_ids.mutable_data<int>(word_emb.place());
#endif

  typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
  decoding_params.stream = stream;
  fastertransformer::Allocator<AllocatorType::PD> allocator_(stream);

  const int hidden_unit = size_per_head * n_head;

#ifdef BUILD_GPT
  auto* model_para_desc = ModelParaDescFactory::CreateModelParaDesc(
      n_head,
      size_per_head,
      num_layer,
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
  tensor_parallel_param.local_head_num_ = n_head;
  tensor_parallel_param.local_hidden_units_ = hidden_unit;

  layer_parallel_param.rank = 0;
  layer_parallel_param.world_size = 1;
  layer_parallel_param.layers_per_group = num_layer;
  layer_parallel_param.local_batch_size = batch_size_;
  int seed = -1;
#endif

  DecodingGptJ<DecodingTraits_::OpType>* gptj_decoding;

  decoding_params.request_batch_size = batch_size_;
  decoding_params.max_input_len = start_len;
  decoding_params.request_input_len = start_len;
  decoding_params.request_output_len = max_len - start_len;

  decoding_params.d_start_ids = const_cast<int *>(input.data<int>());
  decoding_params.d_attn_mask =
      reinterpret_cast<DataType_*>(const_cast<data_t_ *>(attn_mask.data<data_t_>()));
  decoding_params.d_start_lengths = start_length.data<int>();

  gptj_decoding =
      new DecodingGptJ<DecodingTraits_::OpType>(allocator_,
                                               batch_size_,
                                               max_len,
                                               n_head,
                                               size_per_head,
                                               vocab_size,
                                               num_layer,
                                               bos_id,
                                               eos_id,
                                               topk,
                                               topp,
                                               temperature,
                                               tensor_para_size,
                                               layer_para_size,
                                               true, /*is_fuse_QKV*/
                                               repetition_penalty,  /*repetition_penalty*/
                                               seed,
                                               rotary_embedding_dim,
                                               min_length);

  gptj_decoding->set_tensor_parallel_param(tensor_parallel_param);
  gptj_decoding->set_layer_parallel_param(layer_parallel_param);

  DecoderInitParam<DataType_>* params =
      new DecoderInitParam<DataType_>[num_layer];

  for (int i = 0; i < self_ln_weight.size(); ++i) {
    // Allow python passing weights of all layers or only passing the
    // corresponding layers to save memory.
    int layer_idx = self_ln_weight.size() != num_layer
                        ? layer_parallel_param.rank *
                                  layer_parallel_param.layers_per_group +
                              i
                        : i;

    params[layer_idx].stream = stream;
    params[layer_idx].cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
    params[layer_idx].cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

    params[layer_idx].request_batch_size = batch_size_;
    params[layer_idx].request_max_mem_seq_len = start_len;

    params[layer_idx].self_layernorm.gamma =
        reinterpret_cast<const DataType_*>(self_ln_weight[i].data<data_t_>());
    params[layer_idx].self_layernorm.beta =
        reinterpret_cast<const DataType_*>(self_ln_bias[i].data<data_t_>());

    params[layer_idx].self_attention.query_weight.kernel =
        reinterpret_cast<const DataType_*>(self_q_weight[i].data<data_t_>());
    params[layer_idx].self_attention.query_weight.bias = nullptr;

    params[layer_idx].self_attention.attention_output_weight.kernel =
        reinterpret_cast<const DataType_*>(self_out_weight[i].data<data_t_>());
    params[layer_idx].self_attention.attention_output_weight.bias = nullptr;

    params[layer_idx].ffn.intermediate_weight.kernel =
        reinterpret_cast<const DataType_*>(ffn_inter_weight[i].data<data_t_>());
    params[layer_idx].ffn.intermediate_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_inter_bias[i].data<data_t_>());
    params[layer_idx].ffn.output_weight.kernel =
        reinterpret_cast<const DataType_*>(ffn_out_weight[i].data<data_t_>());
    params[layer_idx].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(ffn_out_bias[i].data<data_t_>());
  }

  decoding_params.layernorm.gamma =
      reinterpret_cast<const DataType_*>(decoder_ln_weight.data<data_t_>());
  decoding_params.layernorm.beta =
      reinterpret_cast<const DataType_*>(decoder_ln_bias.data<data_t_>());
  decoding_params.embedding_table =
      reinterpret_cast<const DataType_*>(word_emb.data<data_t_>());
  decoding_params.embedding_kernel =
      reinterpret_cast<const DataType_*>(emb_weight.data<data_t_>());
  decoding_params.embedding_bias =
      reinterpret_cast<const DataType_*>(emb_bias.data<data_t_>());

  gptj_decoding->forward_context(params, decoding_params);
  gptj_decoding->forward(params, decoding_params);

  delete gptj_decoding;
  delete[] params;

  return {output_ids};
}

std::vector<paddle::Tensor> GPTJCUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& emb_weight,
    const paddle::Tensor& emb_bias,
    paddle::Tensor& output_ids,
    const int topk,
    const float topp,
    const int max_len,
    const int n_head,
    const int size_per_head,
    const int num_layer,
    const int bos_id,
    const int eos_id,
    const float temperature,
    const int rotary_embedding_dim,
    const float repetition_penalty,
    const int min_length,
    const bool use_fp16 = false,
    const int tensor_para_size = 1,
    const int layer_para_size = 1,
    const int layer_para_batch_size = 1) {
        
  auto stream = word_embedding.stream();
  cublasSetStream(CublasHandle::GetInstance()->cublas_handle_, stream);

  if (use_fp16) {
    return gptj_kernel<paddle::DataType::FLOAT16>(input,
                                                 attn_mask,
                                                 start_length,
                                                 word_embedding,
                                                 self_ln_weight,
                                                 self_ln_bias,
                                                 self_q_weight,
                                                 self_out_weight,
                                                 ffn_inter_weight,
                                                 ffn_inter_bias,
                                                 ffn_out_weight,
                                                 ffn_out_bias,
                                                 decoder_ln_weight,
                                                 decoder_ln_bias,
                                                 emb_weight,
                                                 emb_bias,
                                                 output_ids,
                                                 topk,
                                                 topp,
                                                 max_len,
                                                 n_head,
                                                 size_per_head,
                                                 num_layer,
                                                 bos_id,
                                                 eos_id,
                                                 temperature,
                                                 rotary_embedding_dim,
                                                 repetition_penalty,
                                                 min_length,
                                                 stream,
                                                 tensor_para_size,
                                                 layer_para_size,
                                                 layer_para_batch_size);
  } else {
    return gptj_kernel<paddle::DataType::FLOAT32>(input,
                                                 attn_mask,
                                                 start_length,
                                                 word_embedding,
                                                 self_ln_weight,
                                                 self_ln_bias,
                                                 self_q_weight,
                                                 self_out_weight,
                                                 ffn_inter_weight,
                                                 ffn_inter_bias,
                                                 ffn_out_weight,
                                                 ffn_out_bias,
                                                 decoder_ln_weight,
                                                 decoder_ln_bias,
                                                 emb_weight,
                                                 emb_bias,
                                                 output_ids,
                                                 topk,
                                                 topp,
                                                 max_len,
                                                 n_head,
                                                 size_per_head,
                                                 num_layer,
                                                 bos_id,
                                                 eos_id,
                                                 temperature,
                                                 rotary_embedding_dim,
                                                 repetition_penalty,
                                                 min_length,
                                                 stream,
                                                 tensor_para_size,
                                                 layer_para_size,
                                                 layer_para_batch_size);
  }
}
