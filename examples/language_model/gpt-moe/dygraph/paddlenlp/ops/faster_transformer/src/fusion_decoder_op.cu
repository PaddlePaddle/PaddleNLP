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
#include "fusion_decoder_op.h"
#include "pd_traits.h"


template <paddle::DataType D>
std::vector<paddle::Tensor> decoder_kernel(
    const paddle::Tensor& from_tensor_input,
    const paddle::Tensor& memory_tensor_input,
    const paddle::Tensor& mem_seq_len_input,
    const paddle::Tensor& self_ln_weight,
    const paddle::Tensor& self_ln_bias,
    const paddle::Tensor& self_q_weight,
    const paddle::Tensor& self_q_bias,
    const paddle::Tensor& self_k_weight,
    const paddle::Tensor& self_k_bias,
    const paddle::Tensor& self_v_weight,
    const paddle::Tensor& self_v_bias,
    const paddle::Tensor& self_out_weight,
    const paddle::Tensor& self_out_bias,
    const paddle::Tensor& cross_ln_weight,
    const paddle::Tensor& cross_ln_bias,
    const paddle::Tensor& cross_q_weight,
    const paddle::Tensor& cross_q_bias,
    const paddle::Tensor& cross_k_weight,
    const paddle::Tensor& cross_k_bias,
    const paddle::Tensor& cross_v_weight,
    const paddle::Tensor& cross_v_bias,
    const paddle::Tensor& cross_out_weight,
    const paddle::Tensor& cross_out_bias,
    const paddle::Tensor& ffn_ln_weight,
    const paddle::Tensor& ffn_ln_bias,
    const paddle::Tensor& ffn_inter_weight,
    const paddle::Tensor& ffn_inter_bias,
    const paddle::Tensor& ffn_out_weight,
    const paddle::Tensor& ffn_out_bias,
    const paddle::Tensor& old_self_cache,
    const paddle::Tensor& old_mem_cache,
    paddle::Tensor& decoder_output_tensor,
    paddle::Tensor& new_self_cache,
    paddle::Tensor& new_mem_cache,
    int n_head,
    int size_per_head,
    cublasHandle_t cublas_handle_,
    cudaStream_t stream) {
  auto input_dims = memory_tensor_input.shape();
  const int batch_size_ = static_cast<int>(input_dims[0]);
  const int max_seq_len_ = static_cast<int>(input_dims[1]);
  const int memory_hidden_dim_ = static_cast<int>(input_dims[2]);

  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;
  typedef DecoderTransformerTraits<traits_::OpType> DecoderTraits_;
  OpenDecoder<DecoderTraits_::OpType>* decoder_;
  decoder_ = new OpenDecoder<DecoderTraits_::OpType>(
      batch_size_, max_seq_len_, n_head, size_per_head, memory_hidden_dim_);

  DataType_* decoder_output = reinterpret_cast<DataType_*>(
      decoder_output_tensor.mutable_data<data_t_>());
  DataType_* self_cache =
      reinterpret_cast<DataType_*>(old_self_cache.data<data_t_>());
  DataType_* memory_cache =
      reinterpret_cast<DataType_*>(old_mem_cache.data<data_t_>());
  const DataType_* from_tensor =
      reinterpret_cast<const DataType_*>(from_tensor_input.data<data_t_>());
  const DataType_* memory_tensor =
      reinterpret_cast<const DataType_*>(memory_tensor_input.data<data_t_>());
  const int* memory_sequence_length = mem_seq_len_input.data<int>();

  DecoderInitParam<DataType_> params;
  params.cublas_handle = cublas_handle_;
  params.stream = stream;
  fastertransformer::Allocator<AllocatorType::PD> allocator_(stream);

  params.self_layernorm.gamma =
      reinterpret_cast<const DataType_*>(self_ln_weight.data<data_t_>());
  params.self_layernorm.beta =
      reinterpret_cast<const DataType_*>(self_ln_bias.data<data_t_>());
  params.self_attention.query_weight.kernel =
      reinterpret_cast<const DataType_*>(self_q_weight.data<data_t_>());
  params.self_attention.query_weight.bias =
      reinterpret_cast<const DataType_*>(self_q_bias.data<data_t_>());
  params.self_attention.key_weight.kernel =
      reinterpret_cast<const DataType_*>(self_k_weight.data<data_t_>());
  params.self_attention.key_weight.bias =
      reinterpret_cast<const DataType_*>(self_k_bias.data<data_t_>());
  params.self_attention.value_weight.kernel =
      reinterpret_cast<const DataType_*>(self_v_weight.data<data_t_>());
  params.self_attention.value_weight.bias =
      reinterpret_cast<const DataType_*>(self_v_bias.data<data_t_>());
  params.self_attention.attention_output_weight.kernel =
      reinterpret_cast<const DataType_*>(self_out_weight.data<data_t_>());
  params.self_attention.attention_output_weight.bias =
      reinterpret_cast<const DataType_*>(self_out_bias.data<data_t_>());
  params.cross_layernorm.gamma =
      reinterpret_cast<const DataType_*>(cross_ln_weight.data<data_t_>());
  params.cross_layernorm.beta =
      reinterpret_cast<const DataType_*>(cross_ln_bias.data<data_t_>());
  params.cross_attention.query_weight.kernel =
      reinterpret_cast<const DataType_*>(cross_q_weight.data<data_t_>());
  params.cross_attention.query_weight.bias =
      reinterpret_cast<const DataType_*>(cross_q_bias.data<data_t_>());
  params.cross_attention.key_weight.kernel =
      reinterpret_cast<const DataType_*>(cross_k_weight.data<data_t_>());
  params.cross_attention.key_weight.bias =
      reinterpret_cast<const DataType_*>(cross_k_bias.data<data_t_>());
  params.cross_attention.value_weight.kernel =
      reinterpret_cast<const DataType_*>(cross_v_weight.data<data_t_>());
  params.cross_attention.value_weight.bias =
      reinterpret_cast<const DataType_*>(cross_v_bias.data<data_t_>());
  params.cross_attention.attention_output_weight.kernel =
      reinterpret_cast<const DataType_*>(cross_out_weight.data<data_t_>());
  params.cross_attention.attention_output_weight.bias =
      reinterpret_cast<const DataType_*>(cross_out_bias.data<data_t_>());
  params.ffn_layernorm.gamma =
      reinterpret_cast<const DataType_*>(ffn_ln_weight.data<data_t_>());
  params.ffn_layernorm.beta =
      reinterpret_cast<const DataType_*>(ffn_ln_bias.data<data_t_>());
  params.ffn.intermediate_weight.kernel =
      reinterpret_cast<const DataType_*>(ffn_inter_weight.data<data_t_>());
  params.ffn.intermediate_weight.bias =
      reinterpret_cast<const DataType_*>(ffn_inter_bias.data<data_t_>());
  params.ffn.output_weight.kernel =
      reinterpret_cast<const DataType_*>(ffn_out_weight.data<data_t_>());
  params.ffn.output_weight.bias =
      reinterpret_cast<const DataType_*>(ffn_out_bias.data<data_t_>());

  const int step = static_cast<int>(old_self_cache.shape()[1]);
  const int hidden_units = n_head * size_per_head;
  DataType_* K_cache = self_cache;
  DataType_* V_cache = self_cache + batch_size_ * step * hidden_units;
  DataType_* K_mem_cache = memory_cache;
  DataType_* V_mem_cache =
      memory_cache + batch_size_ * max_seq_len_ * hidden_units;

  const int decoder_buffer_size =
      decoder_->getWorkspaceSize() * sizeof(DataType_);
  DataType_* decoder_buffer =
      (DataType_*)allocator_.malloc(decoder_buffer_size);

  decoder_->initialize(params, decoder_buffer);
  decoder_->forward(from_tensor,
                    memory_tensor,
                    K_cache,
                    V_cache,
                    K_mem_cache,
                    V_mem_cache,
                    memory_sequence_length,
                    decoder_output,
                    step,
                    true);
  allocator_.free(decoder_buffer);
  delete decoder_;
  return {decoder_output_tensor, new_self_cache, new_mem_cache};
}

std::vector<paddle::Tensor> DecoderCUDAForward(
    const paddle::Tensor& from_tensor,
    const paddle::Tensor& memory_tensor,
    const paddle::Tensor& mem_seq_len,
    const paddle::Tensor& self_ln_weight,
    const paddle::Tensor& self_ln_bias,
    const paddle::Tensor& self_q_weight,
    const paddle::Tensor& self_q_bias,
    const paddle::Tensor& self_k_weight,
    const paddle::Tensor& self_k_bias,
    const paddle::Tensor& self_v_weight,
    const paddle::Tensor& self_v_bias,
    const paddle::Tensor& self_out_weight,
    const paddle::Tensor& self_out_bias,
    const paddle::Tensor& cross_ln_weight,
    const paddle::Tensor& cross_ln_bias,
    const paddle::Tensor& cross_q_weight,
    const paddle::Tensor& cross_q_bias,
    const paddle::Tensor& cross_k_weight,
    const paddle::Tensor& cross_k_bias,
    const paddle::Tensor& cross_v_weight,
    const paddle::Tensor& cross_v_bias,
    const paddle::Tensor& cross_out_weight,
    const paddle::Tensor& cross_out_bias,
    const paddle::Tensor& ffn_ln_weight,
    const paddle::Tensor& ffn_ln_bias,
    const paddle::Tensor& ffn_inter_weight,
    const paddle::Tensor& ffn_inter_bias,
    const paddle::Tensor& ffn_out_weight,
    const paddle::Tensor& ffn_out_bias,
    const paddle::Tensor& old_self_cache,
    const paddle::Tensor& old_mem_cache,
    paddle::Tensor& decoder_output,
    paddle::Tensor& new_self_cache,
    paddle::Tensor& new_mem_cache,
    int n_head,
    int size_per_head) {
  auto stream = memory_tensor.stream();
  cublasHandle_t cublas_handle_;
  cublasCreate(&cublas_handle_);
  cublasSetStream(cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  switch (memory_tensor.type()) {
    case paddle::DataType::FLOAT16: {
      ret = decoder_kernel<paddle::DataType::FLOAT16>(from_tensor,
                                                      memory_tensor,
                                                      mem_seq_len,
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
                                                      old_self_cache,
                                                      old_mem_cache,
                                                      decoder_output,
                                                      new_self_cache,
                                                      new_mem_cache,
                                                      n_head,
                                                      size_per_head,
                                                      cublas_handle_,
                                                      stream);
      break;
    }
    case paddle::DataType::FLOAT32: {
      ret = decoder_kernel<paddle::DataType::FLOAT32>(from_tensor,
                                                      memory_tensor,
                                                      mem_seq_len,
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
                                                      old_self_cache,
                                                      old_mem_cache,
                                                      decoder_output,
                                                      new_self_cache,
                                                      new_mem_cache,
                                                      n_head,
                                                      size_per_head,
                                                      cublas_handle_,
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
  return ret;
}
