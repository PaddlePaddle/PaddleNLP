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
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/faster_transformer.h"
#include "fusion_encoder_op.h"
#include "pd_traits.h"


template <paddle::DataType D>
std::vector<paddle::Tensor> encoder_kernel(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_query_weight,
    const paddle::Tensor& attn_query_bias,
    const paddle::Tensor& attn_key_weight,
    const paddle::Tensor& attn_key_bias,
    const paddle::Tensor& attn_value_weight,
    const paddle::Tensor& attn_value_bias,
    const paddle::Tensor& attn_output_weight,
    const paddle::Tensor& attn_output_bias,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& attn_output_layernorm_weight,
    const paddle::Tensor& attn_output_layernorm_bias,
    const paddle::Tensor& output_layernorm_weight,
    const paddle::Tensor& output_layernorm_bias,
    const paddle::Tensor& ffn_intermediate_weight,
    const paddle::Tensor& ffn_intermediate_bias,
    const paddle::Tensor& ffn_output_weight,
    const paddle::Tensor& ffn_output_bias,
    // const paddle::Tensor& sequence_id_offset,
    // const paddle::Tensor& trt_seqlen_offset,
    // const paddle::Tensor& amax_list,
    paddle::Tensor& encoder_out,
    int64_t head_num_,
    int64_t size_per_head_,
    bool is_gelu,
    bool remove_padding,
    int64_t int8_mode,  // no support now
    int64_t num_layer_,
    int64_t layer_idx_,
    bool allow_gemm_test,
    bool use_trt_kernel_,
    bool normalize_before,
    cublasHandle_t cublas_handle_,
    cudaStream_t stream) {
  int batch_size_ = input.shape()[0];
  int max_seq_len_ = input.shape()[1];
  typedef PDTraits<D> traits_;
  typedef BertEncoderTransformerTraits<traits_::OpType,
                                       cuda::OpenMultiHeadAttention>
      EncoderTraits_;
  fastertransformer::Allocator<AllocatorType::PD>* allocator_ =
      new fastertransformer::Allocator<AllocatorType::PD>(stream);

  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  EncoderInitParam<DataType_> encoder_param;

  encoder_param.stream = stream;
  encoder_param.cublas_handle = cublas_handle_;
  encoder_param.from_tensor =
      reinterpret_cast<const DataType_*>(input.data<data_t_>());

  encoder_param.to_tensor =
      reinterpret_cast<const DataType_*>(input.data<data_t_>());
  encoder_param.transformer_out = reinterpret_cast<DataType_*>(
      encoder_out.mutable_data<data_t_>(input.place()));
  // self attn
  encoder_param.self_attention.query_weight.kernel =
      reinterpret_cast<const DataType_*>(attn_query_weight.data<data_t_>());
  encoder_param.self_attention.query_weight.bias =
      reinterpret_cast<const DataType_*>(attn_query_bias.data<data_t_>());
  encoder_param.self_attention.key_weight.kernel =
      reinterpret_cast<const DataType_*>(attn_key_weight.data<data_t_>());
  encoder_param.self_attention.key_weight.bias =
      reinterpret_cast<const DataType_*>(attn_key_bias.data<data_t_>());
  encoder_param.self_attention.value_weight.kernel =
      reinterpret_cast<const DataType_*>(attn_value_weight.data<data_t_>());
  encoder_param.self_attention.value_weight.bias =
      reinterpret_cast<const DataType_*>(attn_value_bias.data<data_t_>());
  encoder_param.attr_mask =
      reinterpret_cast<const DataType_*>(attn_mask.data<data_t_>());
  encoder_param.self_attention.attention_output_weight.kernel =
      reinterpret_cast<const DataType_*>(attn_output_weight.data<data_t_>());
  encoder_param.self_attention.attention_output_weight.bias =
      reinterpret_cast<const DataType_*>(attn_output_bias.data<data_t_>());
  encoder_param.self_layernorm.gamma = reinterpret_cast<const DataType_*>(
      attn_output_layernorm_weight.data<data_t_>());
  encoder_param.self_layernorm.beta = reinterpret_cast<const DataType_*>(
      attn_output_layernorm_bias.data<data_t_>());
  encoder_param.ffn.intermediate_weight.kernel =
      reinterpret_cast<const DataType_*>(
          ffn_intermediate_weight.data<data_t_>());
  encoder_param.ffn.intermediate_weight.bias =
      reinterpret_cast<const DataType_*>(ffn_intermediate_bias.data<data_t_>());

  encoder_param.ffn.output_weight.kernel =
      reinterpret_cast<const DataType_*>(ffn_output_weight.data<data_t_>());
  encoder_param.ffn.output_weight.bias =
      reinterpret_cast<const DataType_*>(ffn_output_bias.data<data_t_>());

  encoder_param.ffn_layernorm.gamma = reinterpret_cast<const DataType_*>(
      output_layernorm_weight.data<data_t_>());
  encoder_param.ffn_layernorm.beta =
      reinterpret_cast<const DataType_*>(output_layernorm_bias.data<data_t_>());
  int valid_word_num;
  //   if (remove_padding) {
  // valid_word_num = sequence_id_offset.shape()[0];
  // encoder_param.sequence_id_offset = sequence_id_offset.data<int>();
  //   } else {
  encoder_param.sequence_id_offset = nullptr;
  valid_word_num = batch_size_ * max_seq_len_;
  //   }
  encoder_param.valid_word_num = valid_word_num;

  encoder_param.trt_seqlen_offset = nullptr;  // trt_seqlen_offset.data<int>();
  encoder_param.trt_seqlen_size = batch_size_ + 1;
  //   static_cast<int>(trt_seqlen_offset.shape()[0]);
  //   int8_mode = 0;
  //   if (int8_mode != 0) {
  // encoder_param.amaxList =
  // reinterpret_cast<const float*>(amax_list.data<float>());
  // encoder_param.layer_num = num_layer_;
  // encoder_param.layer_idx = layer_idx_;
  //   } else {
  encoder_param.amaxList = nullptr;
  //   }

  BertEncoderTransformer<EncoderTraits_>* encoder =
      new BertEncoderTransformer<EncoderTraits_>(
          int8_mode, allow_gemm_test, normalize_before);

  encoder->allocateBuffer(allocator_,
                          batch_size_,
                          max_seq_len_,
                          max_seq_len_,
                          head_num_,
                          size_per_head_,
                          is_gelu,
                          use_trt_kernel_);
  encoder->initialize(encoder_param);
  encoder->forward();
  encoder->freeBuffer();
  delete allocator_;
  delete encoder;

  return {encoder_out};
}


std::vector<paddle::Tensor> EncoderCUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_query_weight,
    const paddle::Tensor& attn_query_bias,
    const paddle::Tensor& attn_key_weight,
    const paddle::Tensor& attn_key_bias,
    const paddle::Tensor& attn_value_weight,
    const paddle::Tensor& attn_value_bias,
    const paddle::Tensor& attn_output_weight,
    const paddle::Tensor& attn_output_bias,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& attn_output_layernorm_weight,
    const paddle::Tensor& attn_output_layernorm_bias,
    const paddle::Tensor& output_layernorm_weight,
    const paddle::Tensor& output_layernorm_bias,
    const paddle::Tensor& ffn_intermediate_weight,
    const paddle::Tensor& ffn_intermediate_bias,
    const paddle::Tensor& ffn_output_weight,
    const paddle::Tensor& ffn_output_bias,
    // const paddle::Tensor& sequence_id_offset,
    // const paddle::Tensor& trt_seqlen_offset,
    // const paddle::Tensor& amax_list,
    paddle::Tensor& encoder_out,
    int64_t head_num,
    int64_t size_per_head,
    bool is_gelu,
    bool remove_padding,
    int64_t int8_mode,
    int64_t num_layer,
    int64_t layer_idx,
    bool allow_gemm_test,
    bool use_trt_kernel,
    bool normalize_before) {
  auto stream = input.stream();
  cublasHandle_t cublas_handle_;
  cublasCreate(&cublas_handle_);
  cublasSetStream(cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  switch (input.type()) {
    case paddle::DataType::FLOAT16: {
      ret = encoder_kernel<paddle::DataType::FLOAT16>(
          input,
          attn_query_weight,
          attn_query_bias,
          attn_key_weight,
          attn_key_bias,
          attn_value_weight,
          attn_value_bias,
          attn_output_weight,
          attn_output_bias,
          attn_mask,
          attn_output_layernorm_weight,
          attn_output_layernorm_bias,
          output_layernorm_weight,
          output_layernorm_bias,
          ffn_intermediate_weight,
          ffn_intermediate_bias,
          ffn_output_weight,
          ffn_output_bias,
          //   sequence_id_offset,
          //   trt_seqlen_offset,
          //   amax_list,
          encoder_out,
          head_num,
          size_per_head,
          is_gelu,
          remove_padding,
          int8_mode,
          num_layer,
          layer_idx,
          allow_gemm_test,
          use_trt_kernel,
          normalize_before,
          cublas_handle_,
          stream);

      break;
    }
    case paddle::DataType::FLOAT32: {
      ret = encoder_kernel<paddle::DataType::FLOAT32>(
          input,
          attn_query_weight,
          attn_query_bias,
          attn_key_weight,
          attn_key_bias,
          attn_value_weight,
          attn_value_bias,
          attn_output_weight,
          attn_output_bias,
          attn_mask,
          attn_output_layernorm_weight,
          attn_output_layernorm_bias,
          output_layernorm_weight,
          output_layernorm_bias,
          ffn_intermediate_weight,
          ffn_intermediate_bias,
          ffn_output_weight,
          ffn_output_bias,
          //   sequence_id_offset,
          //   trt_seqlen_offset,
          //   amax_list,
          encoder_out,
          head_num,
          size_per_head,
          is_gelu,
          remove_padding,
          int8_mode,
          num_layer,
          layer_idx,
          allow_gemm_test,
          use_trt_kernel,
          normalize_before,
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
