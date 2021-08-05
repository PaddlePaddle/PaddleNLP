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
    const paddle::Tensor& self_attn_query_weight,  // attention param
    const paddle::Tensor& self_attn_query_bias,
    const paddle::Tensor& self_attn_key_weight,
    const paddle::Tensor& self_attn_key_bias,
    const paddle::Tensor& self_attn_value_weight,
    const paddle::Tensor& self_attn_value_bias,
    const paddle::Tensor& self_attn_output_weight,  // attention output
    const paddle::Tensor& self_attn_output_bias,
    const paddle::Tensor& attr_output_layernorm_weight,  // two layer norm param
    const paddle::Tensor& attr_output_layernorm_bias,
    const paddle::Tensor& output_layernorm_weight,
    const paddle::Tensor& output_layernorm_bias,
    const paddle::Tensor& ffn_intermediate_weight,  // two layer ffn
    const paddle::Tensor& ffn_intermediate_bias,
    const paddle::Tensor& ffn_output_weight,
    const paddle::Tensor& ffn_output_bias,
    const paddle::Tensor& amax_list,
    paddle::Tensor& encoder_out,  // output
    int64_t head_num_,
    int64_t size_per_head_,
    int64_t int8_mode,  // no support now
    int64_t num_layer_,
    int64_t layer_idx_,
    bool allow_gemm_test,
    bool use_trt_kernel_,
    int64_t max_seq_len_,
    cublasHandle_t cublas_handle_,
    cudaStream_t stream) {
  int batch_size_ = input.shape()[0];
  typedef PDTraits<D> traits_;
  typedef BertEncoderTransformerTraits<traits_::OpType,
                                       cuda::OpenMultiHeadAttention>
      EncoderTraits_;
  // fastertransformer::Allocator<AllocatorType::PD> allocator_(stream);
  fastertransformer::Allocator<AllocatorType::PD>* allocator_ =
      new fastertransformer::Allocator<AllocatorType::PD>(
          stream);  //(context, stream);

  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  EncoderInitParam<DataType_> encoder_param;

  encoder_param.stream = stream;
  encoder_param.cublas_handle = cublas_handle_;

  // encoder_param.transformer_out =
  // encoder_out.mutable_data<float>(input.place());
  encoder_param.transformer_out =
      encoder_out.mutable_data<DataType_>(input.place());

  // self attn
  encoder_param.self_attention.query_weight.kernel =
      reinterpret_cast<const DataType_*>(
          self_attn_query_weight.data<data_t_>());
  encoder_param.self_attention.query_weight.bias =
      reinterpret_cast<const DataType_*>(self_attn_query_bias.data<data_t_>());
  encoder_param.self_attention.key_weight.kernel =
      reinterpret_cast<const DataType_*>(self_attn_key_weight.data<data_t_>());
  encoder_param.self_attention.key_weight.bias =
      reinterpret_cast<const DataType_*>(self_attn_key_bias.data<data_t_>());
  encoder_param.self_attention.value_weight.kernel =
      reinterpret_cast<const DataType_*>(
          self_attn_value_weight.data<data_t_>());
  encoder_param.self_attention.value_weight.bias =
      reinterpret_cast<const DataType_*>(self_attn_value_bias.data<data_t_>());

  encoder_param.self_attention.attention_output_weight.kernel =
      reinterpret_cast<const DataType_*>(
          self_attn_output_weight.data<data_t_>());
  encoder_param.self_attention.attention_output_weight.bias =
      reinterpret_cast<const DataType_*>(self_attn_output_bias.data<data_t_>());

  encoder_param.self_layernorm.gamma = reinterpret_cast<const DataType_*>(
      attr_output_layernorm_weight.data<data_t_>());
  encoder_param.self_layernorm.beta = reinterpret_cast<const DataType_*>(
      attr_output_layernorm_bias.data<data_t_>());

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

  if (int8_mode) {
    // encoder_param.amaxList = reinterpret_cast<const DataType_ *>(
    encoder_param.amaxList =
        reinterpret_cast<const float*>(amax_list.data<data_t_>());
    encoder_param.layer_num = num_layer_;
    encoder_param.layer_idx = layer_idx_;
  } else {
    encoder_param.amaxList = nullptr;
  }

  BertEncoderTransformer<EncoderTraits_>* encoder =
      new BertEncoderTransformer<EncoderTraits_>(int8_mode, allow_gemm_test);
  encoder->allocateBuffer(allocator_,
                          batch_size_,
                          max_seq_len_,
                          max_seq_len_,
                          head_num_,
                          size_per_head_);  //, use_trt_kernel_);
  encoder->initialize(encoder_param);
  encoder->forward();
  encoder->freeBuffer();
  delete allocator_;
  return {encoder_out};
}


std::vector<paddle::Tensor> EncoderCUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& self_attn_query_weight,  // attention param
    const paddle::Tensor& self_attn_query_bias,
    const paddle::Tensor& self_attn_key_weight,
    const paddle::Tensor& self_attn_key_bias,
    const paddle::Tensor& self_attn_value_weight,
    const paddle::Tensor& self_attn_value_bias,
    const paddle::Tensor& self_attn_output_weight,  // attention output
    const paddle::Tensor& self_attn_output_bias,
    const paddle::Tensor& attr_output_layernorm_weight,  // two layer norm param
    const paddle::Tensor& attr_output_layernorm_bias,
    const paddle::Tensor& output_layernorm_weight,
    const paddle::Tensor& output_layernorm_bias,
    const paddle::Tensor& ffn_intermediate_weight,  // two layer ffn
    const paddle::Tensor& ffn_intermediate_bias,
    const paddle::Tensor& ffn_output_weight,
    const paddle::Tensor& ffn_output_bias,
    const paddle::Tensor& amax_list,
    paddle::Tensor& encoder_out,  // output
    int64_t head_num,
    int64_t size_per_head,
    int64_t int8_mode,  // no support now
    int64_t num_layer,
    int64_t layer_idx,
    bool allow_gemm_test,
    bool use_trt_kernel,
    int64_t max_seq_len) {
  auto stream = input.stream();
  cublasHandle_t cublas_handle_;
  cublasCreate(&cublas_handle_);
  cublasSetStream(cublas_handle_, stream);

  //   paddle::Tensor ret = paddle::Tensor(input.place());
  std::vector<paddle::Tensor> ret;  // = paddle::Tensor(input.place());

  switch (input.type()) {
    case paddle::DataType::FLOAT16: {
      ret = encoder_kernel<paddle::DataType::FLOAT16>(
          input,
          self_attn_query_weight,  // attention param
          self_attn_query_bias,
          self_attn_key_weight,
          self_attn_key_bias,
          self_attn_value_weight,
          self_attn_value_bias,
          self_attn_output_weight,  // attention output
          self_attn_output_bias,
          attr_output_layernorm_weight,  // two layer norm param
          attr_output_layernorm_bias,
          output_layernorm_weight,
          output_layernorm_bias,
          ffn_intermediate_weight,  // two layer ffn
          ffn_intermediate_bias,
          ffn_output_weight,
          ffn_output_bias,
          amax_list,
          encoder_out,  // output
          head_num,
          size_per_head,
          int8_mode,  // no support now
          num_layer,
          layer_idx,
          allow_gemm_test,
          use_trt_kernel,
          max_seq_len,
          cublas_handle_,
          stream);
      break;
    }
    case paddle::DataType::FLOAT32: {
      ret = encoder_kernel<paddle::DataType::FLOAT32>(
          input,
          self_attn_query_weight,  // attention param
          self_attn_query_bias,
          self_attn_key_weight,
          self_attn_key_bias,
          self_attn_value_weight,
          self_attn_value_bias,
          self_attn_output_weight,  // attention output
          self_attn_output_bias,
          attr_output_layernorm_weight,  // two layer norm param
          attr_output_layernorm_bias,
          output_layernorm_weight,
          output_layernorm_bias,
          ffn_intermediate_weight,  // two layer ffn
          ffn_intermediate_bias,
          ffn_output_weight,
          ffn_output_bias,
          amax_list,
          encoder_out,  // output
          head_num,
          size_per_head,
          int8_mode,  // no support now
          num_layer,
          layer_idx,
          allow_gemm_test,
          use_trt_kernel,
          max_seq_len,
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
