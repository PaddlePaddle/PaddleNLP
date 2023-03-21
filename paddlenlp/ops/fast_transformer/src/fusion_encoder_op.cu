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

#include "cublas_handle.h"
#include "fastertransformer/bert_encoder_transformer.h"

#ifndef CUB_NS_QUALIFIER
#define CUB_NS_QUALIFIER ::cub
#endif

#include "fastertransformer/cuda/cub/cub.cuh"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/standard_encoder.h"
#include "fusion_encoder_op.h"
#include "pd_traits.h"


namespace ft = fastertransformer;

template <paddle::DataType D>
std::vector<paddle::Tensor> encoder_kernel(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const std::vector<paddle::Tensor>& attn_query_weight,
    const std::vector<paddle::Tensor>& attn_query_bias,
    const std::vector<paddle::Tensor>& attn_key_weight,
    const std::vector<paddle::Tensor>& attn_key_bias,
    const std::vector<paddle::Tensor>& attn_value_weight,
    const std::vector<paddle::Tensor>& attn_value_bias,
    const std::vector<paddle::Tensor>& attn_output_weight,
    const std::vector<paddle::Tensor>& attn_output_bias,
    /*
    When calling BertEncoderTransformer(Post-Norm):
        norm1 coresponds to BertInitParam.self_layernorm
        norm2 coresponds to BertInitParam.ffn_layernorm
    When calling OpenEncoder(Pre-Norm):
        norm1 coresponds to EncoderInitParam.input_layernorm
        norm2 coresponds to EncoderInitParam.self_layernorm
    */
    const std::vector<paddle::Tensor>& norm1_weight,
    const std::vector<paddle::Tensor>& norm1_bias,
    const std::vector<paddle::Tensor>& norm2_weight,
    const std::vector<paddle::Tensor>& norm2_bias,
    const std::vector<paddle::Tensor>& ffn_intermediate_weight,
    const std::vector<paddle::Tensor>& ffn_intermediate_bias,
    const std::vector<paddle::Tensor>& ffn_output_weight,
    const std::vector<paddle::Tensor>& ffn_output_bias,
    // const paddle::Tensor& sequence_id_offset,
    // const paddle::Tensor& trt_seqlen_offset,
    // const paddle::Tensor& amax_list,
    std::vector<paddle::Tensor>& encoder_out,
    int64_t head_num_,
    int64_t size_per_head_,
    bool use_gelu,
    bool remove_padding,
    int64_t int8_mode,  // no support now
    int64_t num_layer_,
    int64_t layer_idx_,
    bool allow_gemm_test,
    bool use_trt_kernel_,
    bool normalize_before,
    cudaStream_t stream) {

  auto input_shape = input.shape();
  int batch_size_ = input_shape[0];
  int max_seq_len_ = input_shape[1];
  typedef ft::PDTraits<D> traits_;

  ft::Allocator<ft::AllocatorType::PD>* allocator_ =
      new ft::Allocator<ft::AllocatorType::PD>(stream);

  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  int in_id = 0;
  int layers = attn_query_weight.size();

  if (normalize_before == false) {
    typedef ft::BertEncoderTransformerTraits<traits_::OpType,
                                        ft::cuda::OpenMultiHeadAttention>
        EncoderTraits_;

    // Post-Normalization
    ft::BertInitParam<DataType_> encoder_param;

    encoder_param.stream = stream;
    encoder_param.cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
    encoder_param.cublaslt_handle =
        CublasHandle::GetInstance()->cublaslt_handle_;

    encoder_param.attr_mask =
        reinterpret_cast<const DataType_*>(attn_mask.data<data_t_>());

    ft::BertEncoderTransformer<EncoderTraits_>* encoder =
        new ft::BertEncoderTransformer<EncoderTraits_>(
            int8_mode, allow_gemm_test, use_gelu);

    encoder->allocateBuffer(allocator_,
                            batch_size_,
                            max_seq_len_,
                            max_seq_len_,
                            head_num_,
                            size_per_head_,
                            use_trt_kernel_);

    std::vector<data_t_*> enc_buf({
        encoder_out[0].mutable_data<data_t_>(input.place()),
        encoder_out[1].mutable_data<data_t_>(input.place())});

    for (int layer = 0; layer < layers; ++layer) {
        in_id = layer & 0x1;

        if (0 == layer) {
            encoder_param.from_tensor = reinterpret_cast<const DataType_*>(
                input.data<data_t_>());
            encoder_param.to_tensor = reinterpret_cast<const DataType_*>(
                input.data<data_t_>());
            encoder_param.transformer_out = reinterpret_cast<DataType_*>(
                enc_buf[1 - in_id]);
        } else {
            encoder_param.from_tensor = reinterpret_cast<DataType_*>(
                enc_buf[in_id]);
            encoder_param.to_tensor = reinterpret_cast<DataType_*>(
                enc_buf[in_id]);
            encoder_param.transformer_out = reinterpret_cast<DataType_*>(
                enc_buf[1 - in_id]);
        }

        // self attn
        encoder_param.self_attention.query_weight.kernel =
            reinterpret_cast<const DataType_*>(attn_query_weight[layer].data<data_t_>());
        encoder_param.self_attention.query_weight.bias =
            reinterpret_cast<const DataType_*>(attn_query_bias[layer].data<data_t_>());
        encoder_param.self_attention.key_weight.kernel =
            reinterpret_cast<const DataType_*>(attn_key_weight[layer].data<data_t_>());
        encoder_param.self_attention.key_weight.bias =
            reinterpret_cast<const DataType_*>(attn_key_bias[layer].data<data_t_>());
        encoder_param.self_attention.value_weight.kernel =
            reinterpret_cast<const DataType_*>(attn_value_weight[layer].data<data_t_>());
        encoder_param.self_attention.value_weight.bias =
            reinterpret_cast<const DataType_*>(attn_value_bias[layer].data<data_t_>());
        encoder_param.self_attention.attention_output_weight.kernel =
            reinterpret_cast<const DataType_*>(attn_output_weight[layer].data<data_t_>());
        encoder_param.self_attention.attention_output_weight.bias =
            reinterpret_cast<const DataType_*>(attn_output_bias[layer].data<data_t_>());

        // self_attn_layer_norm
        encoder_param.self_layernorm.gamma =
            reinterpret_cast<const DataType_*>(norm1_weight[layer].data<data_t_>());
        encoder_param.self_layernorm.beta =
            reinterpret_cast<const DataType_*>(norm1_bias[layer].data<data_t_>());
        encoder_param.ffn.intermediate_weight.kernel =
            reinterpret_cast<const DataType_*>(
                ffn_intermediate_weight[layer].data<data_t_>());
        encoder_param.ffn.intermediate_weight.bias =
            reinterpret_cast<const DataType_*>(
                ffn_intermediate_bias[layer].data<data_t_>());

        encoder_param.ffn.output_weight.kernel =
            reinterpret_cast<const DataType_*>(ffn_output_weight[layer].data<data_t_>());
        encoder_param.ffn.output_weight.bias =
            reinterpret_cast<const DataType_*>(ffn_output_bias[layer].data<data_t_>());

        // ffn_layer_norm
        encoder_param.ffn_layernorm.gamma =
            reinterpret_cast<const DataType_*>(norm2_weight[layer].data<data_t_>());
        encoder_param.ffn_layernorm.beta =
            reinterpret_cast<const DataType_*>(norm2_bias[layer].data<data_t_>());

        int valid_word_num;

        encoder_param.sequence_id_offset = nullptr;
        valid_word_num = batch_size_ * max_seq_len_;

        encoder_param.valid_word_num = valid_word_num;

        encoder_param.trt_seqlen_offset = nullptr;
        encoder_param.trt_seqlen_size = batch_size_ + 1;

        encoder_param.amaxList = nullptr;

        encoder->initialize(encoder_param);
        encoder->forward();
    }

    encoder->freeBuffer();

    delete allocator_;
    delete encoder;
  } else {
    typedef ft::OpenEncoderTraits<traits_::OpType, ft::cuda::OpenMultiHeadAttention>
        OpenEncoderTraits_;

    // Pre-Normalization
    ft::EncoderInitParam<DataType_> encoder_param;

    encoder_param.stream = stream;
    encoder_param.cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
    encoder_param.cublaslt_handle =
        CublasHandle::GetInstance()->cublaslt_handle_;
    encoder_param.attr_mask =
        reinterpret_cast<const DataType_*>(attn_mask.data<data_t_>());

    ft::OpenEncoder<OpenEncoderTraits_>* encoder =
        new ft::OpenEncoder<OpenEncoderTraits_>(
            int8_mode, allow_gemm_test, use_gelu);

    encoder->allocateBuffer(allocator_,
                            batch_size_,
                            max_seq_len_,
                            max_seq_len_,
                            head_num_,
                            size_per_head_,
                            use_trt_kernel_);
    
    for (int layer = 0; layer < layers; ++layer) {
        in_id = layer & 0x1;

        if (0 == layer) {
            encoder_param.from_tensor = reinterpret_cast<const DataType_*>(
                input.data<data_t_>());
            encoder_param.to_tensor = reinterpret_cast<const DataType_*>(
                input.data<data_t_>());
            encoder_param.transformer_out = reinterpret_cast<DataType_*>(
                encoder_out[1 - in_id].mutable_data<data_t_>(input.place()));
        } else {
            encoder_param.from_tensor = reinterpret_cast<DataType_*>(
                encoder_out[in_id].data<data_t_>());
            encoder_param.to_tensor = reinterpret_cast<DataType_*>(
                encoder_out[in_id].data<data_t_>());
            encoder_param.transformer_out = reinterpret_cast<DataType_*>(
                encoder_out[1 - in_id].mutable_data<data_t_>(input.place()));
        }

        // self attn
        encoder_param.self_attention.query_weight.kernel =
            reinterpret_cast<const DataType_*>(attn_query_weight[layer].data<data_t_>());
        encoder_param.self_attention.query_weight.bias =
            reinterpret_cast<const DataType_*>(attn_query_bias[layer].data<data_t_>());
        encoder_param.self_attention.key_weight.kernel =
            reinterpret_cast<const DataType_*>(attn_key_weight[layer].data<data_t_>());
        encoder_param.self_attention.key_weight.bias =
            reinterpret_cast<const DataType_*>(attn_key_bias[layer].data<data_t_>());
        encoder_param.self_attention.value_weight.kernel =
            reinterpret_cast<const DataType_*>(attn_value_weight[layer].data<data_t_>());
        encoder_param.self_attention.value_weight.bias =
            reinterpret_cast<const DataType_*>(attn_value_bias[layer].data<data_t_>());
        encoder_param.self_attention.attention_output_weight.kernel =
            reinterpret_cast<const DataType_*>(attn_output_weight[layer].data<data_t_>());
        encoder_param.self_attention.attention_output_weight.bias =
            reinterpret_cast<const DataType_*>(attn_output_bias[layer].data<data_t_>());

        // Spicific for Pre-Normalization
        encoder_param.input_layernorm.gamma =
            reinterpret_cast<const DataType_*>(norm1_weight[layer].data<data_t_>());
        encoder_param.input_layernorm.beta =
            reinterpret_cast<const DataType_*>(norm1_bias[layer].data<data_t_>());

        encoder_param.self_layernorm.gamma =
            reinterpret_cast<const DataType_*>(norm2_weight[layer].data<data_t_>());
        encoder_param.self_layernorm.beta =
            reinterpret_cast<const DataType_*>(norm2_bias[layer].data<data_t_>());

        encoder_param.ffn.intermediate_weight.kernel =
            reinterpret_cast<const DataType_*>(
                ffn_intermediate_weight[layer].data<data_t_>());
        encoder_param.ffn.intermediate_weight.bias =
            reinterpret_cast<const DataType_*>(
                ffn_intermediate_bias[layer].data<data_t_>());

        encoder_param.ffn.output_weight.kernel =
            reinterpret_cast<const DataType_*>(ffn_output_weight[layer].data<data_t_>());
        encoder_param.ffn.output_weight.bias =
            reinterpret_cast<const DataType_*>(ffn_output_bias[layer].data<data_t_>());

        int valid_word_num;
        encoder_param.sequence_id_offset = nullptr;
        valid_word_num = batch_size_ * max_seq_len_;

        encoder_param.valid_word_num = valid_word_num;

        encoder_param.trt_seqlen_offset =
            nullptr;  // trt_seqlen_offset.data<int>();
        encoder_param.trt_seqlen_size = batch_size_ + 1;

        encoder_param.amaxList = nullptr;

        encoder->initialize(encoder_param);
        encoder->forward();
    }

    encoder->freeBuffer();
    delete allocator_;
    delete encoder;
  }

  return {encoder_out[1 - in_id]};
}

std::vector<paddle::Tensor> EncoderCUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const std::vector<paddle::Tensor>& attn_query_weight,
    const std::vector<paddle::Tensor>& attn_query_bias,
    const std::vector<paddle::Tensor>& attn_key_weight,
    const std::vector<paddle::Tensor>& attn_key_bias,
    const std::vector<paddle::Tensor>& attn_value_weight,
    const std::vector<paddle::Tensor>& attn_value_bias,
    const std::vector<paddle::Tensor>& attn_output_weight,
    const std::vector<paddle::Tensor>& attn_output_bias,
    /*
    When calling BertEncoderTransformer(Post-Norm):
        norm1 coresponds to BertInitParam.self_layernorm
        norm2 coresponds to BertInitParam.ffn_layernorm
    When calling OpenEncoder(Pre-Norm):
        norm1 coresponds to EncoderInitParam.input_layernorm
        norm2 coresponds to EncoderInitParam.self_layernorm
    */
    const std::vector<paddle::Tensor>& norm1_weight,
    const std::vector<paddle::Tensor>& norm1_bias,
    const std::vector<paddle::Tensor>& norm2_weight,
    const std::vector<paddle::Tensor>& norm2_bias,
    const std::vector<paddle::Tensor>& ffn_intermediate_weight,
    const std::vector<paddle::Tensor>& ffn_intermediate_bias,
    const std::vector<paddle::Tensor>& ffn_output_weight,
    const std::vector<paddle::Tensor>& ffn_output_bias,
    // const paddle::Tensor& sequence_id_offset,
    // const paddle::Tensor& trt_seqlen_offset,
    // const paddle::Tensor& amax_list,
    std::vector<paddle::Tensor>& encoder_out,
    int64_t head_num,
    int64_t size_per_head,
    bool use_gelu,
    bool remove_padding,
    int64_t int8_mode,
    int64_t num_layer,
    int64_t layer_idx,
    bool allow_gemm_test,
    bool use_trt_kernel,
    bool normalize_before) {
  auto stream = input.stream();

  cublasSetStream(CublasHandle::GetInstance()->cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  switch (input.type()) {
    case paddle::DataType::FLOAT16: {
      ret = encoder_kernel<paddle::DataType::FLOAT16>(input,
                                                      attn_mask,
                                                      attn_query_weight,
                                                      attn_query_bias,
                                                      attn_key_weight,
                                                      attn_key_bias,
                                                      attn_value_weight,
                                                      attn_value_bias,
                                                      attn_output_weight,
                                                      attn_output_bias,
                                                      norm1_weight,
                                                      norm1_bias,
                                                      norm2_weight,
                                                      norm2_bias,
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
                                                      use_gelu,
                                                      remove_padding,
                                                      int8_mode,
                                                      num_layer,
                                                      layer_idx,
                                                      allow_gemm_test,
                                                      use_trt_kernel,
                                                      normalize_before,
                                                      stream);

      break;
    }
    case paddle::DataType::FLOAT32: {
      ret = encoder_kernel<paddle::DataType::FLOAT32>(input,
                                                      attn_mask,
                                                      attn_query_weight,
                                                      attn_query_bias,
                                                      attn_key_weight,
                                                      attn_key_bias,
                                                      attn_value_weight,
                                                      attn_value_bias,
                                                      attn_output_weight,
                                                      attn_output_bias,
                                                      norm1_weight,
                                                      norm1_bias,
                                                      norm2_weight,
                                                      norm2_bias,
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
                                                      use_gelu,
                                                      remove_padding,
                                                      int8_mode,
                                                      num_layer,
                                                      layer_idx,
                                                      allow_gemm_test,
                                                      use_trt_kernel,
                                                      normalize_before,
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