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
#include <string>
#include <vector>


#include "fusion_encoder_op.h"

std::vector<paddle::Tensor> EncoderForward(
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
    const int64_t& head_num,
    const int64_t& size_per_head,
    const int64_t& int8_mode,  // no support now
    const int64_t& num_layer,
    const int64_t& layer_idx,
    const bool& allow_gemm_test,
    const bool& use_trt_kernel,
    const int64_t& max_seq_len) {
  if (input.place() == paddle::PlaceType::kGPU) {
    auto shape = input.shape();
    std::vector<int64_t> output_dims = {shape[0], shape[1], shape[2]};
    auto encoder_out = paddle::Tensor(paddle::PlaceType::kGPU, output_dims);
    return EncoderCUDAForward(
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
        max_seq_len);
  } else {
    PD_THROW("Not implemented place. Only GPU is supported. ");
  }
}

std::vector<std::vector<int64_t>> EncoderInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>&
        self_attn_query_weight_shape,  // attention param
    const std::vector<int64_t>& self_attn_query_bias_shape,
    const std::vector<int64_t>& self_attn_key_weight_shape,
    const std::vector<int64_t>& self_attn_key_bias_shape,
    const std::vector<int64_t>& self_attn_value_weight_shape,
    const std::vector<int64_t>& self_attn_value_bias_shape,
    const std::vector<int64_t>&
        self_attn_output_weight_shape,  // attention output
    const std::vector<int64_t>& self_attn_output_bias_shape,
    const std::vector<int64_t>&
        attr_output_layernorm_weight_shape,  // two layer norm param
    const std::vector<int64_t>& attr_output_layernorm_bias_shape,
    const std::vector<int64_t>& output_layernorm_weight_shape,
    const std::vector<int64_t>& output_layernorm_bias_shape,
    const std::vector<int64_t>& ffn_intermediate_weight_shape,  // two layer ffn
    const std::vector<int64_t>& ffn_intermediate_bias_shape,
    const std::vector<int64_t>& ffn_output_weight_shape,
    const std::vector<int64_t>& ffn_output_bias_shape,
    const std::vector<int64_t>& amax_list_shape,
    const int64_t& head_num,
    const int64_t& size_per_head,
    const int64_t& int8_mode,  // no support now
    const int64_t& num_layer,
    const int64_t& layer_idx,
    const bool& allow_gemm_test,
    const bool& use_trt_kernel,
    const int64_t& max_seq_len) {
  return {{input_shape[0], input_shape[1], input_shape[2]}};
}


std::vector<paddle::DataType> EncoderInferDtype(
    const paddle::DataType& input,
    const paddle::DataType& self_attn_query_weight,  // attention param
    const paddle::DataType& self_attn_query_bias,
    const paddle::DataType& self_attn_key_weight,
    const paddle::DataType& self_attn_key_bias,
    const paddle::DataType& self_attn_value_weight,
    const paddle::DataType& self_attn_value_bias,
    const paddle::DataType& self_attn_output_weight,  // attention output
    const paddle::DataType& self_attn_output_bias,
    const paddle::DataType&
        attr_output_layernorm_weight,  // two layer norm param
    const paddle::DataType& attr_output_layernorm_bias,
    const paddle::DataType& output_layernorm_weight,
    const paddle::DataType& output_layernorm_bias,
    const paddle::DataType& ffn_intermediate_weight,  // two layer ffn
    const paddle::DataType& ffn_intermediate_bias,
    const paddle::DataType& ffn_output_weight,
    const paddle::DataType& ffn_output_bias,
    const paddle::DataType& amax_list) {
  switch (input) {
    case paddle::DataType::FLOAT16: {
      return {paddle::DataType::FLOAT16};
    }
    case paddle::DataType::FLOAT32: {
      return {paddle::DataType::FLOAT32};
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and float32 are supported. ");
      break;
    }
  }
}

PD_BUILD_OP(fusion_encoder)
    .Inputs({
        "Input",
        "SelfQueryWeight",
        "SelfQueryBias",
        "SelfKeyWeight",
        "SelfKeyBias",
        "SelfValueWeight",
        "SelfValueBias",
        "SelfAttnOutputWeight",
        "SelfAttnOutputBias",
        "SelfAttnOutputLayernormWeight",
        "SelfAttnOutputLayernormBias",
        "OutputLayernormWeight",
        "OutputLayernormBias",
        "FFNInterWeight",
        "FFNInterBias",
        "FFNOutputWeight",
        "FFNOutputBias",
        "AmaxList",
    })
    .Outputs({"encoder_out"})
    .Attrs({"head_num: int64_t",
            "size_per_head: int64_t",
            "int8_mode: int64_t",
            "num_layer: int64_t",
            "layer_idx: int64_t",
            "allow_gemm_test: bool",
            "use_trt_kernel: bool",
            "max_seq_len: int64_t"})
    .SetKernelFn(PD_KERNEL(EncoderForward))
    .SetInferShapeFn(PD_INFER_DTYPE(EncoderInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(EncoderInferDtype));