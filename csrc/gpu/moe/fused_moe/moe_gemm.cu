// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "cutlass/numeric_conversion.h"
#include "helper.h"
#include "moe/fused_moe_helper.h"

template <paddle::DataType T>
void MoeGemmKernel(const paddle::Tensor& permute_input,
                  const paddle::Tensor& tokens_expert_prefix_sum,
                  const paddle::Tensor& ffn1_weight,
                  const paddle::optional<paddle::Tensor>& ffn1_bias,
                  const paddle::optional<paddle::Tensor>& ffn1_scale,
                  const std::string& quant_method,
                  paddle::Tensor ffn_out) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto permuted_data = permute_input.data<data_t>();
  auto place = permute_input.place();
  auto input_type = permute_input.dtype();
  auto stream = permute_input.stream();

  auto fp16_moe_gemm_runner = MoeGemmRunner<DataType_, DataType_>();
  auto int8_moe_gemm_runner = MoeGemmRunner<DataType_, uint8_t>();
  auto int4_moe_gemm_runner = MoeGemmRunner<DataType_, cutlass::uint4b_t>();

  const int64_t expanded_active_expert_rows = permute_input.dims()[0];
  const int num_experts = ffn1_weight.dims()[0];
  const int hidden_size = ffn1_weight.dims()[1];
  int inter_dim = ffn1_weight.dims()[2];

  if (quant_method == "weight_only_int4") {
    inter_dim = inter_dim * 2;
  }

  const int64_t inter_size = inter_dim;

  auto ffn_out_data = ffn_out.data<data_t>();

  using NvType = typename traits_::DataType;

  auto fc1_expert_biases =
      ffn1_bias
          ? const_cast<paddle::Tensor*>(ffn1_bias.get_ptr())->data<data_t>()
          : nullptr;

  if (quant_method == "weight_only_int8") {
    int8_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permuted_data),
        reinterpret_cast<const uint8_t*>(ffn1_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(
            const_cast<paddle::Tensor*>(ffn1_scale.get_ptr())->data<data_t>()),
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        expanded_active_expert_rows,
        inter_size,
        hidden_size,
        num_experts,
        "none",
        stream);
  } else if (quant_method == "weight_only_int4") {
    int4_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permuted_data),
        reinterpret_cast<const cutlass::uint4b_t*>(ffn1_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(
            const_cast<paddle::Tensor*>(ffn1_scale.get_ptr())->data<data_t>()),
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        expanded_active_expert_rows,
        inter_size,
        hidden_size,
        num_experts,
        "none",
        stream);
  } else {
    fp16_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permuted_data),
        reinterpret_cast<const NvType*>(ffn1_weight.data<data_t>()),
        nullptr,
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        expanded_active_expert_rows,
        inter_size,
        hidden_size,
        num_experts,
        "none",
        stream);
  }
}

std::vector<paddle::Tensor> MoeGemmFFN(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& ffn1_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const std::string& quant_method) {
  const auto input_type = permute_input.dtype();
  auto ffn_out = paddle::empty_like(permute_input);

  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      MoeGemmKernel<paddle::DataType::BFLOAT16>(permute_input,
                                               tokens_expert_prefix_sum,
                                               ffn1_weight,
                                               ffn1_bias,
                                               ffn1_scale,
                                               quant_method,
                                               ffn_out);
      break;
    case paddle::DataType::FLOAT16:
      MoeGemmKernel<paddle::DataType::FLOAT16>(permute_input,
                                              tokens_expert_prefix_sum,
                                              ffn1_weight,
                                              ffn1_bias,
                                              ffn1_scale,
                                              quant_method,
                                              ffn_out);
      break;
    default:
      PD_THROW("Unsupported data type for MoeGemmFFN");
  }
  return {ffn_out};
}

std::vector<std::vector<int64_t>> MoeGemmFFNInferShape(
    const std::vector<int64_t>& permute_input_shape,
    const std::vector<int64_t>& tokens_expert_prefix_sum_shape,
    const std::vector<int64_t>& ffn1_weight_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_bias_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_scale_shape) {
  return {permute_input_shape};
}

std::vector<paddle::DataType> MoeGemmFFNInferDtype(
    const paddle::DataType& permute_input_dtype,
    const paddle::DataType& tokens_expert_prefix_sum_dtype,
    const paddle::DataType& ffn1_weight_dtype,
    const paddle::optional<paddle::DataType>& ffn1_bias_dtype,
    const paddle::optional<paddle::DataType>& ffn1_scale_dtype) {
  return {permute_input_dtype};
}

PD_BUILD_OP(moe_gemm)
    .Inputs({"permute_input",
             "tokens_expert_prefix_sum",
             "ffn1_weight",
             paddle::Optional("ffn1_bias"),
             paddle::Optional("ffn1_scale")})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string"})
    .SetKernelFn(PD_KERNEL(MoeGemmFFN))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeGemmFFNInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeGemmFFNInferDtype));
