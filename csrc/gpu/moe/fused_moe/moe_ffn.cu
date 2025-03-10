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
// #include "cutlass/array.h"
// #include "cutlass/epilogue/thread/linear_combination.h"
// #include "cutlass/epilogue/thread/linear_combination_relu.h"
// #include "cutlass/gemm/device/gemm_grouped.h"
// #include "cutlass/gemm/gemm.h"
// #include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/numeric_conversion.h"
#include "moe/fused_moe_helper.h"

#include "helper.h"

template <paddle::DataType T>
void MoeFFNKernel(const paddle::Tensor& permute_input,
                  const paddle::Tensor& token_nums_per_expert,
                  const paddle::Tensor& ffn1_weight,
                  const paddle::Tensor& ffn2_weight,
                  const paddle::optional<paddle::Tensor>& ffn1_bias,
                  const paddle::optional<paddle::Tensor>& ffn1_scale,
                  const paddle::optional<paddle::Tensor>& ffn2_scale,
                  const std::string& quant_method,
                  paddle::Tensor ffn_out) {

  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto ffn_out_data = ffn_out.data<data_t>();
  auto permuted_data = permute_input.data<data_t>();
  auto place = permute_input.place();
  auto input_type = permute_input.dtype();
  auto stream = permute_input.stream();


  auto fp16_moe_gemm_runner =
      MoeGemmRunner<DataType_,
                    DataType_>();
  auto int8_moe_gemm_runner =
      MoeGemmRunner<DataType_, uint8_t>();
  auto int4_moe_gemm_runner =
      MoeGemmRunner<DataType_,
                    cutlass::uint4b_t>();

  const int64_t expanded_active_expert_rows = permute_input.dims()[0];
  const int num_experts = ffn1_weight.dims()[0];
  const int hidden_size = ffn1_weight.dims()[1];
  int inter_dim = ffn1_weight.dims()[2];

  if (quant_method == "weight_only_int4") {
    inter_dim = inter_dim * 2;
  }

  const int64_t inter_size = inter_dim;


  paddle::Tensor fc1_out_tensor = GetEmptyTensor(
                                          {expanded_active_expert_rows, inter_size},
                                          input_type,
                                          place);
  auto fc1_out = fc1_out_tensor.data<data_t>();

  using NvType = typename traits_::DataType;

  auto fc1_expert_biases = ffn1_bias ? const_cast<paddle::Tensor*>(ffn1_bias.get_ptr())->data<data_t>() : nullptr;

  if (quant_method == "weight_only_int8") {
    int8_moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permuted_data),
        reinterpret_cast<const uint8_t*>(ffn1_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(const_cast<paddle::Tensor*>(ffn1_scale.get_ptr())->data<data_t>()),
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
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
        reinterpret_cast<const NvType*>(const_cast<paddle::Tensor*>(ffn1_scale.get_ptr())->data<data_t>()),
        reinterpret_cast<const NvType*>(fc1_expert_biases),
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
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
        reinterpret_cast<NvType*>(fc1_out),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        inter_size,
        hidden_size,
        num_experts,
        "none",
        stream);
  }

  const int num_rows = expanded_active_expert_rows;



    // check
    paddle::Tensor act_out_tensor = GetEmptyTensor(
            {num_rows, inter_size / 2},
            input_type,
            place);
  
//   const std::string act_type = "swiglu";
  paddle::experimental::swiglu(fc1_out_tensor, act_out_tensor);
  auto act_out = act_out_tensor.data<data_t>();
//   paddle::phi::swiglu(&fc1_out_tensor, nullptr, &act_out_tensor);
//   auto bias_act_helper = BiasActHelper<T>(ctx, act_type, num_rows, inter_size);
//   bias_act_helper.Compute(&fc1_out_tensor, nullptr, &act_out_tensor);

  if (quant_method == "weight_only_int8") {
    int8_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const uint8_t*>(ffn2_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(const_cast<paddle::Tensor*>(ffn2_scale.get_ptr())->data<data_t>()),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        stream);

  } else if (quant_method == "weight_only_int4") {
    int4_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const cutlass::uint4b_t*>(ffn2_weight.data<int8_t>()),
        reinterpret_cast<const NvType*>(const_cast<paddle::Tensor*>(ffn2_scale.get_ptr())->data<data_t>()),
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        stream);
  } else {
    fp16_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out),
        reinterpret_cast<const NvType*>(ffn2_weight.data<data_t>()),
        nullptr,
        reinterpret_cast<NvType*>(ffn_out_data),
        const_cast<int64_t*>(token_nums_per_expert.data<int64_t>()),
        expanded_active_expert_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        stream);
  }
}



std::vector<paddle::Tensor>  MoeExpertFFN(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& token_nums_per_expert,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const std::string& quant_method
) {
    const auto input_type = permute_input.dtype();
    auto ffn_out = paddle::empty_like(permute_input);

    switch (input_type) {
        case paddle::DataType::BFLOAT16:
            MoeFFNKernel<paddle::DataType::BFLOAT16>(
                permute_input,
                token_nums_per_expert,
                ffn1_weight,
                ffn2_weight,
                ffn1_bias,
                ffn1_scale,
                ffn2_scale,
                quant_method,
                ffn_out
            );
    }
    return {ffn_out};

}

PD_BUILD_OP(moe_expert_ffn)
    .Inputs({"permute_input", "token_nums_per_expert", "ffn1_weight", "ffn2_weight", paddle::Optional("ffn1_bias"), paddle::Optional("ffn1_scale"), paddle::Optional("ffn2_scale"),})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string"})
    .SetKernelFn(PD_KERNEL(MoeExpertFFN));
    // .SetInferShapeFn(PD_INFER_SHAPE(TrtLLMFusedMoeInferShape))
    // .SetInferDtypeFn(PD_INFER_DTYPE(TrtLLMFusedMoeInferDtype));