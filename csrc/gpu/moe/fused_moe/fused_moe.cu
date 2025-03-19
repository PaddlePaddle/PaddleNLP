// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// Ignore CUTLASS warnings about type punning

#pragma once

#include "cutlass/numeric_conversion.h"
#include "helper.h"
#include "moe/fused_moe_helper.h"

__global__ void compute_total_rows_before_expert_kernel(
    int* sorted_experts,
    const int64_t sorted_experts_len,
    const int64_t num_experts,
    int64_t* total_rows_before_expert) {
  // First, compute the global tid. We only need 1 thread per expert.
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts) return;

  // This should construct the last index where each expert occurs.
  total_rows_before_expert[expert] =
      find_total_elts_leq_target(sorted_experts, sorted_experts_len, expert);
}

void compute_total_rows_before_expert(int* sorted_indices,
                                      const int64_t total_indices,
                                      const int64_t num_experts,
                                      int64_t* total_rows_before_expert,
                                      cudaStream_t stream) {
  const int threads = std::min(int64_t(1024), num_experts);
  const int blocks = (num_experts + threads - 1) / threads;

  compute_total_rows_before_expert_kernel<<<blocks, threads, 0, stream>>>(
      sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

template <paddle::DataType T>
void FusedMoeKernel(const paddle::Tensor& input,
                    const paddle::Tensor& gate_weight,
                    const paddle::Tensor& ffn1_weight,
                    const paddle::optional<paddle::Tensor>& ffn1_scale,
                    const paddle::optional<paddle::Tensor>& ffn1_bias,
                    const paddle::Tensor& ffn2_weight,
                    const paddle::optional<paddle::Tensor>& ffn2_scale,
                    const paddle::optional<paddle::Tensor>& ffn2_bias,
                    const std::string& quant_method,
                    const int moe_topk,
                    const bool group_moe,
                    const bool norm_topk_prob,
                    paddle::Tensor* output) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto* output_data = output->data<data_t>();

  auto fp16_moe_gemm_runner = MoeGemmRunner<DataType_, DataType_>();
  auto int8_moe_gemm_runner = MoeGemmRunner<DataType_, uint8_t>();
  auto int4_moe_gemm_runner = MoeGemmRunner<DataType_, cutlass::uint4b_t>();

  using NvType = typename traits_::DataType;
  auto moe_compute = MoeHelper<data_t, NvType>(quant_method,
                                               &fp16_moe_gemm_runner,
                                               &int8_moe_gemm_runner,
                                               &int4_moe_gemm_runner);

  moe_compute.ComputeFFN(&input,
                         &gate_weight,
                         &ffn1_weight,
                         ffn1_scale ? ffn1_scale.get_ptr() : nullptr,
                         ffn1_bias ? ffn1_bias.get_ptr() : nullptr,
                         &ffn2_weight,
                         ffn2_scale ? ffn2_scale.get_ptr() : nullptr,
                         ffn2_bias ? ffn2_bias.get_ptr() : nullptr,
                         nullptr,
                         moe_topk,
                         group_moe,
                         norm_topk_prob,
                         1.0,  // ComputeFFN
                         "ffn",
                         output);
}


std::vector<paddle::Tensor> FusedExpertMoe(
    const paddle::Tensor& input,
    const paddle::Tensor& gate_weight,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_bias,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const std::string& quant_method,
    const int moe_topk,
    const bool norm_topk_prob,
    const bool group_moe) {
  const auto input_type = input.dtype();
  auto output = paddle::empty_like(input);

  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      FusedMoeKernel<paddle::DataType::BFLOAT16>(input,
                                                 gate_weight,
                                                 ffn1_weight,
                                                 ffn1_scale,
                                                 ffn1_bias,
                                                 ffn2_weight,
                                                 ffn2_scale,
                                                 ffn2_bias,
                                                 quant_method,
                                                 moe_topk,
                                                 group_moe,
                                                 norm_topk_prob,
                                                 &output);
      break;
    case paddle::DataType::FLOAT16:
      FusedMoeKernel<paddle::DataType::FLOAT16>(input,
                                                gate_weight,
                                                ffn1_weight,
                                                ffn1_scale,
                                                ffn1_bias,
                                                ffn2_weight,
                                                ffn2_scale,
                                                ffn2_bias,
                                                quant_method,
                                                moe_topk,
                                                group_moe,
                                                norm_topk_prob,
                                                &output);
      break;
    default:
      PD_THROW("Unsupported data type for FusedMoeKernel");
  }
  return {output};
}

std::vector<std::vector<int64_t>> FusedExpertMoeInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& gate_weight_shape,
    const std::vector<int64_t>& ffn1_weight_shape,
    const std::vector<int64_t>& ffn2_weight_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_bias_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_bias_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_scale_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> FusedExpertMoeInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& gate_weight_dtype,
    const paddle::DataType& ffn1_weight_dtype,
    const paddle::DataType& ffn2_weight_dtype,
    const paddle::optional<paddle::DataType>& ffn1_bias_dtype,
    const paddle::optional<paddle::DataType>& ffn1_scale_dtype,
    const paddle::optional<paddle::DataType>& ffn2_bias_dtype,
    const paddle::optional<paddle::DataType>& ffn2_scale_dtype) {
  return {input_dtype};
}


PD_BUILD_OP(fused_expert_moe)
    .Inputs({"input",
             "gate_weight",
             "ffn1_weight",
             "ffn2_weight",
             paddle::Optional("ffn1_bias"),
             paddle::Optional("ffn1_scale"),
             paddle::Optional("ffn2_bias"),
             paddle::Optional("ffn2_scale")})
    .Outputs({"output"})
    .Attrs({"quant_method:std::string",
            "moe_topk:int",
            "norm_topk_prob:bool",
            "group_moe:bool"})
    .SetKernelFn(PD_KERNEL(FusedExpertMoe))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedExpertMoeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedExpertMoeInferDtype));
