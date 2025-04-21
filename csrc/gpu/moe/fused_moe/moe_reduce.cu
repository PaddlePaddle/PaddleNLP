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

#include "helper.h"
#include "moe/fused_moe_helper.h"
#include "moe/fused_moe_op.h"

template <paddle::DataType T>
void MoeReduceKernel(const paddle::Tensor& ffn_out,
                     const paddle::Tensor& top_k_weight,
                     const paddle::Tensor& permute_indices_per_token,
                     const paddle::Tensor& top_k_indices,
                     const paddle::optional<paddle::Tensor>& ffn2_bias,
                     const bool norm_topk_prob,
                     const float routed_scaling_factor,
                     const int num_rows,
                     const int hidden_size,
                     const int topk,
                     paddle::Tensor* output) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  auto stream = ffn_out.stream();

  finalize_moe_routing_kernelLauncher(
      ffn_out.data<data_t>(),
      output->data<data_t>(),
      ffn2_bias ? ffn2_bias->data<data_t>() : nullptr,
      top_k_weight.data<float>(),
      permute_indices_per_token.data<int32_t>(),
      top_k_indices.data<int>(),
      num_rows,
      hidden_size,
      topk,
      static_cast<int>(1),
      norm_topk_prob,
      routed_scaling_factor,
      stream);
}


std::vector<paddle::Tensor> MoeExpertReduce(
    const paddle::Tensor& ffn_out,
    const paddle::Tensor& top_k_weight,
    const paddle::Tensor& permute_indices_per_token,
    const paddle::Tensor& top_k_indices,
    const paddle::optional<paddle::Tensor>& ffn2_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor) {
  const auto input_type = ffn_out.dtype();
  auto place = ffn_out.place();

  const int topk = top_k_indices.dims()[1];
  const int num_rows = ffn_out.dims()[0] / topk;
  const int hidden_size = ffn_out.dims()[1];

  auto output = GetEmptyTensor({num_rows, hidden_size}, input_type, place);

  // Avoids ‘invalid configuration argument’ when we launch the kernel.
  if (ffn_out.dims()[0] == 0) return {output};

  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      MoeReduceKernel<paddle::DataType::BFLOAT16>(ffn_out,
                                                  top_k_weight,
                                                  permute_indices_per_token,
                                                  top_k_indices,
                                                  ffn2_bias,
                                                  norm_topk_prob,
                                                  routed_scaling_factor,
                                                  num_rows,
                                                  hidden_size,
                                                  topk,
                                                  &output);
      break;
    case paddle::DataType::FLOAT16:
      MoeReduceKernel<paddle::DataType::BFLOAT16>(ffn_out,
                                                  top_k_weight,
                                                  permute_indices_per_token,
                                                  top_k_indices,
                                                  ffn2_bias,
                                                  norm_topk_prob,
                                                  routed_scaling_factor,
                                                  num_rows,
                                                  hidden_size,
                                                  topk,
                                                  &output);
      break;
    default:
      PD_THROW("Unsupported data type for MoeDispatchKernel");
  }
  return {output};
}


std::vector<std::vector<int64_t>> MoeExpertReduceInferShape(
    const std::vector<int64_t>& ffn_out_shape,
    const std::vector<int64_t>& top_k_weight_shape,
    const std::vector<int64_t>& permute_indices_per_token_shape,
    const std::vector<int64_t>& top_k_indices_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_bias_shape) {
  const int topk = top_k_indices_shape[1];
  std::vector<int64_t> fused_moe_out_shape = {ffn_out_shape[0] / topk,
                                              ffn_out_shape[1]};

  return {fused_moe_out_shape};
}

std::vector<paddle::DataType> MoeExpertReduceInferDtype(
    const paddle::DataType& ffn_out_dtype,
    const paddle::DataType& top_k_weight_dtype,
    const paddle::DataType& permute_indices_per_token_dtype,
    const paddle::DataType& top_k_indices_dtype,
    const paddle::optional<paddle::DataType>& ffn2_bias_dtype) {
  return {ffn_out_dtype};
}


PD_BUILD_OP(moe_expert_reduce)
    .Inputs({"ffn_out",
             "top_k_weight",
             "permute_indices_per_token",
             "top_k_indices",
             paddle::Optional("ffn2_bias")})
    .Outputs({"output"})
    .Attrs({"norm_topk_prob:bool", "routed_scaling_factor:float"})
    .SetKernelFn(PD_KERNEL(MoeExpertReduce))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertReduceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertReduceInferDtype));
