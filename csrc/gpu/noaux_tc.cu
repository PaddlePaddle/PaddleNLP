
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

#pragma once

#include <algorithm>
#include <optional>

#include "helper.h"
#include "noauxtc_kernel.h"

std::vector<paddle::Tensor> NoauxTc(paddle::Tensor& scores,
                                    paddle::Tensor& scores_with_bias,
                                    int n_group,
                                    int topk_group,
                                    int topk,
                                    float routed_scaling_factor) {
  auto input_shape = scores_with_bias.shape();
  int64_t num_tokens = input_shape[0];
  int64_t num_experts = input_shape[1];
  auto input_type = scores_with_bias.dtype();
  auto place = scores_with_bias.place();
  auto group_scores = paddle::empty({num_tokens, n_group}, input_type, place);
  auto stream = scores_with_bias.stream();

  invokeNoAuxTc<float>(reinterpret_cast<float*>(scores.data<float>()),
                       reinterpret_cast<float*>(group_scores.data<float>()),
                       reinterpret_cast<float*>(scores_with_bias.data<float>()),
                       num_tokens,
                       num_experts,
                       n_group,
                       topk_group,
                       topk,
                       routed_scaling_factor,
                       stream);

  return {scores};
}

std::vector<paddle::DataType> NoauxTcInferDtype(
    const paddle::DataType& scores_dtype,
    const paddle::DataType& scores_with_bias_dtype) {
  return {scores_dtype};
}

std::vector<std::vector<int64_t>> NoauxTcInferShape(
    const std::vector<int64_t>& scores_shape,
    const std::vector<int64_t>& gating_output_shape) {
  return {scores_shape};
}

PD_BUILD_OP(noaux_tc)
    .Inputs({"scores", "scores_with_bias"})
    .Outputs({"output_tensor"})
    .Attrs({"n_group: int",
            "topk_group: int",
            "topk:int",
            "routed_scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(NoauxTc))
    .SetInferShapeFn(PD_INFER_SHAPE(NoauxTcInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(NoauxTcInferDtype));