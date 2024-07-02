// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


#include "layers_decoder.h"
#include "paddle/extension.h"
#include "paddle/phi/core/kernel_registry.h"

std::vector<paddle::Tensor> InvokeLLaMALayer(
    const paddle::Tensor &input,
    const paddle::Tensor &ln1Gamma,
    const paddle::Tensor &qkvWeight,
    const paddle::Tensor &attnOutWeight,
    const paddle::Tensor &ln2Gamma,
    const paddle::Tensor &gateWeight,
    const paddle::Tensor &upWeight,
    const paddle::Tensor &downWeight,
    const paddle::Tensor &pastSeqLen,
    const paddle::Tensor &currentSeqLen,
    const paddle::Tensor &step,
    int hiddensize,
    int totalLayer,
    const std::string &computeType,
    const std::string &activation,
    const std::string &normType,
    int currentlayerId,
    int attHeadDim,
    int attHeadNum,
    int kvHeadNum,
    int maxPositions,
    int maxPosEmbed,
    int intermediateSize) {
  auto out = paddle::empty_like(input);
  auto batchSize = input.shape()[0];
  auto inputSeqLen = input.shape()[1];
  auto past_seq_len = pastSeqLen.data<int64_t>()[0];
  auto cur_seq_len = currentSeqLen.data<int64_t>()[0];
  auto step_id = step.data<int64_t>()[0];
  auto input_ptr = reinterpret_cast<const void *>(input.data<float>());
  auto ln1Gamma_ptr = reinterpret_cast<const float *>(ln1Gamma.data<float>());
  auto qkvWeight_ptr = reinterpret_cast<const void *>(qkvWeight.data<float>());
  auto attnOutWeight_ptr =
      reinterpret_cast<const void *>(attnOutWeight.data<float>());
  auto ln2Gamma_ptr = reinterpret_cast<const float *>(ln2Gamma.data<float>());
  auto gate_weight_ptr =
      reinterpret_cast<const void *>(gateWeight.data<float>());
  auto up_weight_ptr = reinterpret_cast<const void *>(upWeight.data<float>());
  auto down_weight_ptr =
      reinterpret_cast<const void *>(downWeight.data<float>());
  auto output_ptr = reinterpret_cast<void *>(out.data<float>());
  auto xft_data_type = xft::DataType::fp16;
  if (computeType == "bf16") {
    xft_data_type = xft::DataType::bf16;
  }
  auto xft_act_type = xft::ActivationType::SILU;
  if (activation == "relu") {
    xft_act_type = xft::ActivationType::RELU;
  } else if (activation == "gelu") {
    xft_act_type = xft::ActivationType::GELU;
  } else if (activation == "swiglu") {
    xft_act_type = xft::ActivationType::SWIGLU;
  }
  auto xft_norm_type = xft::NormType::RMS;
  if (normType == "layernorm") {
    xft_norm_type = xft::NormType::LN;
  }
  invokeLayerLLaMA(xft_data_type,
                   xft_act_type,
                   xft_norm_type,
                   currentlayerId,
                   totalLayer,
                   batchSize,
                   inputSeqLen,
                   attHeadDim,
                   attHeadNum,
                   kvHeadNum,
                   maxPositions,
                   maxPosEmbed,
                   past_seq_len,
                   cur_seq_len,
                   step_id,
                   hiddensize,
                   intermediateSize,
                   (void *)output_ptr,
                   hiddensize,
                   input_ptr,
                   hiddensize,
                   ln1Gamma_ptr,
                   nullptr,
                   qkvWeight_ptr,
                   qkvWeight_ptr + hiddensize,
                   qkvWeight_ptr + 2 * hiddensize,
                   attnOutWeight_ptr,
                   ln2Gamma_ptr,
                   nullptr,
                   gate_weight_ptr,
                   up_weight_ptr,
                   down_weight_ptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   qkvWeight_ptr);

  return {out};
}

std::vector<std::vector<int64_t>> LLaMALayerInferShape(
    std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> LLaMALayerInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(xft_llama_layer)
    .Inputs({
        "x",
        "ln1Gamma",
        "qkvWeight",
        "attnOutWeight",
        "ln2Gamma",
        "gateWeight",
        "upWeight",
        "downWeight",
        "pastSeqLen",
        "currentSeqLen",
        "step",
    })
    .Outputs({"out"})
    .Attrs({"hiddensize :int",
            "totalLayer :int",
            "computeType : std::string",
            "activation :std::string",
            "normType :std::string",
            "currentlayerId: int",
            "attHeadDim: int",
            "attHeadNum: int",
            "kvHeadNum: int",
            "maxPositions: int",
            "maxPosEmbed: int",
            "intermediateSize: int"})
    .SetKernelFn(PD_KERNEL(InvokeLLaMALayer))
    .SetInferShapeFn(PD_INFER_SHAPE(LLaMALayerInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LLaMALayerInferDtype));
