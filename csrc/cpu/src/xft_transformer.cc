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


std::vector<paddle::Tensor> InvokeXftTransformer(
    const paddle::Tensor &input,
    const std::vector<paddle::Tensor> &ln1Gamma,
    const std::vector<paddle::Tensor> &qkvWeight,
    const std::vector<paddle::Tensor> &attnOutWeight,
    const std::vector<paddle::Tensor> &ln2Gamma,
    const std::vector<paddle::Tensor> &gateWeight,
    const std::vector<paddle::Tensor> &upWeight,
    const std::vector<paddle::Tensor> &downWeight,
    const paddle::Tensor &pastSeqLen,
    const paddle::Tensor &currentSeqLen,
    const paddle::Tensor &step,
    int hiddensize,
    int totalLayer,
    const std::string &computeType,
    const std::string &cacheType,
    const std::string &activation,
    const std::string &normType,
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
  auto output_ptr = reinterpret_cast<void *>(out.data<float>());
  auto xft_data_type = xft::DataType::fp16;
  if (computeType == "bf16") {
    xft_data_type = xft::DataType::bf16;
  } else if (computeType == "bf16_int8") {
    xft_data_type = xft::DataType::bf16_int8;
  } else if (computeType == "fp16_int8") {
    xft_data_type = xft::DataType::fp16_int8;
  }
  auto kvc_type = xft::DataType::fp16;
  if (cacheType == "int8") {
    kvc_type = xft::DataType::int8;
  }
  auto rope_type = xft::RopeType::LLAMA_ROPE;
  auto act_type = xft::ActivationType::SILU;
  if (activation == "relu") {
    act_type = xft::ActivationType::RELU;
  } else if (activation == "gelu") {
    act_type = xft::ActivationType::GELU;
  } else if (activation == "swiglu") {
    act_type = xft::ActivationType::SWIGLU;
  }
  auto norm_type = xft::NormType::RMS;
  if (normType == "layernorm") {
    norm_type = xft::NormType::LN;
  }
  auto input_ptr = reinterpret_cast<const void *>(input.data<float>());
  auto gate_bias_ptr = nullptr;
  auto up_bias_ptr = nullptr;
  auto down_bias_ptr = nullptr;
  auto ln2Beta_ptr = nullptr;
  auto qkvBiasWeight_ptr = nullptr;
  auto attnOutBias_ptr = nullptr;
  auto ln1Beta_ptr = nullptr;
  for (int i = 0; i < totalLayer; ++i) {
    auto ln1Gamma_ptr =
        reinterpret_cast<const float *>(ln1Gamma[i].data<float>());
    auto qkvWeight_ptr =
        reinterpret_cast<const void *>(qkvWeight[i].data<float>());
    auto attnOutWeight_ptr =
        reinterpret_cast<const void *>(attnOutWeight[i].data<float>());
    auto ln2Gamma_ptr =
        reinterpret_cast<const float *>(ln2Gamma[i].data<float>());
    auto gate_weight_ptr =
        reinterpret_cast<const void *>(gateWeight[i].data<float>());
    auto up_weight_ptr =
        reinterpret_cast<const void *>(upWeight[i].data<float>());
    auto down_weight_ptr =
        reinterpret_cast<const void *>(downWeight[i].data<float>());
    invokeLayerLLaMA(
        xft_data_type,               // dt
        kvc_type,                    // kvcdt
        rope_type,                   // rope
        act_type,                    // at
        norm_type,                   // nt
        batchSize,                   // batchSize
        inputSeqLen,                 // inputSeqLen
        attHeadDim,                  // attHeadDim
        attHeadNum,                  // attHeadNum
        kvHeadNum,                   // kvHeadNum
        maxPositions,                // maxPositions
        maxPosEmbed,                 // maxPosEmbed
        past_seq_len,                // pastSeqLen
        cur_seq_len,                 // currentSeqLen
        step_id,                     // step
        hiddensize,                  // hiddenSize
        intermediateSize,            // intermediateSize
        (void *)output_ptr,          // output
        hiddensize,                  // outputStride
        input_ptr,                   // input
        hiddensize,                  // inputStride
        ln1Gamma_ptr,                // ln1Gamma
        ln1Beta_ptr,                 // ln1Beta
        qkvWeight_ptr,               // queryWeight
        qkvWeight_ptr + hiddensize,  // keyWeight
        qkvWeight_ptr + hiddensize + kvHeadNum * attHeadDim,  // valueWeight
        attnOutWeight_ptr,                                    // attnOutWeight
        ln2Gamma_ptr,                                         // ln2Gamma
        ln2Beta_ptr,                                          // ln2Beta
        gate_weight_ptr,
        up_weight_ptr,
        down_weight_ptr,
        nullptr,          // queryBias
        nullptr,          // keyBias
        nullptr,          // valueBias
        attnOutBias_ptr,  // attnOutBias
        qkvWeight_ptr,    // myqkvWeight
        gate_bias_ptr,
        up_bias_ptr,
        down_bias_ptr,
        qkvBiasWeight_ptr);
    if (i < totalLayer - 1) {
      memcpy(const_cast<void *>(input_ptr),
             output_ptr,
             batchSize * inputSeqLen * hiddensize * sizeof(float));
    }
  }
  return {out};
}

std::vector<std::vector<int64_t>> XftTransformerInferShape(
    std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> XftTransformerInferDtype(
    paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(xft_transformer)
    .Inputs({
        "x",
        paddle::Vec("ln1Gamma"),
        paddle::Vec("qkvWeight"),
        paddle::Vec("attnOutWeight"),
        paddle::Vec("ln2Gamma"),
        paddle::Vec("gateWeight"),
        paddle::Vec("upWeight"),
        paddle::Vec("downWeight"),
        "pastSeqLen",
        "currentSeqLen",
        "step",
    })
    .Outputs({"out"})
    .Attrs({"hiddensize :int",
            "totalLayer :int",
            "computeType : std::string",
            "cacheType :std::string",
            "activation :std::string",
            "normType :std::string",
            "attHeadDim: int",
            "attHeadNum: int",
            "kvHeadNum: int",
            "maxPositions: int",
            "maxPosEmbed: int",
            "intermediateSize: int"})
    .SetKernelFn(PD_KERNEL(InvokeXftTransformer))
    .SetInferShapeFn(PD_INFER_SHAPE(XftTransformerInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(XftTransformerInferDtype));