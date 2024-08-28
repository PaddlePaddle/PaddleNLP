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
#include "dtype.h"
#include "matmul_helper.h"
#include "my_types.h"
#include "paddle/extension.h"
git adtemplate <typename T>
void AvxCompute(const paddle::Tensor &x,
                const paddle::Tensor &weight,
                bool trans,
                const std::string alog,
                paddle::Tensor &out,
                xft::Matrix<T> &quantizedWeight,
                xft::Vector<float> &WeightScale,
                xft::Vector<float> &WeightZero,
                xft::Vector<float> &WeightSum,
                MMHelper *mmHelper) {
  auto out_data = out.data<float>();
  const float *x_data = reinterpret_cast<const float *>(x.data<float>());
  const float *bias_data = nullptr;
  int m = 1;
  for (int i = 0; i < x.shape().size() - 1; i++) {
    m = m * x.shape()[i];
  }
  int k = x.shape()[x.shape().size() - 1];
  int l = weight.shape()[1];
  int n = weight.shape()[1];
  
  mmHelper->compute(false,
                    m,
                    n,
                    k,
                    1.0f,
                    x_data,
                    k,
                    quantizedWeight.Data(),
                    WeightScale.Data(),
                    WeightZero.Data(),
                    WeightSum.Data(),
                    0.0,
                    out_data,
                    l);
};
template <typename T>
void AvxWeightOnly(const paddle::Tensor &x,
                   const paddle::Tensor &weight,
                   bool trans,
                   const std::string alog,
                   paddle::Tensor &out) {
  static std::unordered_map<std::string,
                            std::tuple<xft::Matrix<T> *,
                                       xft::Vector<float> *,
                                       xft::Vector<float> *,
                                       xft::Vector<float> *>>
      weight_only_hub;
  std::stringstream weights_addr;
  weights_addr << weight.data<float>() << alog;
  std::string weight_only_key = weights_addr.str();
  auto it_created = weight_only_hub.find(weight_only_key);
  static MMHelper *mmHelper;
  int rows = weight.shape()[0], cols = weight.shape()[1];
  xft::Vector<float> *WeightScale =
      new xft::Vector<float>();  // if weight is int8
  xft::Vector<float> *WeightZero =
      new xft::Vector<float>();  // if weight is int8
  xft::Vector<float> *WeightSum =
      new xft::Vector<float>();  // if weight is int8
  xft::Matrix<T> *quantizedWeight = new xft::Matrix<T>();
  if (it_created == weight_only_hub.end()) {
    auto weight_ptr = reinterpret_cast<const float *>(weight.data<float>());
    xft::Matrix<T> convertedWeight;
    mmHelper = new MMHelper(xft::DeviceKind::iCPU, 0);
    mmHelper->convertWeight(trans,
                            rows,
                            cols,
                            weight_ptr,
                            nullptr,
                            nullptr,
                            convertedWeight,
                            *WeightScale,
                            *WeightZero,
                            *WeightSum);
    quantizedWeight->Resize(rows, cols);
    mmHelper->packWeight(trans, convertedWeight, *quantizedWeight);
    weight_only_hub[weight_only_key] =
        std::make_tuple(quantizedWeight, WeightScale, WeightZero, WeightSum);
    AvxCompute<T>(x,
                  weight,
                  trans,
                  alog,
                  out,
                  *quantizedWeight,
                  *WeightScale,
                  *WeightZero,
                  *WeightSum,
                  mmHelper);
  } else {
    AvxCompute<T>(x,
                  weight,
                  trans,
                  alog,
                  out,
                  *(std::get<0>(it_created->second)),
                  *(std::get<1>(it_created->second)),
                  *(std::get<2>(it_created->second)),
                  *(std::get<3>(it_created->second)),
                  mmHelper);
  }
}
std::vector<paddle::Tensor> InvokeAvxWeightOnly(const paddle::Tensor &x,
                                                const paddle::Tensor &weight,
                                                const std::string &alog,
                                                bool trans) {
  auto out_shape = x.shape();
  out_shape[out_shape.size() - 1] = weight.shape()[1];
  auto out = paddle::empty(out_shape, x.dtype(), paddle::CPUPlace());
  if (alog == "int8") {
    AvxWeightOnly<int8_t>(x, weight, trans, alog, out);
  } else if (alog == "fp16") {
    AvxWeightOnly<float16_t>(x, weight, trans, alog, out);
  } else {
    AvxWeightOnly<float16_t>(x, weight, trans, alog, out);
  }
  return {out};
}

std::vector<std::vector<int64_t>> AvxWeightOnlyInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> weigh_shape) {
  int m = 1;
  for (int i = 0; i < x_shape.size() - 1; i++) {
    m = m * x_shape[i];
  }
  return {std::vector<int64_t>{m, weigh_shape[1]}};
}

std::vector<paddle::DataType> AvxWeightOnlyInferDtype(
    paddle::DataType x_dtype,
    paddle::DataType weight_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(avx_weight_only)
    .Inputs({"x", "weight"})
    .Outputs({"out"})
    .Attrs({"alog: std::string", "trans:bool"})
    .SetKernelFn(PD_KERNEL(InvokeAvxWeightOnly))
    .SetInferShapeFn(PD_INFER_SHAPE(AvxWeightOnlyInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AvxWeightOnlyInferDtype));
