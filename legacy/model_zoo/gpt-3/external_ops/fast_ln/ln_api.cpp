// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied fron NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#include "paddle/extension.h"

#include "ln.h"  // NOLINT

/*

Supported Type combinations:

input    compute   weights   output
=======================================
fp32     fp32      fp32      fp32
fp16     fp32      fp16      fp16
bf16     fp32      bf16      bf16
fp32     fp32      fp16      fp16
fp32     fp32      bf16      bf16

Remarks:
Output type = Weight type
Compute always in FP32

*/

namespace layer_norm {

// Create registries and provide runtime versions of config hash functions.

FwdRegistry FWD_FUNCS;
BwdRegistry BWD_FUNCS;

uint32_t get_type_id(paddle::DataType dtype) {
  if (dtype == paddle::DataType::FLOAT16) {
    return TypeToIdTrait<fp16>::Value;
  } else if (dtype == paddle::DataType::BFLOAT16) {
    return TypeToIdTrait<bf16>::Value;
  } else if (dtype == paddle::DataType::FLOAT32) {
    return TypeToIdTrait<float>::Value;
  } else {
    PD_CHECK(false, "Type not supported: ", dtype);
  }
}

uint64_t get_key(paddle::DataType weight_type,
                 paddle::DataType input_type,
                 paddle::DataType output_type,
                 paddle::DataType compute_type,
                 uint64_t hidden_size) {
  uint64_t type_key =
      get_type_id(weight_type) | (get_type_id(input_type) << 2) |  // NOLINT
      (get_type_id(output_type) << 4) | (get_type_id(compute_type) << 6);
  uint64_t launcher_key = (type_key << 32) | hidden_size;
  return launcher_key;
}

}  // namespace layer_norm

layer_norm::FwdFunction &get_fwd_launcher(paddle::DataType weight_type,
                                          paddle::DataType input_type,
                                          paddle::DataType output_type,
                                          paddle::DataType compute_type,
                                          uint32_t hidden_size) {
  auto iter = layer_norm::FWD_FUNCS.find(layer_norm::get_key(
      weight_type, input_type, output_type, compute_type, hidden_size));
  if (iter != layer_norm::FWD_FUNCS.end()) {
    return iter->second;
  } else {
    PD_CHECK(false,
             "FWD: Unsupported hidden_size or types: ",
             hidden_size,
             weight_type,
             input_type,
             output_type,
             compute_type);
  }
}

layer_norm::BwdFunction &get_bwd_launcher(paddle::DataType weight_type,
                                          paddle::DataType input_type,
                                          paddle::DataType output_type,
                                          paddle::DataType compute_type,
                                          uint32_t hidden_size) {
  auto iter = layer_norm::BWD_FUNCS.find(layer_norm::get_key(
      weight_type, input_type, output_type, compute_type, hidden_size));
  if (iter != layer_norm::BWD_FUNCS.end()) {
    return iter->second;
  } else {
    PD_CHECK(false,
             "BWD: Unsupported hidden_size or types: ",
             hidden_size,
             weight_type,
             input_type,
             output_type,
             compute_type);
  }
}

static cudaDeviceProp GetDevicePropImpl() {
  int device = -1;
  PD_CHECK(cudaGetDevice(&device) == cudaSuccess);
  cudaDeviceProp prop;
  PD_CHECK(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
  return prop;
}

static cudaDeviceProp *GetDeviceProp() {
  static auto prop = GetDevicePropImpl();
  return &prop;
}

std::vector<paddle::Tensor> LnFwd(const paddle::Tensor &x,
                                  const paddle::Tensor &scale,
                                  const paddle::Tensor &bias,
                                  const float epsilon) {
  auto input_type = x.type();
  auto weight_type = scale.type();
  auto output_type = weight_type;
  auto compute_type = paddle::DataType::FLOAT32;

  PD_CHECK(bias.type() == weight_type);

  PD_CHECK(!x.is_cpu());
  PD_CHECK(!scale.is_cpu());
  PD_CHECK(!bias.is_cpu());

  auto sizes = x.shape();
  PD_CHECK(sizes.size() >= 2);

  int rows = 1;
  for (size_t i = 0; i + 1 < sizes.size(); ++i) {
    rows *= sizes[i];
  }

  const int cols = sizes[sizes.size() - 1];
  auto hidden_size = scale.numel();

  PD_CHECK(scale.shape() == bias.shape());
  PD_CHECK(hidden_size == cols);

  PD_CHECK(epsilon >= 0.f);

  auto place = x.place();

  auto y = paddle::empty(sizes, output_type, place);

  auto mean = paddle::empty({rows}, compute_type, place);
  auto invvar = paddle::empty({rows}, compute_type, place);

  layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

  launch_params.props = GetDeviceProp();
  launch_params.stream = x.stream();

  // Request the kernel launcher.
  auto launcher = get_fwd_launcher(
      weight_type, input_type, output_type, compute_type, hidden_size);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  paddle::Tensor workspace, barrier;

  // Set the kernel runtime parameters.
  layer_norm::FwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = const_cast<void *>(x.data());
  params.mean = mean.data();
  params.invvar = invvar.data();
  params.scale = const_cast<void *>(scale.data());
  params.bias = const_cast<void *>(bias.data());
  params.y = y.data();
  params.epsilon = epsilon;

  if (launch_params.barrier_size > 0) {
    barrier = paddle::zeros(
        {launch_params.barrier_size}, paddle::DataType::INT32, place);
    workspace = paddle::empty(
        {launch_params.workspace_bytes}, paddle::DataType::UINT8, place);
    params.workspace = workspace.data();
    params.barrier = barrier.data<int>();
  }

  launcher(launch_params, false);

  return {y, mean, invvar};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<paddle::Tensor> LnBwd(const paddle::Tensor &x,
                                  const paddle::Tensor &scale,
                                  const paddle::Tensor &mean,
                                  const paddle::Tensor &invvar,
                                  const paddle::Tensor &dy,
                                  const float epsilon) {
  auto input_type = x.type();
  auto weight_type = scale.type();
  auto output_type = weight_type;
  auto compute_type = paddle::DataType::FLOAT32;

  PD_CHECK(dy.dtype() == output_type);
  PD_CHECK(mean.dtype() == compute_type);
  PD_CHECK(invvar.dtype() == compute_type);

  PD_CHECK(!x.is_cpu());
  PD_CHECK(!dy.is_cpu());
  PD_CHECK(!mean.is_cpu());
  PD_CHECK(!invvar.is_cpu());
  PD_CHECK(!scale.is_cpu());

  auto sizes = x.shape();
  PD_CHECK(sizes.size() >= 2);
  PD_CHECK(dy.shape() == sizes);

  int64_t rows = 1;
  for (size_t i = 0; i + 1 < sizes.size(); ++i) {
    rows *= sizes[i];
  }
  auto cols = sizes[sizes.size() - 1];

  auto hidden_size = scale.numel();

  PD_CHECK(mean.numel() == rows);
  PD_CHECK(mean.shape() == invvar.shape());

  PD_CHECK(scale.numel() == cols);

  auto dx = paddle::empty_like(x);
  auto dscale = paddle::empty_like(scale);
  auto dbias = paddle::empty_like(scale);

  layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
  launch_params.stream = x.stream();
  launch_params.props = GetDeviceProp();

  auto launcher = get_bwd_launcher(
      weight_type, input_type, output_type, compute_type, hidden_size);

  launcher(launch_params, true);

  auto place = x.place();
  auto dscale_part = paddle::empty(
      {launch_params.params.ctas_per_col, hidden_size}, compute_type, place);
  auto dbias_part = paddle::empty(
      {launch_params.params.ctas_per_col, hidden_size}, compute_type, place);
  paddle::Tensor workspace, barrier;

  layer_norm::BwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = const_cast<void *>(x.data());
  params.mean = const_cast<void *>(mean.data());
  params.invvar = const_cast<void *>(invvar.data());
  params.scale = const_cast<void *>(scale.data());
  params.dy = const_cast<void *>(dy.data());
  params.dx = dx.data();
  params.dbias = dbias.data();
  params.dscale = dscale.data();
  params.dbias_part = dbias_part.data();
  params.dscale_part = dscale_part.data();

  if (launch_params.barrier_size > 0) {
    barrier = paddle::zeros(
        {launch_params.barrier_size}, paddle::DataType::INT32, place);
    workspace = paddle::empty(
        {launch_params.workspace_bytes}, paddle::DataType::UINT8, place);
    params.workspace = workspace.data();
    params.barrier = barrier.data<int>();
  }

  launcher(launch_params, false);

  return {dx, dscale, dbias};
}

std::vector<std::vector<int64_t>> LnFwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    std::vector<int64_t> bias_shape,
    float epsilon) {
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < x_shape.size(); ++i) {
    rows *= x_shape[i];
  }
  return {x_shape, {rows}, {rows}};
}

std::vector<paddle::DataType> LnFwdInferDtype(paddle::DataType x_dtype,
                                              paddle::DataType scale_dtype,
                                              paddle::DataType bias_dtype) {
  return {scale_dtype, paddle::DataType::FLOAT32, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> LnBwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    std::vector<int64_t> mean_shape,
    std::vector<int64_t> invvar_shape,
    std::vector<int64_t> dy_shape,
    float epsilon) {
  return {x_shape, scale_shape, scale_shape};
}

PD_BUILD_OP(fast_ln)
    .Inputs({"x", "scale", "bias"})
    .Outputs({"y", "mean", "invvar"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(LnFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(LnFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LnFwdInferDtype));

PD_BUILD_GRAD_OP(fast_ln)
    .Inputs({"x", "scale", "mean", "invvar", paddle::Grad("y")})
    .Outputs({paddle::Grad("x"), paddle::Grad("scale"), paddle::Grad("bias")})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(LnBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(LnBwdInferShape));
