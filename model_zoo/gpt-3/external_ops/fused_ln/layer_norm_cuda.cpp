/*  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
    Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
    This code is copied fron NVIDIA apex: 
    https://github.com/NVIDIA/apex with minor changes.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and 
    limitations under the License. */

#include "paddle/extension.h"
#include <vector>
#include <cassert>

namespace {

void compute_n1_n2(
    const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &normalized_shape,
    int& n1,
    int& n2) {
    int idiff = input_shape.size() - normalized_shape.size();
    n2 = 1;
    for (int i = 0;  i < (int)normalized_shape.size();  ++i) {
	    assert( input_shape[i+idiff] == normalized_shape[i] );
	    n2 *= normalized_shape[i];
    }
    n1 = 1;
    for (int i = 0;  i < idiff;  ++i) {
	    n1 *= input_shape[i];
    }
}

void check_args(
    const std::vector<int64_t> &normalized_shape,
    const paddle::Tensor &gamma,
    const paddle::Tensor &beta
    )
{
    PD_CHECK(!gamma.initialized() || gamma.shape() == normalized_shape);
    PD_CHECK(!beta.initialized() || beta.shape() == normalized_shape);
}

void check_args(
    const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &normalized_shape,
    int& n1,
    int& n2
    )
{
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape.size()="
         << normalized_ndim;
      throw std::runtime_error(ss.str());
    }

    int64_t input_ndim = input_shape.size();
    
    auto is_valid = [&]{
        if (input_ndim < normalized_ndim) return false;
        int64_t offset = input_ndim - normalized_ndim;
        for (int64_t i = offset; i < input_ndim; ++i) {
            if (input_shape[i] != normalized_shape[i - offset]) {
                return false;
            }
        }
        return true;
    };

    if (!is_valid()) {
      std::stringstream ss;
      ss << "Expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got wrong input";
      throw std::runtime_error(ss.str());
    }

    compute_n1_n2(input_shape,normalized_shape,n1,n2);
}

void check_args(
    const paddle::Tensor &input,
    const std::vector<int64_t> &normalized_shape,
    int& n1,
    int& n2
    ) {
    check_args(input.shape(), normalized_shape, n1, n2);
}


void check_args(
    const paddle::Tensor &input,
    const std::vector<int64_t> &normalized_shape,
    const paddle::Tensor &gamma,
    const paddle::Tensor &beta,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma,beta);
}
}

void cuda_layer_norm(
    paddle::Tensor* output,
    paddle::Tensor* mean,
    paddle::Tensor* invvar,
    const paddle::Tensor &input,
    int n1,
    int n2,
    const paddle::Tensor *gamma,
    const paddle::Tensor *beta,
    double epsilon);

#define CHECK_CUDA(x) PD_CHECK(!x.is_cpu(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x)

std::vector<paddle::Tensor> layer_norm_affine(
    const paddle::Tensor &input,
    const std::vector<int64_t> &normalized_shape,
    const paddle::Tensor &gamma,
    const paddle::Tensor &beta,
    double epsilon) {
  
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1, n2;
  check_args(input, normalized_shape, gamma, beta, n1, n2);
  
  auto place = input.place();
  auto output = paddle::empty(input.shape(), gamma.type(), place);
  auto mean = paddle::empty({n1}, paddle::DataType::FLOAT32, place);
  auto invvar = paddle::empty_like(mean);

  cuda_layer_norm(&output, &mean, &invvar, input, n1, n2, &gamma, &beta, epsilon);

  return {output, mean, invvar};

}


void cuda_layer_norm_gradient(
    const paddle::Tensor &dout,
    const paddle::Tensor &mean,
    const paddle::Tensor &invvar,
    const paddle::Tensor &input,
    int n1,
    int n2,
    const paddle::Tensor *gamma,
    const paddle::Tensor *beta,
    double epsilon,
    paddle::Tensor* grad_input,
    paddle::Tensor* grad_gamma,
    paddle::Tensor* grad_beta);

std::vector<paddle::Tensor> layer_norm_gradient_affine(
    const paddle::Tensor &dout,
    const paddle::Tensor &mean,
    const paddle::Tensor &invvar,
    const paddle::Tensor &input,
    const std::vector<int64_t> &normalized_shape,
    const paddle::Tensor &gamma,
    const paddle::Tensor &beta,
    double epsilon) {

  CHECK_INPUT(dout);
  CHECK_INPUT(mean);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1, n2;
  check_args(input, normalized_shape, gamma, beta, n1, n2);

  auto grad_input = paddle::empty_like(input);
  auto grad_gamma = paddle::empty_like(gamma);
  auto grad_beta = paddle::empty_like(beta);

  cuda_layer_norm_gradient(dout, mean, invvar, input, n1, n2,
      &gamma, &beta, epsilon,
      &grad_input, &grad_gamma, &grad_beta);

  return {grad_input, grad_gamma, grad_beta};

}

std::vector<paddle::Tensor> LnFwd(const paddle::Tensor &input,
    const paddle::Tensor &gamma,
    const paddle::Tensor &beta,
    float epsilon) {
  return layer_norm_affine(input, gamma.shape(), gamma, beta, epsilon);
}

std::vector<std::vector<int64_t>> LnFwdInferShape(std::vector<int64_t> input_shape,
    std::vector<int64_t> gamma_shape, std::vector<int64_t> beta_shape,
    float epsilon) {
  int n1, n2;
  check_args(input_shape, gamma_shape, n1, n2);
  return {input_shape, {n1}, {n2}};
}

std::vector<paddle::DataType> LnFwdInferDtype(paddle::DataType input_dtype,
    paddle::DataType gamma_dtype, paddle::DataType beta_dtype) {
  return {gamma_dtype, paddle::DataType::FLOAT32, paddle::DataType::FLOAT32};
}

std::vector<paddle::Tensor> LnBwd(const paddle::Tensor &input,
    const paddle::Tensor &gamma,
    const paddle::Tensor &beta,
    const paddle::Tensor &mean,
    const paddle::Tensor &invvar,
    const paddle::Tensor &dout,
    float epsilon) {
  return layer_norm_gradient_affine(dout, mean, invvar, input, gamma.shape(), gamma, beta, epsilon);
}

std::vector<std::vector<int64_t>> LnBwdInferShape(std::vector<int64_t> input_shape,
    std::vector<int64_t> gamma_shape, std::vector<int64_t> beta_shape,
    std::vector<int64_t> mean_shape, std::vector<int64_t> invvar_shape,
    std::vector<int64_t> dout_shape, float epsilon) {
  return {input_shape, gamma_shape, beta_shape};
}

PD_BUILD_OP(fused_ln)
    .Inputs({"x", "gamma", "beta"})
    .Outputs({"out", "mean", "invvar"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(LnFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(LnFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LnFwdInferDtype));

PD_BUILD_GRAD_OP(fused_ln)
    .Inputs({"x", "gamma", "beta", "mean", "invvar", paddle::Grad("out")})
    .Outputs({paddle::Grad("x"), paddle::Grad("gamma"), paddle::Grad("beta")})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(LnBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(LnBwdInferShape));

