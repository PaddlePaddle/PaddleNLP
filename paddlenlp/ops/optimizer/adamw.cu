// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/extension.h"


template <typename T, typename MT>
__global__ void AdamKernelREG(MT beta1,
                              MT beta2,
                              MT epsilon,
                              MT beta1_pow_,
                              MT beta2_pow_,
                              const MT* moment1,
                              MT* moment1_out,
                              const MT* moment2,
                              MT* moment2_out,
                              const MT* lr_,
                              MT weight_decay,
                              MT lr_ratio,
                              const T* grad,
                              const T* param,
                              T* param_out,
                              const MT* master_param,
                              MT* master_param_out,
                              int ndim) {
  MT lr = *lr_ * lr_ratio;
  MT lr_orig = lr;
  MT beta1_pow = beta1_pow_;
  MT beta2_pow = beta2_pow_;

  lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
        (static_cast<MT>(1.0) - beta1_pow);

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = moment1[id];
    MT mom2 = moment2[id];
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;
    p -= lr_orig * weight_decay * p;
    p -= lr * (mom1 /
               (sqrt(mom2) + epsilon * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename MT>
__global__ void AdamKernelMEM(MT beta1,
                              MT beta2,
                              MT epsilon,
                              const MT* beta1_pow_,
                              const MT* beta2_pow_,
                              const MT* moment1,
                              MT* moment1_out,
                              const MT* moment2,
                              MT* moment2_out,
                              const MT* lr_,
                              MT weight_decay,
                              MT lr_ratio,
                              const T* grad,
                              const T* param,
                              T* param_out,
                              const MT* master_param,
                              MT* master_param_out,
                              int ndim) {
  MT lr = *lr_ * lr_ratio;
  MT lr_orig = lr;
  MT beta1_pow = *beta1_pow_;
  MT beta2_pow = *beta2_pow_;

  lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
        (static_cast<MT>(1.0) - beta1_pow);

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;
    p -= lr_orig * weight_decay * p;
    p -= lr * (mom1 /
               (sqrt(mom2) + epsilon * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T>
__global__ void UpdateBetaPow(T beta1,
                              T beta2,
                              const T* beta1_pow_,
                              const T* beta2_pow_,
                              T* beta1_pow_out,
                              T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}


std::vector<paddle::Tensor> adam_cuda_forward(
    // Tensor inputs
    const paddle::Tensor& Param,
    const paddle::Tensor& Grad,
    const paddle::Tensor& LearningRate,
    const paddle::Tensor& Moment1,
    const paddle::Tensor& Moment2,
    const paddle::Tensor& Beta1Pow,
    const paddle::Tensor& Beta2Pow,
    // const paddle::Tensor& Beta1Tensor,
    // const paddle::Tensor& Beta2Tensor,
    // const paddle::Tensor& MasterParam,

    // Attrs inputs
    float beta1,
    float beta2,
    float epsilon,
    bool lazy_mode,
    int64_t min_row_size_to_use_multithread,
    bool multi_precision,
    float weight_decay,
    float lr_ratio) {
  auto ParamOut = paddle::Tensor(paddle::PlaceType::kGPU);
  auto Moment1Out = paddle::Tensor(paddle::PlaceType::kGPU);
  auto Moment2Out = paddle::Tensor(paddle::PlaceType::kGPU);
  auto Beta1PowOut = paddle::Tensor(Beta1Pow.place());
  auto Beta2PowOut = paddle::Tensor(Beta2Pow.place());
  // auto MasterParamOut = paddle::Tensor(paddle::PlaceType::kGPU);

  ParamOut.reshape(Param.shape());
  Moment1Out.reshape(Moment1.shape());
  Moment2Out.reshape(Moment2.shape());
  Beta1PowOut.reshape(Beta1Pow.shape());
  Beta2PowOut.reshape(Beta2Pow.shape());

  PD_CHECK(Beta1PowOut.size() == 1,
           "beta1 pow output size should be 1, but received "
           "value is:",
           Beta1PowOut.size());
  PD_CHECK(Beta2PowOut.size() == 1,
           "beta2 pow output size should be 1, but received "
           "value is:",
           Beta2PowOut.size());

  PD_CHECK(Param.type() == paddle::DataType::FLOAT32,
           "Custom adam support fp32 for now.");

  using T = float;
  auto place = Param.place();
  T beta1_t = static_cast<T>(beta1);
  T beta2_t = static_cast<T>(beta2);
  T epsilon_t = static_cast<T>(epsilon);
  T weight_decay_t = static_cast<T>(weight_decay);
  T lr_ratio_t = static_cast<T>(lr_ratio);

  int threads = 512;
  int blocks = (Param.size() + threads - 1) / threads;

  auto Moment1Out_data = Moment1Out.mutable_data<T>(place);
  auto Moment2Out_data = Moment2Out.mutable_data<T>(place);
  auto ParamOut_data = ParamOut.mutable_data<T>(place);

  if (Beta1Pow.place() == paddle::PlaceType::kCPU &&
      Beta2Pow.place() == paddle::PlaceType::kCPU) {
    // Compute with betapow in REG
    AdamKernelREG<T, T><<<blocks, threads, 0, Param.stream()>>>(
        beta1_t,
        beta2_t,
        epsilon_t,
        *Beta1Pow.data<T>(),
        *Beta2Pow.data<T>(),
        Moment1.data<T>(),
        Moment1Out_data,
        Moment2.data<T>(),
        Moment2Out_data,
        LearningRate.data<T>(),
        weight_decay_t,
        lr_ratio_t,
        Grad.data<T>(),
        Param.data<T>(),
        ParamOut_data,
        nullptr,
        nullptr,
        Param.size());
    // Cpu update
    Beta1PowOut.mutable_data<T>(Beta1Pow.place())[0] =
        beta1_t * Beta1Pow.data<T>()[0];
    Beta2PowOut.mutable_data<T>(Beta2Pow.place())[0] =
        beta2_t * Beta2Pow.data<T>()[0];
  } else {
    // Compute with betapow in MEM
    AdamKernelMEM<T, T><<<blocks, threads, 0, Param.stream()>>>(
        beta1_t,
        beta2_t,
        epsilon_t,
        Beta1Pow.data<T>(),
        Beta2Pow.data<T>(),
        Moment1.data<T>(),
        Moment1Out_data,
        Moment2.data<T>(),
        Moment2Out_data,
        LearningRate.data<T>(),
        weight_decay_t,
        lr_ratio_t,
        Grad.data<T>(),
        Param.data<T>(),
        ParamOut_data,
        nullptr,
        nullptr,
        int(Param.size()));
    // Update with gpu
    UpdateBetaPow<T><<<1, 32, 0, Param.stream()>>>(
        beta1_t,
        beta2_t,
        Beta1Pow.data<T>(),
        Beta2Pow.data<T>(),
        Beta1PowOut.mutable_data<T>(place),
        Beta2PowOut.mutable_data<T>(place));
  }

  return {ParamOut, Moment1Out, Moment2Out, Beta1PowOut, Beta2PowOut};
}
