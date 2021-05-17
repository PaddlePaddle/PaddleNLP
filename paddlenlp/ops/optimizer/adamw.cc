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

#include <iostream>
#include <vector>

#include "paddle/extension.h"

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
    float lr_ratio);

std::vector<paddle::Tensor> AdamForward(
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
  // TODO: Check Input
  if (Param.place() == paddle::PlaceType::kCPU) {
    PD_THROW("Not implemented.");
  } else if (Param.place() == paddle::PlaceType::kGPU) {
    return adam_cuda_forward(Param,
                             Grad,
                             LearningRate,
                             Moment1,
                             Moment2,
                             Beta1Pow,
                             Beta2Pow,
                             beta1,
                             beta2,
                             epsilon,
                             lazy_mode,
                             min_row_size_to_use_multithread,
                             multi_precision,
                             weight_decay,
                             lr_ratio);
  } else {
    PD_THROW("Not implemented.");
  }
}


std::vector<std::vector<int64_t>> AdamInferShape(
    std::vector<int64_t> param_shape,
    std::vector<int64_t> grad_shape,
    std::vector<int64_t> lr_shape,
    std::vector<int64_t> m1_shape,
    std::vector<int64_t> m2_shape,
    std::vector<int64_t> b1_shape,
    std::vector<int64_t> b2_shape) {
  return {param_shape, m1_shape, m2_shape, b1_shape, b2_shape};
}

std::vector<paddle::DataType> AdamInferDtype(paddle::DataType param_dtype,
                                             paddle::DataType grad_dtype,
                                             paddle::DataType lr_dtype,
                                             paddle::DataType m1_dtype,
                                             paddle::DataType m2_dtype,
                                             paddle::DataType b1_dtype,
                                             paddle::DataType b2_dtype) {
  return {param_dtype, m1_dtype, m2_dtype, b1_dtype, b2_dtype};
}


PD_BUILD_OP(adamw)
    .Inputs({
        "Param",         // "(Tensor) Input parameter"
        "Grad",          // "(Tensor) Input gradient"
        "LearningRate",  // "(Tensor) Learning rate"
        "Moment1",       // "(Tensor) Input first moment"
        "Moment2",       // "(Tensor) Input second moment"
        "Beta1Pow",      // "(Tensor) Input beta1 power accumulator"
        "Beta2Pow",      // "(Tensor) Input beta2 power accumulator"
        // "Beta1Tensor",    // "(Tensor<float32>, optional) If provided, Adam
        // will use this as beta1, this has a higher priority than attr(beta1),
        // the shape of this tensor MUST BE [1].").AsDispensable();
        // "Beta2Tensor",    // "(Tensor<float32>, optional) If provided, Adam
        // will use this as beta2, this has a higher priority than attr(beta2),
        // the shape of this tensor MUST BE [1].").AsDispensable();
        // "MasterParam",   // "FP32 master weight for AMP.").AsDispensable()
    })
    .Outputs({
        "ParamOut",     //  "(Tensor) Output parameter");
        "Moment1Out",   //  "(Tensor) Output first moment");
        "Moment2Out",   //  "(Tensor) Output second moment");
        "Beta1PowOut",  //  "(Tensor) Output beta1 power accumulator");
        "Beta2PowOut",  //  "(Tensor) Output beta2 power accumulator");
        // "MasterParamOut" // "The updated FP32 master weight for AMP. It
        // shared memory with Input(MasterParam).").AsDispensable();
    })
    .Attrs({
        "beta1: float",  // "(float, default 0.9) " "Exponential decay rate for
                         // the ""first moment estimates.").SetDefault(0.9f);
        "beta2: float",  // "(float, default 0.999) ""exponential decay rate for
                         // the ""second moment estimates.").SetDefault(0.999f);
        "epsilon: float",   // "(float, default 1.0e-8) ""Constant for numerical
                            // stability").SetDefault(1.0e-8f);
        "lazy_mode: bool",  // "(bool, default false) ""only update the
                            // parameter that has gradient in sparse
                            // update").SetDefault(false);
        "min_row_size_to_use_multithread: int64_t",  // "(int64_t, default 0)
                                                     // ""when not zero, if
                                                     // param row size is larger
                                                     // then
        // ""min_row_size_to_use_multithread
        // and
        // ""inner_op_parallelism
        // is larger then 0, sparse
        // update ""will run in
        // multithread
        // mode").SetDefault(1000);
        "multi_precision: bool",  // "(bool, default false) ""Whether to use
                                  // multi-precision during weight
                                  // updating.").SetDefault(false);
        "weight_decay: float",    // "(float, default 0.0) ""Weight decay
                                  // rate.").SetDefault(0.0f);
        "lr_ratio: float",        // "(float, default 1.0) ""Weight decay
                                  // rate.").SetDefault(1.0f);
    })
    .SetKernelFn(PD_KERNEL(AdamForward))
    .SetInferShapeFn(PD_INFER_SHAPE(AdamInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AdamInferDtype));
