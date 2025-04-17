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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"
#include <xft/xdnn_plugin.h>
#include "cublasLt.h"
#include "ops.h"

namespace xftkernel = baidu::xpu::xftkernel;
std::vector<paddle::Tensor> Bmm(const paddle::Tensor& input, 
                                             const paddle::Tensor& weight) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xpu::ctx_guard RAII_GUARD(xpu_ctx->x_context());
  using XPUType = typename XPUTypeTrait<bfloat16>::Type; // only support bfloat16
  typedef paddle::bfloat16 data_t;
  const int batch = input.dims()[0];
  const int m = input.dims()[1];
  const int n = weight.dims()[2];
  const int k = weight.dims()[1];

  auto out = paddle::full({batch, m, n}, -2, input.type(), input.place());           
  int ret = baidu::xpu::xblas::fc_batched<XPUType, XPUType, XPUType, float, float, 0>(
          xpu_ctx->x_context(),
          batch,
          false,
          false,
          m,
          n,
          k,
          1.0f,
          reinterpret_cast<const XPUType*>(input.data<data_t>()),
          m * k,
          reinterpret_cast<const XPUType*>(weight.data<data_t>()),
          k * n,
          0.0f,
          reinterpret_cast<XPUType*>(out.data<data_t>()),
          m * n,
          nullptr,
          nullptr);
  return {out};
}

std::vector<std::vector<int64_t>> BmmInferShape(const std::vector<int64_t>& input_shape,
                                                             const std::vector<int64_t>& weight_shape) {
    int64_t batch = input_shape[0];
    int64_t m = input_shape[1];
    int64_t n = weight_shape[2];
    return {{batch, m, n}};
}

std::vector<paddle::DataType> BmmInferDtype(const paddle::DataType& input_dtype,
                                                         const paddle::DataType& weight_dtype) {
    return {input_dtype};
}

PD_BUILD_OP(batch_matmul_xpu)
    .Inputs({"imput", "weight"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(Bmm))
    .SetInferShapeFn(PD_INFER_SHAPE(BmmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BmmInferDtype));