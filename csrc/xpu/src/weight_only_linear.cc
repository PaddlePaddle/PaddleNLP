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
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"
#include <xft/xdnn_plugin.h>
#include "xblas_legacy_api.h"

namespace xftkernel = baidu::xpu::xftkernel;
namespace api = baidu::xpu::api;
// namespace xblas = baidu::xpu::xblas;

std::vector<paddle::Tensor> WeightOnlyLinearKernel(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::optional<paddle::Tensor>& weight_scale,
    const paddle::optional<paddle::Tensor>& bias,
    const std::string& weight_dtype="int8",
    const int32_t arch=0,
    const int32_t group_size=64
    ) {

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  using XPUType = typename XPUTypeTrait<bfloat16>::Type;
  typedef paddle::bfloat16 data_t;

  int64_t n = weight.dims()[0];
  int64_t k = weight.dims()[1];
  int64_t m = x.numel() / k;
  if(weight_dtype == "int4") {
    n = n * 2;
  }

  paddle::Tensor bias_fp32;
  if (bias.is_initialized() && bias.get().dtype() == phi::DataType::FLOAT16) {
    bias_fp32 = paddle::full(bias->shape(), -1, paddle::DataType::FLOAT32, bias->place());
    int r = baidu::xpu::api::cast<XPUType, float>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType*>(
            bias.get().data<phi::dtype::float16>()),
        bias_fp32.data<float>(),
        n);
  }

  paddle::Tensor weight_scale_fp32;
  if (weight_scale.is_initialized() && weight_scale.get().dtype() == phi::DataType::FLOAT16) {
    weight_scale_fp32 = paddle::full(weight_scale->shape(), -1, paddle::DataType::FLOAT32, weight_scale->place());
    int r = baidu::xpu::api::cast<XPUType, float>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType*>(
            weight_scale.get().data<phi::dtype::float16>()),
        weight_scale_fp32.data<float>(),
        n);
  }

  auto output = paddle::full({m, n}, -1, x.type(), x.place());

  auto input_x = reinterpret_cast<const XPUType*>(x.data<data_t>());
  auto input_y = reinterpret_cast<XPUType*>(output.data<data_t>());

  baidu::xpu::xblas::FcFusionTensor<const XPUType>
          tensor_x{input_x, nullptr, m, k, k, false};
  baidu::xpu::xblas::FcFusionTensor<const XPUType> tensor_y_const{
      input_y, nullptr, m, n, n, false};
  baidu::xpu::xblas::FcFusionTensor<XPUType> tensor_y{
      input_y, nullptr, m, n, n, false};

  baidu::xpu::xblas::FcFusionEpilogue<float, float> epilogue{
      api::Activation_t::LINEAR,
      bias ? bias_fp32.data<float>() : nullptr,
      nullptr,
      weight_scale ? weight_scale_fp32.data<float>() : nullptr,  // weight_scale_fp32.data<float>(), 
      0,
      1,
      nullptr};
  if(weight_dtype == "int4") {
    baidu::xpu::xblas::FcFusionDesc<int4_wo_int15, float, XPUType> desc{1.0f, 0.0f};
    baidu::xpu::xblas::FcFusionTensor<const int4_t> tensor_w{
        reinterpret_cast<const int4_t*>(weight.data<int8_t>()), nullptr, n, k, k, true};
    int r1 =  baidu::xpu::xblas::fc_fusion<XPUType,
                                        int4_t,
                                        XPUType,
                                        XPUType,
                                        int4_wo_int15, // int8_wo_t
                                        float,
                                        XPUType,
                                        float,
                                        float>(xpu_ctx->x_context(),
                                                tensor_x,
                                                tensor_w,
                                                tensor_y_const,
                                                tensor_y,
                                                desc,
                                                epilogue);
    PD_CHECK(r1 == 0, "xblas::fc_fusion failed");
  } else {
    baidu::xpu::xblas::FcFusionDesc<float, float, float> desc{1.0f, 0.0f};
    baidu::xpu::xblas::FcFusionTensor<const XPUType> tensor_w{
        // reinterpret_cast<const XPUType*>(weight.data<phi::dtype::bfloat16>()), nullptr, k, n, n, true};
        reinterpret_cast<const XPUType*>(weight.data<phi::dtype::bfloat16>()), nullptr, n, k, k, true};
    int r1 =  baidu::xpu::xblas::fc_fusion<XPUType,
                                        bfloat16,
                                        XPUType,
                                        XPUType,
                                        float, 
                                        float,
                                        float,
                                        float,
                                        float,
                                        XPUType>(xpu_ctx->x_context(),
                                                tensor_x,
                                                tensor_w,
                                                tensor_y_const,
                                                tensor_y,
                                                desc,
                                                epilogue);
    PD_CHECK(r1 == 0, "xblas::fc_fusion failed");
  }
  return {output};
}

PD_BUILD_OP(weight_only_linear_kernel)
    .Inputs({"x", "weight", paddle::Optional("weight_scale"), paddle::Optional("bias")})
    .Outputs({"output"})
    .Attrs({"weight_dtype: std::string", "arch: int", "group_size: int"})
    .SetKernelFn(PD_KERNEL(WeightOnlyLinearKernel));
