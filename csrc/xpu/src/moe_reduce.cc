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
namespace xftkernel = baidu::xpu::xftkernel;
std::vector<paddle::Tensor> MoeReduceKernel(
    const paddle::Tensor& ffn_out,
    const paddle::Tensor& expert_scales_float,
    const paddle::Tensor& permute_indices_per_token,
    const paddle::Tensor& top_k_indices,
    const paddle::Tensor& ffn2_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor) {

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  using XPUType = typename XPUTypeTrait<bfloat16>::Type;
  typedef paddle::bfloat16 data_t;

  const int topk = top_k_indices.shape()[1];
  const int num_rows = ffn_out.shape()[0] / topk;
  const int hidden_size = ffn_out.shape()[1];

  auto output = paddle::full({num_rows, hidden_size}, -1, ffn_out.type(), ffn_out.place());

 int ret = xftkernel::xft_moe_ffn_post_fusion<XPUType, int32_t>(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPUType*>(ffn_out.data<data_t>()),
            top_k_indices.data<int32_t>(),
            reinterpret_cast<const XPUType*>(expert_scales_float.data<data_t>()),
            reinterpret_cast<XPUType*>(output.data<data_t>()),
            num_rows,
            hidden_size,
            -1,
            topk);
    PD_CHECK(ret == 0, "xftkernel::xft_moe_ffn_post_fusion failed");

  return {output};
}

PD_BUILD_OP(mod_reduce_xpu)
    .Inputs({"ffn_out",
             "expert_scales_float",
             "permute_indices_per_token",
             "top_k_indices",
             "ffn2_bias"})
    .Outputs({"output"})
    .Attrs({"norm_topk_prob: bool", "routed_scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(MoeReduceKernel));
