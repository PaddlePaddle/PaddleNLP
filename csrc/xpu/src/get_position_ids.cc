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

void GetPositionIdsKernel(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& position_ids) {

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const int bs = seq_lens_encoder.shape()[0];

  int ret = baidu::xpu::api::plugin::get_position_ids(
        xpu_ctx->x_context(),
        seq_lens_encoder.data<int32_t>(),
        seq_lens_decoder.data<int32_t>(),
        seq_lens_this_time.data<int32_t>(),
        const_cast<int32_t*>(position_ids.data<int32_t>()),
        bs
  );
  PD_CHECK(ret == 0, "api::plugin::get_position_ids failed");
}

PD_BUILD_OP(get_position_ids)
    .Inputs({"seq_lens_encoder", "seq_lens_decoder", "seq_lens_this_time", "position_ids"})
    .Outputs({"position_ids_out"})
    .SetInplaceMap({{"position_ids", "position_ids_out"}})
    .SetKernelFn(PD_KERNEL(GetPositionIdsKernel));
