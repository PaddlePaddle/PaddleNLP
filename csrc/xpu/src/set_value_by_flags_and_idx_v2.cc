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

void SetValueByFlagsAndIdx(const paddle::Tensor& pre_ids_all,
                           const paddle::Tensor& input_ids,
                           const paddle::Tensor& seq_lens_this_time,
                           const paddle::Tensor& seq_lens_encoder,
                           const paddle::Tensor& seq_lens_decoder,
                           const paddle::Tensor& step_idx,
                           const paddle::Tensor& stop_flags) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  std::vector<int64_t> pre_ids_all_shape = pre_ids_all.shape();
  int bs = seq_lens_this_time.shape()[0];
  int length = pre_ids_all.shape()[1];
  int length_input_ids = input_ids.shape()[1];
  int r = baidu::xpu::api::plugin::set_value_by_flags_and_idx(
      xpu_ctx->x_context(),
      stop_flags.data<bool>(),
      const_cast<int64_t*>(pre_ids_all.data<int64_t>()),
      input_ids.data<int64_t>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      step_idx.data<int64_t>(),
      bs,
      length,
      length_input_ids);
  PD_CHECK(r == 0, "xpu::plugin::set_value_by_flags_and_idx failed.");
}

PD_BUILD_OP(set_value_by_flags_and_idx_v2)
    .Inputs({"pre_ids_all",
             "input_ids",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "stop_flags"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdx));
