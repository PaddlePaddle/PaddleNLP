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

void UpdateInputes(const paddle::Tensor& stop_flags,
                   const paddle::Tensor& not_need_stop,  // xpu
                   const paddle::Tensor& seq_lens_this_time,
                   const paddle::Tensor& seq_lens_encoder,
                   const paddle::Tensor& seq_lens_decoder,
                   const paddle::Tensor& input_ids,
                   const paddle::Tensor& stop_nums,
                   const paddle::Tensor& next_tokens,
                   const paddle::Tensor& is_block_step) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const int max_bsz = stop_flags.shape()[0];
  PADDLE_ENFORCE_LE(
      max_bsz,
      1024,
      phi::errors::InvalidArgument(
          "Only support max_bs <= 1024, but received max_bs is %d", max_bsz));
  const int now_bsz = seq_lens_this_time.shape()[0];
  const int input_ids_stride = input_ids.shape()[1];
  int r = baidu::xpu::api::plugin::update_inputs(
      xpu_ctx->x_context(),
      const_cast<bool*>(not_need_stop.data<bool>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_encoder.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int64_t*>(input_ids.data<int64_t>()),
      stop_nums.data<int64_t>(),
      stop_flags.data<bool>(),
      is_block_step.data<bool>(),
      next_tokens.data<int64_t>(),
      now_bsz,
      max_bsz,
      input_ids_stride);
  PD_CHECK(r == 0, "baidu::xpu::api::plugin::update_inputs failed.");
}

PD_BUILD_OP(update_inputs)
    .Inputs({"stop_flags",
             "not_need_stop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step"})
    .Outputs({"not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "input_ids_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"input_ids", "input_ids_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputes));
