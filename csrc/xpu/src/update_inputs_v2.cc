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



void UpdateInputesV2(const paddle::Tensor& stop_flags,
               const paddle::Tensor& step_idx,
               const paddle::Tensor& not_need_stop, // cpu
               const paddle::Tensor& seq_lens_this_time,
               const paddle::Tensor& seq_lens_encoder,
               const paddle::Tensor& seq_lens_decoder,
               const paddle::Tensor& max_dec_len,
               const paddle::Tensor& input_ids,
               const paddle::Tensor& stop_nums,
               const paddle::Tensor& next_tokens,
               const paddle::Tensor& is_block_step,
               const paddle::Tensor& end_ids,
               const paddle::Tensor& kwargs_next_tokens) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  auto not_need_stop_xpu = not_need_stop.copy_to(stop_flags.place(), false);

  const int max_bsz = stop_flags.shape()[0];
  PADDLE_ENFORCE_LE(
      max_bsz,
      1024,
      phi::errors::InvalidArgument(
          "Only support max_bs <= 1024, but received max_bs is %d", max_bsz));
  const int now_bsz = seq_lens_this_time.shape()[0];
  const int input_ids_stride = input_ids.shape()[1];
  const int end_length = end_ids.shape()[0];
  int r = baidu::xpu::api::plugin::update_inputs_v2(
      xpu_ctx->x_context(),
      const_cast<bool*>(not_need_stop_xpu.data<bool>()),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      const_cast<bool*>(stop_flags.data<bool>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_encoder.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int64_t*>(next_tokens.data<int64_t>()),
      const_cast<int64_t*>(kwargs_next_tokens.data<int64_t>()),
      const_cast<int64_t*>(input_ids.data<int64_t>()),
      end_ids.data<int64_t>(),
      stop_nums.data<int64_t>(),
      is_block_step.data<bool>(),
      max_dec_len.data<int64_t>(),
      now_bsz,
      max_bsz,
      input_ids_stride,
      end_length);
  PD_CHECK(r == 0, "baidu::xpu::api::plugin::update_inputs failed.");

  auto not_need_stop_cpu = not_need_stop_xpu.copy_to(not_need_stop.place(), false);
  bool *not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}
PD_BUILD_OP(update_inputs_v2)
    .Inputs({"stop_flags", 
             "step_idx",
             "not_need_stop", 
             "seq_lens_this_time", 
             "seq_lens_encoder", 
             "seq_lens_decoder",
             "max_dec_len",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step",
             "end_ids",
             "kwargs_next_tokens"})
    .Outputs({"stop_flags_out", 
        "not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "input_ids_out",
              "next_tokens_out",
              "kwargs_next_tokens_out",
              "step_idx_out"})
    .SetInplaceMap({{"stop_flags", "stop_flags_out"},
        {"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"input_ids", "input_ids_out"},
                    {"next_tokens", "next_tokens_out"},
                    {"kwargs_next_tokens", "kwargs_next_tokens_out"},
                    {"step_idx", "step_idx_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputesV2));
