// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

void StepPaddle(const paddle::Tensor& stop_flags,
                const paddle::Tensor& seq_lens_this_time,
                const paddle::Tensor& ori_seq_lens_encoder,
                const paddle::Tensor& seq_lens_encoder,
                const paddle::Tensor& seq_lens_decoder,
                const paddle::Tensor& block_tables,  // [bsz, block_num_per_seq]
                const paddle::Tensor& encoder_block_lens,
                const paddle::Tensor& is_block_step,
                const paddle::Tensor& step_block_list,
                const paddle::Tensor& step_lens,
                const paddle::Tensor& recover_block_list,
                const paddle::Tensor& recover_lens,
                const paddle::Tensor& need_block_list,
                const paddle::Tensor& need_block_len,
                const paddle::Tensor& used_list_len,
                const paddle::Tensor& free_list,
                const paddle::Tensor& free_list_len,
                const paddle::Tensor& input_ids,
                const paddle::Tensor& pre_ids,
                const paddle::Tensor& step_idx,
                const paddle::Tensor& next_tokens,
                const paddle::Tensor& first_token_ids,
                const int block_size,
                const int encoder_decoder_block_num) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const int bsz = seq_lens_this_time.shape()[0];
  PADDLE_ENFORCE_LE(
      bsz,
      640,
      phi::errors::InvalidArgument(
          "Only support bsz <= 640, but received bsz is %d", bsz));
  const int block_num_per_seq = block_tables.shape()[1];
  const int length = input_ids.shape()[1];
  const int pre_id_length = pre_ids.shape()[1];
  const int max_decoder_block_num = pre_id_length / block_size;
  int r = baidu::xpu::api::plugin::free_and_dispatch_block(
      xpu_ctx->x_context(),
      const_cast<bool*>(stop_flags.data<bool>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int*>(block_tables.data<int>()),
      const_cast<int*>(encoder_block_lens.data<int>()),
      const_cast<bool*>(is_block_step.data<bool>()),
      const_cast<int*>(step_block_list.data<int>()),
      const_cast<int*>(step_lens.data<int>()),
      const_cast<int*>(recover_block_list.data<int>()),
      const_cast<int*>(recover_lens.data<int>()),
      const_cast<int*>(need_block_list.data<int>()),
      const_cast<int*>(need_block_len.data<int>()),
      const_cast<int*>(used_list_len.data<int>()),
      const_cast<int*>(free_list.data<int>()),
      const_cast<int*>(free_list_len.data<int>()),
      const_cast<int64_t*>(first_token_ids.data<int64_t>()),
      bsz,
      block_size,
      block_num_per_seq,
      max_decoder_block_num);
  PD_CHECK(r == 0, "free_and_dispatch_block failed.");
  auto recover_lens_cpu = recover_lens.copy_to(paddle::CPUPlace(), false);
  int recover_lens_cpu_data = recover_lens_cpu.data<int>()[0];
  if (recover_lens_cpu_data > 0) {
    r = baidu::xpu::api::plugin::recover_block(
        xpu_ctx->x_context(),
        const_cast<int*>(recover_block_list.data<int>()),
        const_cast<int*>(recover_lens.data<int>()),
        const_cast<bool*>(stop_flags.data<bool>()),
        const_cast<int*>(seq_lens_this_time.data<int>()),
        ori_seq_lens_encoder.data<int>(),
        const_cast<int*>(seq_lens_encoder.data<int>()),
        seq_lens_decoder.data<int>(),
        const_cast<int*>(block_tables.data<int>()),
        const_cast<int*>(free_list.data<int>()),
        const_cast<int*>(free_list_len.data<int>()),
        const_cast<int64_t*>(input_ids.data<int64_t>()),
        pre_ids.data<int64_t>(),
        step_idx.data<int64_t>(),
        encoder_block_lens.data<int>(),
        used_list_len.data<int>(),
        next_tokens.data<int64_t>(),
        first_token_ids.data<int64_t>(),
        bsz,
        block_num_per_seq,
        length,
        pre_id_length);
    PD_CHECK(r == 0, "recover_block failed.");
  }
}

PD_BUILD_OP(step_paddle)
    .Inputs({"stop_flags",
             "seq_lens_this_time",
             "ori_seq_lens_encoder",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables",
             "encoder_block_lens",
             "is_block_step",
             "step_block_list",
             "step_lens",
             "recover_block_list",
             "recover_lens",
             "need_block_list",
             "need_block_len",
             "used_list_len",
             "free_list",
             "free_list_len",
             "input_ids",
             "pre_ids",
             "step_idx",
             "next_tokens",
             "first_token_ids"})
    .Attrs({"block_size: int", "encoder_decoder_block_num: int"})
    .Outputs({"stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "block_tables_out",
              "encoder_block_lens_out",
              "is_block_step_out",
              "step_block_list_out",
              "step_lens_out",
              "recover_block_list_out",
              "recover_lens_out",
              "need_block_list_out",
              "need_block_len_out",
              "used_list_len_out",
              "free_list_out",
              "free_list_len_out",
              "input_ids_out",
              "first_token_ids_out"})
    .SetInplaceMap({{"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"block_tables", "block_tables_out"},
                    {"encoder_block_lens", "encoder_block_lens_out"},
                    {"is_block_step", "is_block_step_out"},
                    {"step_block_list", "step_block_list_out"},
                    {"step_lens", "step_lens_out"},
                    {"recover_block_list", "recover_block_list_out"},
                    {"recover_lens", "recover_lens_out"},
                    {"need_block_list", "need_block_list_out"},
                    {"need_block_len", "need_block_len_out"},
                    {"used_list_len", "used_list_len_out"},
                    {"free_list", "free_list_out"},
                    {"free_list_len", "free_list_len_out"},
                    {"input_ids", "input_ids_out"},
                    {"first_token_ids", "first_token_ids_out"}})
    .SetKernelFn(PD_KERNEL(StepPaddle));
