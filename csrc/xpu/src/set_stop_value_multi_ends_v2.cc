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

#include <fcntl.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include "paddle/extension.h"
#include "xpu/plugin.h"

void GetStopFlagsMulti(const paddle::Tensor &topk_ids,
                       const paddle::Tensor &stop_flags,
                       const paddle::Tensor &seq_lens,
                       const paddle::Tensor &end_ids,
                       const paddle::Tensor &next_tokens) {
  PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  std::vector<int64_t> shape = topk_ids.shape();
  int64_t bs_now = shape[0];
  int64_t end_length = end_ids.shape()[0];
  bool beam_search = false;
  int r = baidu::xpu::api::plugin::set_stop_value_multi_ends<int64_t>(
      xpu_ctx->x_context(),
      const_cast<bool *>(stop_flags.data<bool>()),
      const_cast<int64_t *>(topk_ids.data<int64_t>()),
      const_cast<int64_t *>(next_tokens.data<int64_t>()),
      end_ids.data<int64_t>(),
      seq_lens.data<int>(),
      bs_now,
      end_length,
      beam_search);
  PD_CHECK(r == 0, "xpu::plugin::set_stop_value_multi_ends failed.");
}

PD_BUILD_OP(set_stop_value_multi_ends_v2)
    .Inputs({"topk_ids", "stop_flags", "seq_lens", "end_ids", "next_tokens"})
    .Outputs({"topk_ids_out", "stop_flags_out", "next_tokens_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti));
