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

void TokenPenaltyMultiScores(const paddle::Tensor& pre_ids,
                             const paddle::Tensor& logits,
                             const paddle::Tensor& penalty_scores,
                             const paddle::Tensor& frequency_scores,
                             const paddle::Tensor& presence_scores,
                             const paddle::Tensor& temperatures,
                             const paddle::Tensor& bad_tokens,
                             const paddle::Tensor& cur_len,
                             const paddle::Tensor& min_len,
                             const paddle::Tensor& eos_token_id) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  int64_t bs = logits.shape()[0];
  PADDLE_ENFORCE_LE(
      bs,
      640,
      phi::errors::InvalidArgument(
          "Only support bsz <= 1024, but received bsz is %d", bs));
  int64_t length = logits.shape()[1];
  int64_t length_id = pre_ids.shape()[1];
  int64_t length_bad_words = bad_tokens.shape()[0];
  int64_t end_length = eos_token_id.shape()[0];
  switch (logits.type()) {
    case paddle::DataType::FLOAT16: {
      using XPUType = typename XPUTypeTrait<float16>::Type;
      typedef paddle::float16 data_t;
      int r = baidu::xpu::api::plugin::token_penalty_multi_scores(
          xpu_ctx->x_context(),
          pre_ids.data<int64_t>(),
          reinterpret_cast<XPUType*>(
              const_cast<data_t*>(logits.data<data_t>())),
          reinterpret_cast<const XPUType*>(penalty_scores.data<data_t>()),
          reinterpret_cast<const XPUType*>(frequency_scores.data<data_t>()),
          reinterpret_cast<const XPUType*>(presence_scores.data<data_t>()),
          temperatures.data<float>(),
          cur_len.data<int64_t>(),
          min_len.data<int64_t>(),
          eos_token_id.data<int64_t>(),
          bad_tokens.data<int64_t>(),
          bs,
          length,
          length_id,
          end_length,
          length_bad_words);
      PD_CHECK(r == 0, "xpu::plugin::token_penalty_multi_scores failed.");
    } break;
    case paddle::DataType::FLOAT32: {
      int r = baidu::xpu::api::plugin::token_penalty_multi_scores(
          xpu_ctx->x_context(),
          pre_ids.data<int64_t>(),
          const_cast<float*>(logits.data<float>()),
          penalty_scores.data<float>(),
          frequency_scores.data<float>(),
          presence_scores.data<float>(),
          temperatures.data<float>(),
          cur_len.data<int64_t>(),
          min_len.data<int64_t>(),
          eos_token_id.data<int64_t>(),
          bad_tokens.data<int64_t>(),
          bs,
          length,
          length_id,
          end_length,
          length_bad_words);
      PD_CHECK(r == 0, "xpu::plugin::token_penalty_multi_scores failed.");
    } break;
    default:
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and float32 are supported. ");
      break;
  }
}

PD_BUILD_OP(get_token_penalty_multi_scores_v2)
    .Inputs({"pre_ids",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "temperatures",
             "bad_tokens",
             "cur_len",
             "min_len",
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetInplaceMap({{"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores));
