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

#include <core/ctx_manager.h>
#include <core/xft_check.h>
#include <core/xft_event.h>
#include <core/xft_params.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <xft/operation/xft_fc_helper.h>
#include <xft/xdnn_plugin.h>

#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"

namespace xftkernel = baidu::xpu::xftkernel;

std::vector<paddle::Tensor> MoeFFNKernel(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& token_nums_per_expert,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const std::string& quant_method) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  using XPUType = typename XPUTypeTrait<bfloat16>::Type;
  typedef paddle::bfloat16 data_t;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  auto ffn_out_paddle = permute_input.copy_to(permute_input.place(), false);


  const int64_t expanded_active_expert_rows = permute_input.shape()[0];
  const int num_experts = ffn1_weight.shape()[0];
  const int hidden_size = ffn1_weight.shape()[2];
  int64_t inter_size = ffn1_weight.shape()[1];
  if (quant_method == "weight_only_int4") {
    inter_size = inter_size * 2;
  }


  auto ffn1_input = baidu::xpu::xft::xftTensor<XPUType, 2>(
      reinterpret_cast<XPUType*>(
          const_cast<paddle::bfloat16*>(permute_input.data<data_t>())),
      std::array<int64_t, 2>{permute_input.shape()[0],
                             permute_input.shape()[1]});

  auto ffn1_bias_tenosor = baidu::xpu::xft::xftVec<float>(
      ffn1_bias ? const_cast<float*>(ffn1_bias->data<float>()) : nullptr,
      std::array<int64_t, 1>{inter_size});

  auto ffn1_out_paddle = paddle::full({expanded_active_expert_rows, inter_size},
                                      -1,
                                      permute_input.type(),
                                      permute_input.place());
  auto ffn1_out_tensor = baidu::xpu::xft::xftTensor<XPUType, 2>(
      reinterpret_cast<XPUType*>(ffn1_out_paddle.data<data_t>()),
      std::array<int64_t, 2>{ffn1_out_paddle.shape()[0],
                             ffn1_out_paddle.shape()[1]});

  auto token_nums_per_expert_tensor = baidu::xpu::xft::xftVec<int32_t>(
      const_cast<int32_t*>(token_nums_per_expert.data<int>()),
      std::array<int64_t, 1>{num_experts});

  auto empty_tensor = baidu::xpu::xft::xftTensor<int32_t, 2>();

  auto sorted_tokens_idx_paddle = paddle::full({expanded_active_expert_rows, 1},
                                               -1,
                                               paddle::DataType::INT32,
                                               permute_input.place());

  auto sorted_tokens_idx_tensor = baidu::xpu::xft::xftTensor<int32_t, 2>(
      sorted_tokens_idx_paddle.data<int>(),
      std::array<int64_t, 2>{sorted_tokens_idx_paddle.shape()[0],
                             sorted_tokens_idx_paddle.shape()[1]});
  int ret = 0;
  if (quant_method == "weight_only_int4") {
    auto ffn1_w_tensor = baidu::xpu::xft::xftMat<int4_t>(
        reinterpret_cast<int4_t*>(
            const_cast<int8_t*>(ffn1_weight.data<int8_t>())),
        nullptr,
        const_cast<float*>(
            ffn1_scale->data<
                float>()),  // 这个地方GPU是float16，我们需要的是float，需要check
        std::array<int64_t, 2>{num_experts * inter_size, hidden_size});

    ret = baidu::xpu::xft::xft_moe_sort_fc_block<XPUType,
                                                 int4_t,
                                                 XPUType,
                                                 float,
                                                 int32_t,
                                                 int4_wo_int15>(
        xpu_ctx->x_context(),
        ffn1_input,
        ffn1_w_tensor,
        ffn1_out_tensor,
        &ffn1_bias_tenosor,
        token_nums_per_expert_tensor,
        sorted_tokens_idx_tensor,
        nullptr,
        empty_tensor,
        baidu::xpu::api::Activation_t::LINEAR,
        false,
        true,
        1.0,
        0.0,
        num_experts,
        1,  // topk
        0,
        2,
        1);
  } else {
    auto ffn1_w_tensor = baidu::xpu::xft::xftMat<int8_t>(
        const_cast<int8_t*>(ffn1_weight.data<int8_t>()),
        nullptr,
        const_cast<float*>(
            ffn1_scale->data<
                float>()),  // 这个地方GPU是float16，我们需要的是float，需要check
        std::array<int64_t, 2>{num_experts * inter_size, hidden_size});
    ret = baidu::xpu::xft::xft_moe_sort_fc_block<XPUType,
                                                 int8_t,
                                                 XPUType,
                                                 float,
                                                 int32_t,
                                                 float>(
        xpu_ctx->x_context(),
        ffn1_input,
        ffn1_w_tensor,
        ffn1_out_tensor,
        &ffn1_bias_tenosor,
        token_nums_per_expert_tensor,
        sorted_tokens_idx_tensor,
        nullptr,
        empty_tensor,
        baidu::xpu::api::Activation_t::LINEAR,
        false,
        true,
        1.0,
        0.0,
        num_experts,
        1,  // topk
        0,
        2,
        1);
  }

  auto ffn2_input_paddle =
      paddle::full({expanded_active_expert_rows, inter_size / 2},
                   -1,
                   permute_input.type(),
                   permute_input.place());


  auto ffn2_input_tensor = baidu::xpu::xft::xftTensor<XPUType, 2>(
      reinterpret_cast<XPUType*>(ffn2_input_paddle.data<data_t>()),
      std::array<int64_t, 2>{ffn2_input_paddle.shape()[0],
                             ffn2_input_paddle.shape()[1]});
  ret = xftkernel::xft_fast_swiglu_add_mul_fusion<XPUType>(
      xpu_ctx->x_context(),
      ffn1_out_tensor.data(),
      ffn2_input_tensor.data(),
      expanded_active_expert_rows,
      inter_size,
      nullptr,
      nullptr,
      true,
      nullptr,
      nullptr);


  //   auto ffn2_bias_tenosor = baidu::xpu::xft::xftVec<float>(
  //         ffn1_bias ? const_cast<float*>(ffn2_bias->data<float>()):
  //         nullptr, std::array<int64_t, 1>{hidden_size});
  auto ffn2_out_tensor = baidu::xpu::xft::xftTensor<XPUType, 2>(
      reinterpret_cast<XPUType*>(ffn_out_paddle.data<data_t>()),
      std::array<int64_t, 2>{ffn_out_paddle.shape()[0],
                             ffn_out_paddle.shape()[1]});
  if (quant_method == "weight_only_int4") {
    auto ffn2_w_tensor = baidu::xpu::xft::xftMat<int4_t>(
        reinterpret_cast<int4_t*>(
            const_cast<int8_t*>(ffn2_weight.data<int8_t>())),
        nullptr,
        const_cast<float*>(
            ffn2_scale->data<
                float>()),  // 这个地方GPU是float16，我们需要的是float，需要check
        std::array<int64_t, 2>{num_experts * hidden_size, inter_size / 2});
    ret = baidu::xpu::xft::xft_moe_sort_fc_block<XPUType,
                                                 int4_t,
                                                 XPUType,
                                                 float,
                                                 int32_t,
                                                 int4_wo_int15>(
        xpu_ctx->x_context(),
        ffn2_input_tensor,
        ffn2_w_tensor,
        ffn2_out_tensor,
        nullptr,
        token_nums_per_expert_tensor,
        sorted_tokens_idx_tensor,
        nullptr,
        empty_tensor,
        baidu::xpu::api::Activation_t::LINEAR,
        false,
        true,
        1.0,
        0.0,
        num_experts,
        1,  // topk
        0,
        2,
        1);
  } else {
    auto ffn2_w_tensor = baidu::xpu::xft::xftMat<int8_t>(
        const_cast<int8_t*>(ffn2_weight.data<int8_t>()),
        nullptr,
        const_cast<float*>(
            ffn2_scale->data<
                float>()),  // 这个地方GPU是float16，我们需要的是float，需要check
        std::array<int64_t, 2>{num_experts * hidden_size, inter_size / 2});
    ret = baidu::xpu::xft::xft_moe_sort_fc_block<XPUType,
                                                 int8_t,
                                                 XPUType,
                                                 float,
                                                 int32_t,
                                                 float>(
        xpu_ctx->x_context(),
        ffn2_input_tensor,
        ffn2_w_tensor,
        ffn2_out_tensor,
        nullptr,
        token_nums_per_expert_tensor,
        sorted_tokens_idx_tensor,
        nullptr,
        empty_tensor,
        baidu::xpu::api::Activation_t::LINEAR,
        false,
        true,
        1.0,
        0.0,
        num_experts,
        1,  // topk
        0,
        2,
        1);
  }


  return {
      ffn_out_paddle,
  };
}


PD_BUILD_OP(moe_ffn_xpu)
    .Inputs({"permute_input",
             "token_nums_per_expert",
             "ffn1_weight",
             "ffn2_weight",
             "ffn1_bias",
             "ffn1_scale",
             "ffn2_scale"})
    .Outputs({"ffn_out"})
    .Attrs({"quant_method: std::string"})
    .SetKernelFn(PD_KERNEL(MoeFFNKernel));
