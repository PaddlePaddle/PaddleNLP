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
#include <core/xft_tensor.h>
#include <core/xft_check.h>
#include <core/xft_event.h>
#include <core/xft_params.h>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include <xft/operation/ffn_moe_vllm.h>
#include <xft/xdnn_plugin.h>

#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"

namespace xftkernel = baidu::xpu::xftkernel;
namespace xft = baidu::xpu::xft;

inline uint32_t get_xpu_moe_speed_layers() {
    static const char* xpu_moe_speed_layers_env = std::getenv("FLAGS_xpu_moe_speed_layers");
    static const uint32_t xpu_moe_speed_layers =
            xpu_moe_speed_layers_env == nullptr ? 0 : std::stoul(std::string(xpu_moe_speed_layers_env));
    return xpu_moe_speed_layers;
}

std::vector<paddle::Tensor> MoeFusedKernel(
    const paddle::Tensor& input, //[980. 7168]
    const paddle::Tensor& gate_weight,//[7168. 256] ->[256,7168]
    const paddle::Tensor& ffn_inter_weights,//[256, 7168,4096] reshape
    const paddle::Tensor& ffn_output_weights,//[256, 2048, 7168] reshape
    const paddle::Tensor& ffn1_weights_scale,// [256, 4096]
    const paddle::Tensor& ffn2_weights_scale,// [256, 7168]
    const paddle::Tensor& score_bias, // 256
    int moe_top_k,
    int expert_group_num,
    int moe_topk_group,
    int layer) {

  baidu::xpu::api::plugin::print_times("[TIME BEGIN] MoeFusedKernel");

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  using XPUType = typename XPUTypeTrait<bfloat16>::Type;
  typedef paddle::bfloat16 data_t;

  const int64_t m = input.shape()[0];
  const int64_t hidden_size = input.shape()[1];
  const int64_t expert_num = gate_weight.shape()[0];
  const int64_t ffn_inter_hidden_size = ffn_inter_weights.shape()[2];
  const int64_t ffn_output_hidden_size = ffn_output_weights.shape()[2];

  xft::Tensor input_tensor(
      reinterpret_cast<XPUType*>(const_cast<data_t*>(input.data<data_t>())),
      xft::DataType::DT_BFLOAT16,
      input.shape());
  xft::Tensor gate_weight_tensor(
      const_cast<float*>(gate_weight.data<float>()),
      xft::DataType::DT_FLOAT,
      gate_weight.shape());
  xft::Tensor ffn_inter_weights_tensor(
      const_cast<int8_t*>(ffn_inter_weights.data<int8_t>()),
      nullptr,
      const_cast<float*>(ffn1_weights_scale.data<float>()),
      xft::DataType::DT_INT8,
      {expert_num, ffn_inter_hidden_size, hidden_size});
  xft::Tensor ffn_output_weights_tensor(
      const_cast<int8_t*>(ffn_output_weights.data<int8_t>()),
      nullptr,
      const_cast<float*>(ffn2_weights_scale.data<float>()),
      xft::DataType::DT_INT8,
      {expert_num, ffn_output_hidden_size, ffn_inter_hidden_size / 2});
  xft::Tensor score_bias_tensor(const_cast<float*>(score_bias.data<float>()),
                                xft::DataType::DT_FLOAT,
                                score_bias.shape());

  auto output = paddle::full({m, hidden_size}, -1, input.type(), input.place());
  xft::Tensor output_tensor(reinterpret_cast<XPUType*>(output.data<data_t>()),
                            xft::DataType::DT_BFLOAT16,
                            output.shape());

  xft::MoeFFNWeight moe_weight{&gate_weight_tensor,
                               &ffn_inter_weights_tensor,
                               nullptr,
                               &ffn_output_weights_tensor,
                               nullptr,
                               nullptr,
                               nullptr,
                               &score_bias_tensor};
  xft::MoeFFNParam moe_param{
    expert_num,
    moe_top_k,
    false,
    true,
    expert_group_num,
    moe_topk_group,
    true,
    false,
    1,
    0,
    "sigmoid"
  };

    if (layer >= get_xpu_moe_speed_layers()) {
        xft::xft_moe_ffn_block_sorted<XPUType, int8_t, XPUType, float>(
            xpu_ctx->x_context(), &input_tensor, &output_tensor, moe_weight, moe_param);
    } else {
        xft::xft_moe_ffn_block_sorted<XPUType, int8_t, XPUType, int8_wo_t>(
            xpu_ctx->x_context(), &input_tensor, &output_tensor, moe_weight, moe_param);
    }
  
  baidu::xpu::api::plugin::print_times("[TIME END] MoeFusedKernel");

  return {
      output,
  };
}


std::vector<std::vector<int64_t>> MoeFusedInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& gate_weight_shape,
    const std::vector<int64_t>& ffn_inter_weights_shape,
    const std::vector<int64_t>& ffn_output_weights_shape,
    const std::vector<int64_t>& ffn1_weights_scale_shape,
    const std::vector<int64_t>& ffn2_weights_scale_shape,
    const std::vector<int64_t>& score_bias_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> MoeFusedInferDtype(
    const paddle::DataType& input_type,
    const paddle::DataType& gate_weight_type,
    const paddle::DataType& ffn_inter_weights_type,
    const paddle::DataType& ffn_output_weights_type,
    const paddle::DataType& ffn1_weights_scale_type,
    const paddle::DataType& ffn2_weights_scale_type,
    const paddle::DataType& score_bias_type,
    const int moe_top_k,
    const int expert_group_num,
    const int moe_topk_group,
    const int layer) {
        return {input_type};
}

PD_BUILD_OP(moe_fused_xpu)
    .Inputs({
        "input",
        "gate_weight",
        "ffn_inter_weights",
        "ffn_outer_weights",
        "ffn1_weights_scale",
        "ffn2_weights_scale",
        "score_bias",
    })
    .Outputs({"output"})
    .Attrs({"moe_top_k: int", "expert_group_num: int", "moe_topk_group: int", "layer: int"})
    .SetKernelFn(PD_KERNEL(MoeFusedKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeFusedInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeFusedInferDtype));
