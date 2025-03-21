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
std::vector<paddle::Tensor> MoeDispatchKernel(
        const paddle::Tensor& X,
        const paddle::Tensor& gating_output,
        const int moe_topk,
        const bool group_moe,
        const bool topk_only_mode) {
    phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
    auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
    auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
    int token_rows = 0;
    auto input_dims = X.shape();
    if (input_dims.size() == 3) {
        token_rows = input_dims[0] * input_dims[1];
    } else {
        token_rows = input_dims[0];
    }
    const int num_rows = token_rows;
    const int hidden_size = X.shape()[input_dims.size() - 1];
    auto gating_dims = gating_output.shape();
    const int expert_num = gating_dims[gating_dims.size() - 1];
    if (group_moe) {
        // Check if expert_num is divisible by moe_topk, else throw an error
        PADDLE_ENFORCE_EQ(
                expert_num % moe_topk,
                0,
                common::errors::InvalidArgument(
                        "The number of experts (expert_num) "
                        "must be divisible by moe_topk. "
                        "Got expert_num = %d and moe_topk = %d.",
                        expert_num,
                        moe_topk));
    }

    auto permute_input = paddle::full({num_rows * moe_topk, hidden_size}, -1, X.type(), X.place());
    auto token_nums_per_expert = paddle::full({expert_num}, 0, paddle::DataType::INT32, X.place());
    auto permute_indices_per_token = paddle::full({moe_topk, num_rows}, -1, paddle::DataType::INT32, X.place());
    auto expert_scales_float = paddle::full({num_rows, moe_topk}, 0.0f, paddle::DataType::FLOAT32, X.place());
    auto top_k_indices = paddle::full({num_rows, moe_topk}, -1, paddle::DataType::INT32, X.place());

    using XPUType = typename XPUTypeTrait<bfloat16>::Type;
    typedef paddle::bfloat16 data_t;
    auto index_data = permute_indices_per_token.data<int32_t>();
//     auto index = paddle::full({num_rows, moe_topk}, -1, paddle::DataType::INT32, X.place());

    int ret = xftkernel::xft_moe_group_topk_fusion<float, float, int32_t>(
            xpu_ctx->x_context(),
            gating_output.data<float>(),
            expert_scales_float.data<float>(),
            index_data,
            nullptr,
            num_rows,
            expert_num,
            0,
            0,
            moe_topk,  // num of shared expert
            0);
    PD_CHECK(ret == 0, "xftkernel::xft_moe_softmax_topk_norm_fusion failed");
    auto sorted_tokens_num_lod = paddle::full({expert_num + 1}, -1, paddle::DataType::INT32, X.place());
     ret = xftkernel::xft_moe_ffn_pre_sorted<XPUType, int32_t>(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPUType*>(X.data<data_t>()),
            index_data,
            nullptr,
            reinterpret_cast<XPUType*>(permute_input.data<data_t>()),
            top_k_indices.data<int32_t>(),
            token_nums_per_expert.data<int32_t>(),
            sorted_tokens_num_lod.data<int32_t>(),
            token_rows,
            hidden_size,
            expert_num,
            moe_topk,
            0);
    PD_CHECK(ret == 0, "xftkernel::xft_moe_ffn_pre_sorted failed");
    return {permute_input,
            sorted_tokens_num_lod,
            permute_indices_per_token,
            expert_scales_float,
            top_k_indices};
}

PD_BUILD_OP(moe_dispatch_xpu)
        .Inputs({"X", "gating_output"})
        .Outputs(
                {"permute_input",
                 "token_nums_per_expert",
                 "permute_indices_per_token",
                 "expert_scales_float",
                 "expert_scales_int"})
        .Attrs({"moe_topk: int", "group_moe: bool", "topk_only_mode: bool"})
        .SetKernelFn(PD_KERNEL(MoeDispatchKernel));
