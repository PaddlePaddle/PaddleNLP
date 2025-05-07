// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "ops.h"
#include <xft/xdnn_plugin.h>



void FusedRotaryPositionEncoding(
    paddle::Tensor& query,  // [num_tokens, num_heads, head_size] or
                            // [num_tokens, num_heads * head_size]
    paddle::Tensor& key,
    // [num_tokens, num_kv_heads, head_size] or [num_tokens, num_kv_heads *
    // head_size]
    const paddle::Tensor& position_ids,   // [num_tokens]
    const paddle::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    int head_size,
    bool is_neox) {

  baidu::xpu::api::plugin::print_times("[TIME BEGIN] FusedRotaryPositionEncoding" );

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xpu::ctx_guard RAII_GUARD(xpu_ctx->x_context());

  typedef paddle::bfloat16 data_t;
  using XPUType = typename XPUTypeTrait<data_t>::Type;

  int64_t num_tokens = query.dims()[0];
  int num_heads = query.numel() / num_tokens / head_size;
  int num_kv_heads = key.numel() / num_tokens / head_size;
  int rot_dim = cos_sin_cache.dims()[1];
  int64_t query_stride = num_heads * head_size;
  int64_t key_stride = num_kv_heads * head_size;

  baidu::xpu::api::plugin::rotary_embedding_neox<XPUType, XPUType>(xpu_ctx->x_context(),
  position_ids.data<int>(),
  reinterpret_cast<XPUType*>(query.data<data_t>()),
  reinterpret_cast<XPUType*>(key.data<data_t>()),
  reinterpret_cast<const XPUType*>(cos_sin_cache.data<data_t>()),
  rot_dim,
  num_heads,
  num_kv_heads,
  head_size,
  num_tokens);
  baidu::xpu::api::plugin::print_times("[TIME END] FusedRotaryPositionEncoding" );

}


PD_BUILD_OP(fused_rotary_position_encoding)
    .Inputs({"query", "key", "position_ids", "cos_sin_cache"})
    .Outputs({"query_out", "key_out"})
    .Attrs({"head_size: int", "is_neox: bool"})
    .SetInplaceMap({{"query", "query_out"}, {"key", "key_out"}})
    .SetKernelFn(PD_KERNEL(FusedRotaryPositionEncoding));
    