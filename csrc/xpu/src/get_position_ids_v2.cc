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
#include "xblas_legacy_api.h"

namespace xftkernel = baidu::xpu::xftkernel;
namespace api = baidu::xpu::api;
// namespace xblas = baidu::xpu::xblas;

std::vector<paddle::Tensor> GetPositionIdsKernelV2(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time) {

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  const int bs = seq_lens_this_time.shape()[0];

  std::vector<int> seq_lens_this_time_cpu(bs, 0);
  int r = xpu_memcpy(seq_lens_this_time_cpu.data(),
                 seq_lens_this_time.data<int>(),
                 sizeof(int32_t) * bs,
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  int total_len = 0;
  for (int i = 0; i < bs; ++i) {
    total_len += seq_lens_this_time_cpu[i];
  }

  auto position_ids_out = paddle::full({total_len}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());

  int ret = baidu::xpu::api::plugin::get_position_ids(
        xpu_ctx->x_context(),
        seq_lens_encoder.data<int32_t>(),
        seq_lens_decoder.data<int32_t>(),
        seq_lens_this_time.data<int32_t>(),
        const_cast<int32_t*>(position_ids_out.data<int32_t>()),
        bs
  );
  PD_CHECK(ret == 0, "api::plugin::get_position_ids failed");
  return {position_ids_out};
}

std::vector<std::vector<int64_t>> GetPositionIdsV2InferShape(const std::vector<int64_t>& seq_lens_encoder,
                                                             const std::vector<int64_t>& seq_lens_decoder,
                                                             const std::vector<int64_t>& seq_lens_this_time) {
    return {{-1}};
}

std::vector<paddle::DataType> GetPositionIdsV2InferDtype(const paddle::DataType& seq_lens_encoder_dtype,
                                                         const paddle::DataType& seq_lens_decoder_dtype,
                                                         const paddle::DataType& seq_lens_this_time_dtype) {
    return {seq_lens_encoder_dtype};
}

PD_BUILD_OP(get_position_ids_v2)
    .Inputs({"seq_lens_encoder", "seq_lens_decoder", "seq_lens_this_time"})
    .Outputs({"position_ids_out"})
    .SetKernelFn(PD_KERNEL(GetPositionIdsKernelV2))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPositionIdsV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPositionIdsV2InferDtype));;
