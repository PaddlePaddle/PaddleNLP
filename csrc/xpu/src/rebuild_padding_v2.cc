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
#include "xpu/plugin.h"

std::vector<paddle::Tensor> RebuildPaddingV2(const paddle::Tensor& tmp_out, // [token_num, dim_embed]
                                             const paddle::Tensor& cum_offsets, // [bsz, 1]
                                             const paddle::Tensor& seq_lens_decoder,
                                             const paddle::Tensor& seq_lens_encoder,
                                             int max_input_length) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  std::vector<int64_t> tmp_out_shape = tmp_out.shape();
  const int token_num = tmp_out_shape[0];
  const int dim_embed = tmp_out_shape[1];
  const int bsz = cum_offsets.shape()[0];
  auto out = paddle::full({bsz, dim_embed}, 0, tmp_out.dtype(), tmp_out.place());
  int elem_nums = out.numel();
  switch (tmp_out.type()) {
    case paddle::DataType::FLOAT16: {
      using XPUType = typename XPUTypeTrait<float16>::Type;
      typedef paddle::float16 data_t;
      int r = baidu::xpu::api::plugin::rebuild_padding(
          xpu_ctx->x_context(),
          reinterpret_cast<XPUType*>(out.data<data_t>()),
          reinterpret_cast<const XPUType*>(tmp_out.data<data_t>()),
          cum_offsets.data<int>(),
          seq_lens_decoder.data<int>(), 
          seq_lens_encoder.data<int>(),
          max_input_length, 
          dim_embed, 
          elem_nums
          );
      PD_CHECK(r == 0, "xpu::plugin::rebuild_padding failed.");
    } break;
    case paddle::DataType::FLOAT32: {
      int r = baidu::xpu::api::plugin::rebuild_padding(
          xpu_ctx->x_context(),
          out.data<float>(),
          tmp_out.data<float>(),
          cum_offsets.data<int>(),
          seq_lens_decoder.data<int>(), 
          seq_lens_encoder.data<int>(),
          max_input_length, 
          dim_embed, 
          elem_nums
          );
      PD_CHECK(r == 0, "xpu::plugin::rebuild_padding failed.");
    } break;
    default:
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and float32 are supported. ");
      break;
  }
  return {out};
}

std::vector<std::vector<int64_t>> RebuildPaddingV2InferShape(const std::vector<int64_t>& tmp_out_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& seq_lens_decoder_shape,
                                                             const std::vector<int64_t>& seq_lens_encoder_shape) {
    int64_t bsz = cum_offsets_shape[0];
    int64_t dim_embed = tmp_out_shape[1];
    return {{bsz, dim_embed}};
}

std::vector<paddle::DataType> RebuildPaddingV2InferDtype(const paddle::DataType& tmp_out_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& seq_lens_decoder_dtype,
                                                         const paddle::DataType& seq_lens_encoder_dtype) {
    return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding_v2)
    .Inputs({"tmp_out", "cum_offsets", "seq_lens_decoder", "seq_lens_encoder"})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(RebuildPaddingV2))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingV2InferDtype));