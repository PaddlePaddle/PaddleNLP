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
#include <xft/xdnn_plugin.h>
namespace xftkernel = baidu::xpu::xftkernel;
std::vector<paddle::Tensor> AdjustBatch(const paddle::Tensor& tmp_out, // [token_num, dim_embed]
                                             const paddle::Tensor& cum_offsets, // [bsz, 1]
                                             const paddle::Tensor& encoder_seq_lod,
                                             const paddle::Tensor& encoder_batch_idx,
                                             const paddle::Tensor& decoder_batch_idx,
                                             const paddle::Tensor& encoder_seq_lod_cpu,
                                             const paddle::Tensor& encoder_batch_idx_cpu,
                                             const paddle::Tensor& decoder_batch_idx_cpu,
                                             const paddle::Tensor& enc_batch_tensor,
                                             const paddle::Tensor& dec_batch_tensor,
                                             const paddle::optional<paddle::Tensor>& output_padding_offset,
                                             int max_input_length) {

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  using XPUType = typename XPUTypeTrait<bfloat16>::Type; // only support bfloat16
  typedef paddle::bfloat16 data_t;
  const int token_num = tmp_out.dims()[0]; 
  const int dim = tmp_out.dims()[1]; 
  const int bsz = cum_offsets.shape()[0];  
  int enc_batch = enc_batch_tensor.data<int32_t>()[0];
  int dec_batch = dec_batch_tensor.data<int32_t>()[0];

  baidu::xpu::api::VectorParam<int32_t> encoder_seqs_lods_vp{const_cast<int32_t*>(encoder_seq_lod_cpu.data<int32_t>()), enc_batch + 1, const_cast<int32_t*>(encoder_seq_lod.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> encoder_batch_map_vp{const_cast<int32_t*>(encoder_batch_idx_cpu.data<int32_t>()), enc_batch, const_cast<int32_t*>(encoder_batch_idx.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> decoder_batch_map_vp{const_cast<int32_t*>(decoder_batch_idx_cpu.data<int32_t>()), dec_batch, const_cast<int32_t*>(decoder_batch_idx.data<int32_t>())};

  auto out = paddle::full({token_num, dim}, -2, tmp_out.type(), tmp_out.place()); 

  int r = xftkernel::xft_eb_adjust_batch<XPUType, XPUType>(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPUType*>(tmp_out.data<data_t>()),
          reinterpret_cast<XPUType*>(out.data<data_t>()),
          encoder_seqs_lods_vp,
          encoder_batch_map_vp,
          decoder_batch_map_vp,
          dim);
  return {out};
}

std::vector<std::vector<int64_t>> AdjustBatchInferShape(const std::vector<int64_t>& tmp_out_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& encoder_seq_lod_shape,
                                                             const std::vector<int64_t>& encoder_batch_idx_shape,
                                                             const std::vector<int64_t>& decoder_batch_idx_shape,
                                                             const std::vector<int64_t>& encoder_seq_lod_cpu_shape,
                                                             const std::vector<int64_t>& encoder_batch_idx_cpu_shape,
                                                             const std::vector<int64_t>& decoder_batch_idx_cpu_shape, 
                                                             const std::vector<int64_t>& enc_batch_tensor_shape,
                                                             const std::vector<int64_t>& dec_batch_tensor_shape,                                                       
                                                             const paddle::optional<std::vector<int64_t>>& output_padding_offset_shape) {
    if (output_padding_offset_shape) {
      PD_THROW("speculative decoding is not supported in XPU.");
    }
    int64_t token_num = tmp_out_shape[0];
    int64_t dim_embed = tmp_out_shape[1];
    return {{token_num, dim_embed}};
}

std::vector<paddle::DataType> AdjustBatchInferDtype(const paddle::DataType& tmp_out_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& encoder_seq_lod_dtype,
                                                         const paddle::DataType& encoder_batch_idx_dtype,
                                                         const paddle::DataType& decoder_batch_idx_dtype,
                                                         const paddle::DataType& encoder_seq_lod_cpu_dtype,
                                                         const paddle::DataType& encoder_batch_idx_cpu_dtype,
                                                         const paddle::DataType& decoder_batch_idx_cpu_dtype,  
                                                         const paddle::DataType& enc_batch_tensor_dtype,
                                                         const paddle::DataType& dec_batch_tensor_dtype,
                                                         const paddle::optional<paddle::DataType>& output_padding_offset_dtype) {
    return {tmp_out_dtype};
}

PD_BUILD_OP(adjust_batch)
    .Inputs({"tmp_out", "cum_offsets", "encoder_seq_lod", "encoder_batch_idx", "decoder_batch_idx", "encoder_seq_lod_cpu", "encoder_batch_idx_cpu", "decoder_batch_idx_cpu", "enc_batch_tensor", "dec_batch_tensor", paddle::Optional("output_padding_offset")})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(AdjustBatch))
    .SetInferShapeFn(PD_INFER_SHAPE(AdjustBatchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AdjustBatchInferDtype));