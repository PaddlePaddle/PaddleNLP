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
                                             const paddle::Tensor& seq_lens_decoder,
                                             const paddle::Tensor& seq_lens_encoder,
                                             const paddle::optional<paddle::Tensor>& output_padding_offset,
                                             int max_input_length) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xpu::ctx_guard RAII_GUARD(xpu_ctx->x_context());
  using XPUType = typename XPUTypeTrait<bfloat16>::Type; // only support bfloat16
  typedef paddle::bfloat16 data_t;
  const int token_num = tmp_out.dims()[0]; 
  const int dim = tmp_out.dims()[1]; 
  const int bsz = cum_offsets.shape()[0];

  std::vector<int> seq_lens_encoder_cpu(bsz, 0);
  std::vector<int> seq_lens_decoder_cpu(bsz, 0);
  std::vector<int> encoder_batch_idx; // 去除空隙的batch map
  std::vector<int> decoder_batch_idx; // 去除空隙的batch map
  std::vector<int> encoder_seq_lod;
  int r = xpu_memcpy(seq_lens_encoder_cpu.data(),
                 seq_lens_encoder.data<int>(),
                 sizeof(int32_t) * bsz,
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  r = xpu_memcpy(seq_lens_decoder_cpu.data(),
                 seq_lens_decoder.data<int>(),
                 sizeof(int32_t) * bsz,
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  int enc_batch = 0, dec_batch = 0;
  int batch_offset = 0;
  encoder_seq_lod.push_back(0);
  for(int i = 0; i < bsz; ++i){
    if(seq_lens_encoder_cpu[i] > 0){
      enc_batch++;
      encoder_batch_idx.push_back(i - batch_offset);
      encoder_seq_lod.push_back(seq_lens_encoder_cpu[i]);
      encoder_seq_lod[enc_batch] += encoder_seq_lod[enc_batch - 1];
    }
    else if(seq_lens_decoder_cpu[i] > 0){
      dec_batch++;
      decoder_batch_idx.push_back(i - batch_offset);
    }
    else{
        batch_offset++;
    }
  }         
  baidu::xpu::api::VectorParam<int32_t> encoder_seqs_lods_vp =
      baidu::xpu::api::VectorParam<int32_t>{encoder_seq_lod.data(), enc_batch + 1, nullptr}
          .to_xpu(RAII_GUARD);
  baidu::xpu::api::VectorParam<int32_t> encoder_batch_map_vp =
      baidu::xpu::api::VectorParam<int32_t>{encoder_batch_idx.data(), enc_batch, nullptr}
          .to_xpu(RAII_GUARD);
  baidu::xpu::api::VectorParam<int32_t> decoder_batch_map_vp =
      baidu::xpu::api::VectorParam<int32_t>{decoder_batch_idx.data(), dec_batch, nullptr}
          .to_xpu(RAII_GUARD);
  auto out = paddle::full({token_num, dim}, -2, tmp_out.type(), tmp_out.place()); 

  r = xftkernel::xft_eb_adjust_batch<XPUType, XPUType>(
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
                                                             const std::vector<int64_t>& seq_lens_decoder_shape,
                                                             const std::vector<int64_t>& seq_lens_encoder_shape,
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
                                                         const paddle::DataType& seq_lens_decoder_dtype,
                                                         const paddle::DataType& seq_lens_encoder_dtype,
                                                         const paddle::optional<paddle::DataType>& output_padding_offset_dtype) {
    return {tmp_out_dtype};
}

PD_BUILD_OP(adjust_batch)
    .Inputs({"tmp_out", "cum_offsets", "seq_lens_decoder", "seq_lens_encoder", paddle::Optional("output_padding_offset")})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(AdjustBatch))
    .SetInferShapeFn(PD_INFER_SHAPE(AdjustBatchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AdjustBatchInferDtype));