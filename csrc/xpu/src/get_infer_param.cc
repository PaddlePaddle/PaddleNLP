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
std::vector<paddle::Tensor> GetInferParam(const paddle::Tensor& seq_lens_encoder,
                                          const paddle::Tensor& seq_lens_decoder) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  const int bsz = seq_lens_encoder.dims()[0];
  // 判断逻辑
  std::vector<int32_t> seq_lens_encoder_cpu(bsz, 0);
  std::vector<int32_t> seq_lens_decoder_cpu(bsz, 0);
  std::vector<int32_t> encoder_batch_map(bsz, 0); // 
  std::vector<int32_t> decoder_batch_map(bsz, 0); // 
  std::vector<int32_t> encoder_batch_idx(bsz, 0); // 去除空隙的batch map
  std::vector<int32_t> decoder_batch_idx(bsz, 0); // 去除空隙的batch map
  std::vector<int32_t> encoder_seq_lod(bsz + 1, 0);
  std::vector<int32_t> decoder_context_len(bsz, 0);
  std::vector<int32_t> decoder_context_len_cache(bsz, 0);
  xpu_wait(xpu_ctx->x_context()->xpu_stream); // 是否需要！！！！TODO
  int r = xpu_memcpy(seq_lens_encoder_cpu.data(),
                 seq_lens_encoder.data<int32_t>(),
                 sizeof(int32_t) * bsz,
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  r = xpu_memcpy(seq_lens_decoder_cpu.data(),
                 seq_lens_decoder.data<int32_t>(),
                 sizeof(int32_t) * bsz,
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);

  int enc_batch = 0, dec_batch = 0;
  int total_enc_len = 0;
  int batch_offset = 0;
  for(int i = 0; i < bsz; ++i){
    if(seq_lens_encoder_cpu[i] > 0){
      enc_batch++;
      total_enc_len += seq_lens_encoder_cpu[i];
      encoder_batch_map[enc_batch - 1] = i;
      encoder_batch_idx[enc_batch - 1] = i - batch_offset;
      encoder_seq_lod[enc_batch] = seq_lens_encoder_cpu[i];
      encoder_seq_lod[enc_batch] += encoder_seq_lod[enc_batch - 1];
    }
    else if(seq_lens_decoder_cpu[i] > 0){
      dec_batch++;
      decoder_batch_map[dec_batch - 1] = i;
      decoder_batch_idx[dec_batch - 1] = i - batch_offset;
      decoder_context_len[dec_batch - 1] = seq_lens_decoder_cpu[i] + 1;
      decoder_context_len_cache[dec_batch - 1] = seq_lens_decoder_cpu[i];
    }
    else{
        batch_offset++;
    }
  }

  auto encoder_batch_map_tensor = paddle::full({encoder_batch_map.size()}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());
  auto decoder_batch_map_tensor = paddle::full({decoder_batch_map.size()}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());
  auto encoder_batch_idx_tensor = paddle::full({encoder_batch_idx.size()}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());
  auto decoder_batch_idx_tensor = paddle::full({decoder_batch_idx.size()}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());
  auto encoder_seq_lod_tensor = paddle::full({encoder_seq_lod.size()}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());
  auto decoder_context_len_tensor = paddle::full({decoder_context_len.size()}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());
  auto decoder_context_len_cache_tensor = paddle::full({decoder_context_len_cache.size()}, 0, seq_lens_encoder.type(), seq_lens_encoder.place());


  auto encoder_batch_map_tensor_cpu = paddle::full({encoder_batch_map.size()}, 0, seq_lens_encoder.type(), paddle::CPUPlace());
  auto decoder_batch_map_tensor_cpu = paddle::full({decoder_batch_map.size()}, 0, seq_lens_encoder.type(), paddle::CPUPlace());
  auto encoder_batch_idx_tensor_cpu = paddle::full({encoder_batch_idx.size()}, 0, seq_lens_encoder.type(), paddle::CPUPlace());
  auto decoder_batch_idx_tensor_cpu = paddle::full({decoder_batch_idx.size()}, 0, seq_lens_encoder.type(), paddle::CPUPlace());
  auto encoder_seq_lod_tensor_cpu = paddle::full({encoder_seq_lod.size()}, 0, seq_lens_encoder.type(), paddle::CPUPlace());
  auto decoder_context_len_tensor_cpu = paddle::full({decoder_context_len.size()}, 0, seq_lens_encoder.type(), paddle::CPUPlace());
  auto decoder_context_len_cache_tensor_cpu = paddle::full({decoder_context_len_cache.size()}, 0, seq_lens_encoder.type(), paddle::CPUPlace());
  
  auto enc_batch_tensor = paddle::full({1}, enc_batch, seq_lens_encoder.type(), paddle::CPUPlace());
  auto dec_batch_tensor = paddle::full({1}, dec_batch, seq_lens_encoder.type(), paddle::CPUPlace());
  auto total_enc_len_tensor = paddle::full({1}, total_enc_len, seq_lens_encoder.type(), paddle::CPUPlace());

  int ret = 0;
  
  ret = xpu_memcpy(reinterpret_cast<int32_t*>(const_cast<int32_t*>(encoder_batch_map_tensor.data<int32_t>())), encoder_batch_map.data(), sizeof(int32_t) * encoder_batch_map.size(), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  ret = xpu_memcpy(reinterpret_cast<int32_t*>(const_cast<int32_t*>(decoder_batch_map_tensor.data<int32_t>())), decoder_batch_map.data(), sizeof(int32_t) * decoder_batch_map.size(), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  ret = xpu_memcpy(reinterpret_cast<int32_t*>(const_cast<int32_t*>(encoder_batch_idx_tensor.data<int32_t>())), encoder_batch_idx.data(), sizeof(int32_t) * encoder_batch_idx.size(), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  ret = xpu_memcpy(reinterpret_cast<int32_t*>(const_cast<int32_t*>(decoder_batch_idx_tensor.data<int32_t>())), decoder_batch_idx.data(), sizeof(int32_t) * decoder_batch_idx.size(), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  ret = xpu_memcpy(reinterpret_cast<int32_t*>(const_cast<int32_t*>(encoder_seq_lod_tensor.data<int32_t>())), encoder_seq_lod.data(), sizeof(int32_t) * encoder_seq_lod.size(), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  ret = xpu_memcpy(reinterpret_cast<int32_t*>(const_cast<int32_t*>(decoder_context_len_tensor.data<int32_t>())), decoder_context_len.data(), sizeof(int32_t) * decoder_context_len.size(), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  ret = xpu_memcpy(reinterpret_cast<int32_t*>(const_cast<int32_t*>(decoder_context_len_cache_tensor.data<int32_t>())), decoder_context_len_cache.data(), sizeof(int32_t) * decoder_context_len_cache.size(), XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  
  std::memcpy(encoder_batch_map_tensor_cpu.data<int32_t>(), encoder_batch_map.data(), sizeof(int32_t) * encoder_batch_map.size());
  std::memcpy(decoder_batch_map_tensor_cpu.data<int32_t>(), decoder_batch_map.data(), sizeof(int32_t) * decoder_batch_map.size());
  std::memcpy(encoder_batch_idx_tensor_cpu.data<int32_t>(), encoder_batch_idx.data(), sizeof(int32_t) * encoder_batch_idx.size());
  std::memcpy(decoder_batch_idx_tensor_cpu.data<int32_t>(), decoder_batch_idx.data(), sizeof(int32_t) * decoder_batch_idx.size());
  std::memcpy(encoder_seq_lod_tensor_cpu.data<int32_t>(), encoder_seq_lod.data(), sizeof(int32_t) * encoder_seq_lod.size());
  std::memcpy(decoder_context_len_tensor_cpu.data<int32_t>(), decoder_context_len.data(), sizeof(int32_t) * decoder_context_len.size());
  std::memcpy(decoder_context_len_cache_tensor_cpu.data<int32_t>(), decoder_context_len_cache.data(), sizeof(int32_t) * decoder_context_len_cache.size());
  
  return {encoder_batch_map_tensor, 
          decoder_batch_map_tensor, 
          encoder_batch_idx_tensor, 
          decoder_batch_idx_tensor, 
          encoder_seq_lod_tensor, 
          decoder_context_len_tensor, 
          decoder_context_len_cache_tensor,
          encoder_batch_map_tensor_cpu,
          decoder_batch_map_tensor_cpu,
          encoder_batch_idx_tensor_cpu,
          decoder_batch_idx_tensor_cpu,
          encoder_seq_lod_tensor_cpu,
          decoder_context_len_tensor_cpu,
          decoder_context_len_cache_tensor_cpu,
          enc_batch_tensor,
          dec_batch_tensor,
          total_enc_len_tensor};
}

std::vector<std::vector<int64_t>> GetInferParamInferShape(const std::vector<int64_t>& seq_lens_encoder_shape,
                                                             const std::vector<int64_t>& seq_lens_decoder_shape) {
    return {seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape, 
            seq_lens_encoder_shape,
            {1},{1},{1}};
}

std::vector<paddle::DataType> GetInferParamInferDtype(const paddle::DataType& seq_lens_encoder_dtype,
                                                         const paddle::DataType& seq_lens_decoder_dtype) {
    return {seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype,
            seq_lens_encoder_dtype};
}

PD_BUILD_OP(get_infer_param)
    .Inputs({"seq_lens_encoder", "seq_lens_decoder"})
    .Outputs({"encoder_batch_map_tensor", 
              "decoder_batch_map_tensor", 
              "encoder_batch_idx_tensor", 
              "decoder_batch_idx_tensor", 
              "encoder_seq_lod_tensor", 
              "decoder_context_len_tensor", 
              "decoder_context_len_cache_tensor", 
              "encoder_batch_map_tensor_cpu", 
              "decoder_batch_map_tensor_cpu", 
              "encoder_batch_idx_tensor_cpu", 
              "decoder_batch_idx_tensor_cpu", 
              "encoder_seq_lod_tensor_cpu",
              "decoder_context_len_tensor_cpu",
              "decoder_context_len_cache_tensor_cpu",
              "enc_batch_tensor",
              "dec_batch_tensor",
              "total_enc_len_tensor"})
    .SetKernelFn(PD_KERNEL(GetInferParam))
    .SetInferShapeFn(PD_INFER_SHAPE(GetInferParamInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetInferParamInferDtype));