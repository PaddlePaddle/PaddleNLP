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
#include "ops.h"

namespace xftkernel = baidu::xpu::xftkernel;

template <paddle::DataType D>
class PaddleTypeToXPUType;

template <>
class PaddleTypeToXPUType<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PaddleTypeToXPUType<paddle::DataType::FLOAT16> {
public:
  typedef float16 DataType;
  typedef paddle::float16 data_t;
};

template <>
class PaddleTypeToXPUType<paddle::DataType::BFLOAT16> {
public:
  typedef bfloat16 DataType;
  typedef paddle::bfloat16 data_t;
};

template <paddle::DataType T>
std::vector<paddle::Tensor> PrefillMLAWriteCache(
                    const paddle::Tensor& kv_nope,
                    const paddle::Tensor& kv_pe,
                    const paddle::Tensor& encoder_seq_lod,
                    const paddle::Tensor& encoder_batch_map,
                    const paddle::Tensor& start_token_raw, // encoder cache写时 start token为0 （不算prefix cache）
                    const paddle::Tensor& encoder_seq_lod_cpu,
                    const paddle::Tensor& encoder_batch_map_cpu,
                    const paddle::Tensor& start_token_raw_cpu, // encoder cache写时 start token为0 （不算prefix cache）
                    const paddle::Tensor& enc_batch_tensor,
                    const paddle::Tensor& dec_batch_tensor,
                    const paddle::Tensor& padding_offsets,
                    const paddle::Tensor& cum_offsets,
                    const paddle::Tensor& block_tables,
                    const int max_seq_len,
                    const int kv_num_heads,
                    paddle::Tensor* kv_cache) {
  baidu::xpu::api::plugin::print_times("[TIME BEGIN] PrefillMLAWriteCache");

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  typedef PaddleTypeToXPUType<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const auto& kv_nope_dims = kv_nope.dims();
  const auto& kv_pe_dims = kv_pe.dims();
  const auto& kv_cache_dims = kv_cache->dims();
  auto num_tokens = kv_nope_dims[0];
  auto block_size = kv_cache_dims[2];
  auto block_batch = block_tables.dims()[0];
  auto max_blocks_per_seq = block_tables.dims()[1];
  auto kv_lora_rank = kv_nope_dims[kv_nope_dims.size() - 1];
  auto rope_head_dim = kv_pe_dims[kv_pe_dims.size() - 1];
  auto bsz = cum_offsets.dims()[0];
  int enc_batch = enc_batch_tensor.data<int32_t>()[0];
  int dec_batch = dec_batch_tensor.data<int32_t>()[0];
  // 初始化输入
  auto kv_nope_xft = baidu::xpu::xft::xftTensor<DataType_, 2>(
      reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
      std::array<int64_t, 2>{num_tokens,
                             kv_lora_rank});
  auto kv_pe_xft = baidu::xpu::xft::xftTensor<DataType_, 2>(
      reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
      std::array<int64_t, 2>{num_tokens,
                             rope_head_dim});
  // 初始化输入 cache
  auto cache_xft = baidu::xpu::xft::xftTensor<DataType_, 4>(
  reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_cache->data<data_t>())),
  std::array<int64_t, 4>{kv_cache->shape()[0],
                          kv_cache->shape()[1],
                          kv_cache->shape()[2],
                          kv_cache->shape()[3]});  
  // 初始化输入：block table
  auto block_tables_xft = baidu::xpu::xft::xftTensor<int, 2>(
  reinterpret_cast<int*>(const_cast<int*>(block_tables.data<int>())),
  std::array<int64_t, 2>{block_tables.shape()[0],
                          block_tables.shape()[1]}); 
  // 初始化concat
  std::vector<const DataType_*> concat_input;
  concat_input.push_back(kv_nope_xft.data());
  concat_input.push_back(kv_pe_xft.data());
  auto concat_out = paddle::full({num_tokens, kv_lora_rank + rope_head_dim}, -2, kv_nope.type(), kv_nope.place()); 
  auto concat_out_xft = baidu::xpu::xft::xftTensor<DataType_, 2>(
      reinterpret_cast<DataType_*>(const_cast<data_t*>(concat_out.data<data_t>())),
      std::array<int64_t, 2>{concat_out.shape()[0],
                             concat_out.shape()[1]});

  // 拼接kv_nope和kv_pe
  baidu::xpu::api::concat<DataType_>(xpu_ctx->x_context(),
                                      concat_input,
                                      concat_out_xft.data(),
                                      {{num_tokens, kv_lora_rank},{num_tokens, rope_head_dim}},
                                      1);


  baidu::xpu::api::VectorParam<int32_t> context_len_vp{const_cast<int32_t*>(encoder_seq_lod_cpu.data<int32_t>()), enc_batch + 1, const_cast<int32_t*>(encoder_seq_lod.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> valid_batch_vp{const_cast<int32_t*>(encoder_batch_map_cpu.data<int32_t>()), enc_batch, const_cast<int32_t*>(encoder_batch_map.data<int32_t>())};
  baidu::xpu::api::VectorParam<int32_t> start_tokens_vp{const_cast<int32_t*>(start_token_raw_cpu.data<int32_t>()), enc_batch, const_cast<int32_t*>(start_token_raw.data<int32_t>())};
  int ret_cache = xftkernel::xft_reshape_cached_kv<float16, float16, int>(
          xpu_ctx->x_context(),
          reinterpret_cast<float16*>(concat_out_xft.data()),
          reinterpret_cast<float16*>(cache_xft.data()),
          block_tables_xft.data(),
          context_len_vp,
          start_tokens_vp,
          valid_batch_vp,
          enc_batch,
          1, // num_head == 1 吸收版
          kv_lora_rank + rope_head_dim,
          block_batch,
          block_size,
          max_blocks_per_seq,
          "BLHD", // qkv_layout,
          "HLD", // page_layout,
          nullptr, // scale,
          nullptr, // batch_max_ptrs, 
          nullptr); // max_ptrs

  baidu::xpu::api::plugin::print_times("[TIME END] PrefillMLAWriteCache");

  return {};
}

std::vector<paddle::Tensor> PrefillMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& start_token_raw, // encoder cache写时 start token为0 （不算prefix cache）
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& start_token_raw_cpu,
    const paddle::Tensor& enc_batch_tensor,
    const paddle::Tensor& dec_batch_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len,
    const int kv_num_heads) {

  switch (kv_pe.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return PrefillMLAWriteCache<paddle::DataType::BFLOAT16>(
                              kv_nope,
                              kv_pe,
                              encoder_seq_lod,
                              encoder_batch_map,
                              start_token_raw,
                              encoder_seq_lod_cpu,
                              encoder_batch_map_cpu,                            
                              start_token_raw_cpu,
                              enc_batch_tensor,
                              dec_batch_tensor,
                              padding_offsets,
                              cum_offsets,
                              block_tables,
                              max_seq_len,
                              kv_num_heads,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
    case paddle::DataType::FLOAT16: {
      return PrefillMLAWriteCache<paddle::DataType::FLOAT16>(
                              kv_nope,
                              kv_pe,
                              encoder_seq_lod,
                              encoder_batch_map,
                              start_token_raw,
                              encoder_seq_lod_cpu,
                              encoder_batch_map_cpu,                            
                              start_token_raw_cpu,
                              enc_batch_tensor,
                              dec_batch_tensor,
                              padding_offsets,
                              cum_offsets,
                              block_tables,
                              max_seq_len,
                              kv_num_heads,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
  }
  return {};
}

template <paddle::DataType T>
std::vector<paddle::Tensor> DecodeMLAWriteCache(
                    const paddle::Tensor& kv_nope,
                    const paddle::Tensor& kv_pe,
                    const paddle::Tensor& decoder_context_len_cache,
                    const paddle::Tensor& decoder_batch_map,
                    const paddle::Tensor& kv_seq_lod_raw, // decoder cache写时，lod为0，1，2，3，……
                    const paddle::Tensor& decoder_context_len_cache_cpu,
                    const paddle::Tensor& decoder_batch_map_cpu,
                    const paddle::Tensor& kv_seq_lod_raw_cpu, // decoder cache写时，lod为0，1，2，3，……
                    const paddle::Tensor& enc_batch_tensor,
                    const paddle::Tensor& dec_batch_tensor,
                    const paddle::Tensor& padding_offsets,
                    const paddle::Tensor& cum_offsets,
                    const paddle::Tensor& block_tables,
                    const int max_seq_len,
                    const int kv_num_heads,
                    const bool speculate_decoder,
                    paddle::Tensor* kv_cache) {
 
  baidu::xpu::api::plugin::print_times("[TIME BEGIN] DecodeMLAWriteCache");
                       
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xpu::ctx_guard RAII_GUARD(xpu_ctx->x_context());
  typedef PaddleTypeToXPUType<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const auto& kv_nope_dims = kv_nope.dims();
  const auto& kv_pe_dims = kv_pe.dims();
  const auto& kv_cache_dims = kv_cache->dims();
  auto num_tokens = kv_nope_dims[0];
  auto block_size = kv_cache_dims[2];
  auto block_batch = block_tables.dims()[0];
  auto max_blocks_per_seq = block_tables.dims()[1];
  auto kv_lora_rank = kv_nope_dims[kv_nope_dims.size() - 1];
  auto rope_head_dim = kv_pe_dims[kv_pe_dims.size() - 1];
  auto bsz = cum_offsets.dims()[0];
  int enc_batch = enc_batch_tensor.data<int32_t>()[0];
  int dec_batch = dec_batch_tensor.data<int32_t>()[0];

  // 初始化输入
  auto kv_nope_xft = baidu::xpu::xft::xftTensor<DataType_, 2>(
      reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
      std::array<int64_t, 2>{num_tokens,
                             kv_lora_rank});
  auto kv_pe_xft = baidu::xpu::xft::xftTensor<DataType_, 2>(
      reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
      std::array<int64_t, 2>{num_tokens,
                             rope_head_dim});
  // 初始化输入 cache
  auto cache_xft = baidu::xpu::xft::xftTensor<DataType_, 4>(
  reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_cache->data<data_t>())),
  std::array<int64_t, 4>{kv_cache->shape()[0],
                          kv_cache->shape()[1],
                          kv_cache->shape()[2],
                          kv_cache->shape()[3]});  
  // 初始化输入：block table
  auto block_tables_xft = baidu::xpu::xft::xftTensor<int, 2>(
  reinterpret_cast<int*>(const_cast<int*>(block_tables.data<int>())),
  std::array<int64_t, 2>{block_tables.shape()[0],
                          block_tables.shape()[1]}); 
  // 初始化concat
  // std::vector<DataType_*> concat_input = {kv_nope_xft.data(), kv_pe_xft.data()};
  std::vector<const DataType_*> concat_input;
  concat_input.push_back(kv_nope_xft.data());
  concat_input.push_back(kv_pe_xft.data());
  auto concat_out = paddle::full({num_tokens, kv_lora_rank + rope_head_dim}, -2, kv_nope.type(), kv_nope.place()); 
  auto concat_out_xft = baidu::xpu::xft::xftTensor<DataType_, 2>(
      reinterpret_cast<DataType_*>(const_cast<data_t*>(concat_out.data<data_t>())),
      std::array<int64_t, 2>{concat_out.shape()[0],
                             concat_out.shape()[1]});

  // 拼接kv_nope和kv_pe
  baidu::xpu::api::concat<DataType_>(xpu_ctx->x_context(),
                                      concat_input,
                                      concat_out_xft.data(),
                                      {{num_tokens, kv_lora_rank},{num_tokens, rope_head_dim}},
                                      1);




  // context_len
  baidu::xpu::api::VectorParam<int32_t> context_len_vp_cache{const_cast<int32_t*>(decoder_context_len_cache_cpu.data<int32_t>()), dec_batch, const_cast<int32_t*>(decoder_context_len_cache.data<int32_t>())};
  // real batch     
  baidu::xpu::api::VectorParam<int32_t> valid_batch_vp{const_cast<int32_t*>(decoder_batch_map_cpu.data<int32_t>()), dec_batch, const_cast<int32_t*>(decoder_batch_map.data<int32_t>())};
  // baidu::xpu::api::VectorParam<int32_t> kv_seq_lod_vp{const_cast<int32_t*>(kv_seq_lod_raw_cpu.data<int32_t>()), dec_batch + 1, const_cast<int32_t*>(kv_seq_lod_raw.data<int32_t>())};
  
  // baidu::xpu::api::VectorParam<int32_t> kv_seq_lod_vp = baidu::xpu::api::VectorParam<int32_t>{const_cast<int32_t*>(kv_seq_lod_raw_cpu.data<int32_t>()), dec_batch + 1, const_cast<int32_t*>(kv_seq_lod_raw.data<int32_t>())};
  
  std::vector<int> kv_seq_lod(dec_batch + 1);
  std::iota(kv_seq_lod.begin(), kv_seq_lod.end(), 0);
  baidu::xpu::api::VectorParam<int32_t> kv_seq_lod_vp =
      baidu::xpu::api::VectorParam<int32_t>{kv_seq_lod.data(), dec_batch + 1, nullptr}.to_xpu(RAII_GUARD);


    // std::cout << "Tensor dtype: " << kv_seq_lod_raw_cpu.dtype() << std::endl;
    // std::cout << "Tensor place: " << kv_seq_lod_raw_cpu.place() << std::endl;
    
    // // 打印形状信息
    // std::cout << "Tensor shape: [";
    // for (int i = 0; i < kv_seq_lod_raw_cpu.dims().size(); ++i) {
    //     if (i > 0) std::cout << ", ";
    //     std::cout << kv_seq_lod_raw_cpu.dims()[i];
    // }
    // std::cout << "]" << std::endl;

    // std::cout << "Tensor dtype: " << kv_seq_lod_raw.dtype() << std::endl;
    // std::cout << "Tensor place: " << kv_seq_lod_raw.place() << std::endl;
    
    // // 打印形状信息
    // std::cout << "Tensor shape: [";
    // for (int i = 0; i < kv_seq_lod_raw.dims().size(); ++i) {
    //     if (i > 0) std::cout << ", ";
    //     std::cout << kv_seq_lod_raw.dims()[i];
    // }
    // std::cout << "]" << std::endl;


  int ret_cache = xftkernel::xft_reshape_cached_kv<float16, float16, int>(
          xpu_ctx->x_context(),
          reinterpret_cast<float16*>(concat_out_xft.data()),
          reinterpret_cast<float16*>(cache_xft.data()),
          block_tables_xft.data(),
          kv_seq_lod_vp,
          context_len_vp_cache,
          valid_batch_vp,
          dec_batch,
          1, //  num_head == 1
          kv_lora_rank + rope_head_dim,
          block_batch,
          block_size,
          max_blocks_per_seq,
          "BLHD", // qkv_layout,
          "HLD", // page_layout,
          nullptr, // scale,
          nullptr, // batch_max_ptrs, 
          nullptr); // max_ptrs

 
  baidu::xpu::api::plugin::print_times("[TIME END] DecodeMLAWriteCache");

  return {};
}

std::vector<paddle::Tensor> DecodeMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& decoder_context_len_cache,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& kv_seq_lod_raw,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& kv_seq_lod_raw_cpu,
    const paddle::Tensor& enc_batch_tensor,
    const paddle::Tensor& dec_batch_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len,
    const int kv_num_heads,
    const bool speculate_decoder) {
  switch (kv_pe.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return DecodeMLAWriteCache<paddle::DataType::BFLOAT16>(
                              kv_nope,
                              kv_pe,
                              decoder_context_len_cache,
                              decoder_batch_map,
                              kv_seq_lod_raw,
                              decoder_context_len_cache_cpu,
                              decoder_batch_map_cpu,
                              kv_seq_lod_raw_cpu,
                              enc_batch_tensor,
                              dec_batch_tensor,
                              padding_offsets,
                              cum_offsets,
                              block_tables,
                              max_seq_len,
                              kv_num_heads,
                              speculate_decoder,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
    case paddle::DataType::FLOAT16: {
      return DecodeMLAWriteCache<paddle::DataType::FLOAT16>(
                              kv_nope,
                              kv_pe,
                              decoder_context_len_cache,
                              decoder_batch_map,
                              kv_seq_lod_raw,
                              decoder_context_len_cache_cpu,
                              decoder_batch_map_cpu,
                              kv_seq_lod_raw_cpu,
                              enc_batch_tensor,
                              dec_batch_tensor,
                              padding_offsets,
                              cum_offsets,
                              block_tables,
                              max_seq_len,
                              kv_num_heads,
                              speculate_decoder,
                              const_cast<paddle::Tensor*>(&kv_cache));
    }
  }
  return {};
}


PD_BUILD_OP(prefill_mla_write_cache_xpu)
    .Inputs({"kv_nope",
             "kv_pe",
             "kv_cache",
             "encoder_seq_lod",
             "encoder_batch_map",
             "start_token_raw",
             "encoder_seq_lod_cpu",
             "encoder_batch_map_cpu",
             "start_token_raw_cpu",
             "enc_batch_tensor",
             "dec_batch_tensor",
             "padding_offsets",
             "cum_offsets",
             "block_tables"})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string",
            "max_seq_len: int",
            "kv_num_heads: int"})
    .SetKernelFn(PD_KERNEL(PrefillMLAWriteCacheKernel));

PD_BUILD_OP(decode_mla_write_cache_xpu)
    .Inputs({"kv_nope",
             "kv_pe",
             "kv_cache",
             "decoder_context_len_cache",
             "decoder_batch_map",
             "kv_seq_lod_raw",
             "decoder_context_len_cache_cpu",
             "decoder_batch_map_cpu",
             "kv_seq_lod_raw_cpu",
             "enc_batch_tensor",
             "dec_batch_tensor",
             "padding_offsets",
             "cum_offsets",
             "block_tables"})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string",
            "max_seq_len: int",
            "kv_num_heads: int",
            "speculate_decoder: bool"})
    .SetKernelFn(PD_KERNEL(DecodeMLAWriteCacheKernel));