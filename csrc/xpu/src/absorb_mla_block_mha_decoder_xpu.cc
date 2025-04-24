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
#include <core/ctx_manager.h>
#include <core/xft_check.h>
#include <core/xft_event.h>
#include <core/xft_params.h>
#include <xft/xdnn_plugin.h>
#include <xft/operation/page_attn.h>
#include <xft/operation/fmha.h>
#include <flash_api.h> // link xfa
#include "ops.h"

namespace xftkernel = baidu::xpu::xftkernel;

template <typename T>
struct kl3_pa_TL_trait {
    using TL = T;
};
template <>
struct kl3_pa_TL_trait<bfloat16> {
    using TL = float;
};
std::vector<paddle::Tensor> MlaDeAttn(
    const paddle::Tensor& q,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& decoder_context_len,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& dec_batch_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const float softmax_scale,
    const int block_size,
    const int num_head,
    const int kv_lora_rank, 
    const int rope_head_dim,
    const int dim_qk,
    const int dim_v) {  
  baidu::xpu::api::plugin::print_times("[TIME BEGIN] MlaDeAttn" );
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xpu::ctx_guard RAII_GUARD(xpu_ctx->x_context());

  using QType = typename XPUTypeTrait<bfloat16>::Type;
  using CacheType = typename XPUTypeTrait<bfloat16>::Type;
  typedef paddle::bfloat16 qdata_t, cache_t;
  const auto& input_dims = q.dims();
  const int bsz = cum_offsets.dims()[0];
  const int token_num = input_dims[0];
  const int block_batch = block_tables.dims()[0]; // TODO参数含义 block_batch_  PageParam page_param_
  const int max_block_per_seq = block_tables.dims()[1];
  const int max_seq_len = block_size * max_block_per_seq;
  int dec_batch = dec_batch_tensor.data<int32_t>()[0];
  // 初始化输入：q k v
  auto q_xft = baidu::xpu::xft::xftTensor<QType, 3>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(q.data<qdata_t>())),
      std::array<int64_t, 3>{q.shape()[0],
                             q.shape()[1],
                             q.shape()[2]});  
  // 初始化输入：k cache
  auto kv_cache_xft = baidu::xpu::xft::xftTensor<CacheType, 4>(
  reinterpret_cast<CacheType*>(const_cast<paddle::bfloat16*>(kv_cache.data<cache_t>())),
  std::array<int64_t, 4>{kv_cache.shape()[0],
                          kv_cache.shape()[1],
                          kv_cache.shape()[2],
                          kv_cache.shape()[3]});                            
  // 初始化输入：block table
  auto block_tables_xft = baidu::xpu::xft::xftTensor<int, 2>(
  reinterpret_cast<int*>(const_cast<int*>(block_tables.data<int>())),
  std::array<int64_t, 2>{block_tables.shape()[0],
                          block_tables.shape()[1]}); 
  // 初始化输出tensor
  auto fmha_out = paddle::full({q.shape()[0], num_head * kv_lora_rank}, -2, q.type(), q.place()); 
  auto fmha_out_xft = baidu::xpu::xft::xftTensor<QType, 2>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(fmha_out.data<qdata_t>())),
      std::array<int64_t, 2>{fmha_out.shape()[0],
                             fmha_out.shape()[1]});

  // decoder
  if(dec_batch > 0){
    // context_len
    baidu::xpu::api::VectorParam<int32_t> context_len_vp{const_cast<int32_t*>(decoder_context_len_cpu.data<int32_t>()), dec_batch, const_cast<int32_t*>(decoder_context_len.data<int32_t>())};
    // real batch     
    baidu::xpu::api::VectorParam<int32_t> valid_batch_vp{const_cast<int32_t*>(decoder_batch_map_cpu.data<int32_t>()), dec_batch, const_cast<int32_t*>(decoder_batch_map.data<int32_t>())};
  
    // multi_latent_attention
    using TQ = bfloat16; 
    using TKVCACHE = bfloat16; 
    using TO = TQ; 
    using TGEMM = float; 
    using TEW = float;
    using TID = int;
    constexpr int quant_mode = 0;
    // xpu_ctx->x_context().set_debug_level(0xa1);
    int ret = baidu::xpu::xfa::multi_latent_attention<
            TQ, 
            TKVCACHE, 
            TO, 
            TGEMM,
            TEW, 
            TID, 
            quant_mode>(
            xpu_ctx->x_context(),
            fmha_out_xft.data(),
            q_xft.data(),
            kv_cache_xft.data(),
            block_tables_xft.data(),
            context_len_vp,
            valid_batch_vp,
            block_batch,
            max_seq_len,
            num_head,
            kv_lora_rank,
            rope_head_dim,
            nullptr, // attn_mask
            softmax_scale, // 0.13523377478122711f, // scale
            block_size,
            max_block_per_seq,
            -1,
            nullptr,
            nullptr,
            nullptr);
  } 
    baidu::xpu::api::plugin::print_times("[TIME END] MlaDeAttn");
    return {fmha_out};   
}

std::vector<std::vector<int64_t>> MlaDeAttnInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& kv_cache_shape,
    const std::vector<int64_t>& decoder_context_len_shape,
    const std::vector<int64_t>& decoder_batch_map_shape,
    const std::vector<int64_t>& decoder_context_len_cpu_shape,
    const std::vector<int64_t>& decoder_batch_map_cpu_shape,
    const std::vector<int64_t>& dec_batch_tensor_shape,
    const std::vector<int64_t>& padding_offsets_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& block_tables_shape,
    const float softmax_scale,
    const int block_size,
    const int num_head,
    const int kv_lora_rank, 
    const int rope_head_dim,
    const int dim_qk,
    const int dim_v) {  
  return {{q_shape[0], num_head * kv_lora_rank}};
}

std::vector<paddle::DataType> MlaDeAttnInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& kv_cache_dtype,
    const paddle::DataType& decoder_context_len_dtype,
    const paddle::DataType& decoder_batch_map_dtype, 
    const paddle::DataType& decoder_context_len_cpu_dtype,
    const paddle::DataType& decoder_batch_map_cpu_dtype,
    const paddle::DataType& dec_batch_tensor_dtype,
    const paddle::DataType& padding_offsets_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& block_tables_dtype,
    const float softmax_scale,
    const int block_size,
    const int num_head,
    const int kv_lora_rank, 
    const int rope_head_dim,
    const int dim_qk,
    const int dim_v) {  
    if (q_dtype == paddle::DataType::FLOAT16) {
        return {paddle::DataType::FLOAT16};
    } else if(q_dtype == paddle::DataType::BFLOAT16){
        return {paddle::DataType::BFLOAT16};
    } 
    else {
    PD_THROW("Only supported attr of compute_dtype in ['fp16','bfp16'].");
    }
}

PD_BUILD_OP(absorb_mla_block_mha_decoder_xpu)
    .Inputs({"q",
             "kv_cache",
             "decoder_context_len",
             "decoder_batch_map",
             "decoder_context_len_cpu",
             "decoder_batch_map_cpu",
             "dec_batch_tensor",
             "padding_offsets",
             "cum_offsets",
             "block_tables"})
    .Outputs({"fmha_out"})
    .Attrs({"softmax_scale: float",
            "block_size: int",
            "num_head: int",
            "kv_lora_rank: int",
            "rope_head_dim: int",
            "dim_qk: int",
            "dim_v: int"})
    .SetKernelFn(PD_KERNEL(MlaDeAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(MlaDeAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MlaDeAttnInferDtype));

