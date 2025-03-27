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
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& max_enc_len_this_time,
    const paddle::Tensor& max_dec_len_this_time,
    const paddle::Tensor& max_len_kv,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_input_length,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const int block_size,
    const int num_head,
    const int kv_lora_rank, 
    const int rope_head_dim,
    const int dim_qk,
    const int dim_v,
    const bool causal,
    const bool speculate_decoder) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);

  xpu::ctx_guard RAII_GUARD(xpu_ctx->x_context());

  using QType = typename XPUTypeTrait<bfloat16>::Type;
  using CacheType = typename XPUTypeTrait<bfloat16>::Type;
  typedef paddle::bfloat16 qdata_t, cache_t;
  const auto& input_dims = q.dims();
  const int bsz = seq_lens_encoder.dims()[0];
  const int token_num = input_dims[0];
  const int block_batch = block_tables.dims()[0]; // TODO参数含义 block_batch_  PageParam page_param_
  const int max_block_per_seq = block_tables.dims()[1];
  const int max_seq_len = block_size * max_block_per_seq;
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
  // encoder 判断逻辑
  std::vector<int> seq_lens_encoder_cpu(bsz, 0);
  std::vector<int> seq_lens_decoder_cpu(bsz, 0);
  std::vector<int> encoder_batch_map; // 
  std::vector<int> decoder_batch_map; // 
  std::vector<int> encoder_batch_idx; // 去除空隙的batch map
  std::vector<int> decoder_batch_idx; // 去除空隙的batch map
  std::vector<int> encoder_seq_lod;
  std::vector<int> decoder_context_len;
  int r = xpu_memcpy(seq_lens_encoder_cpu.data(),
                 seq_lens_encoder.data<int>(),
                 sizeof(int32_t) * bsz,
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
  r = xpu_memcpy(seq_lens_decoder_cpu.data(),
                 seq_lens_decoder.data<int>(),
                 sizeof(int32_t) * bsz,
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);

  int enc_batch = 0, dec_batch = 0;
  int64_t total_enc_len = 0;
  int batch_offset = 0;
  encoder_seq_lod.push_back(0);
  for(int i = 0; i < bsz; ++i){
    if(seq_lens_encoder_cpu[i] > 0){
      enc_batch++;
      total_enc_len += seq_lens_encoder_cpu[i];
      encoder_batch_map.push_back(i);
      encoder_batch_idx.push_back(i - batch_offset);
      encoder_seq_lod.push_back(seq_lens_encoder_cpu[i]);
      encoder_seq_lod[enc_batch] += encoder_seq_lod[enc_batch - 1];
    }
    else if(seq_lens_decoder_cpu[i] > 0){
      dec_batch++;
      decoder_batch_map.push_back(i);
      decoder_batch_idx.push_back(i - batch_offset);
      decoder_context_len.push_back(seq_lens_decoder_cpu[i] + 1);
    }
    else{
        batch_offset++;
    }
  }

  // decoder
  if(max_dec_len_this_time.data<int>()[0] > 0){
    // context_len
    baidu::xpu::api::VectorParam<int32_t> context_len_vp =
        baidu::xpu::api::VectorParam<int32_t>{decoder_context_len.data(), dec_batch, nullptr}
            .to_xpu(RAII_GUARD);
    // real batch     
    baidu::xpu::api::VectorParam<int32_t> valid_batch_vp =
        baidu::xpu::api::VectorParam<int32_t>{decoder_batch_map.data(), dec_batch, nullptr}
            .to_xpu(RAII_GUARD);

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

    return {fmha_out};   
}

std::vector<std::vector<int64_t>> MlaDeAttnInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& kv_cache_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& padding_offsets_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& encoder_batch_ids_shape,
    const std::vector<int64_t>& encoder_tile_ids_per_batch_shape,
    const std::vector<int64_t>& encoder_num_blocks_shape,
    const std::vector<int64_t>& kv_batch_ids_shape,
    const std::vector<int64_t>& kv_tile_ids_per_batch_shape,
    const std::vector<int64_t>& kv_num_blocks_shape,
    const std::vector<int64_t>& decoder_batch_ids_shape,
    const std::vector<int64_t>& decoder_tile_ids_per_batch_shape,
    const std::vector<int64_t>& decoder_num_blocks_shape,
    const std::vector<int64_t>& max_enc_len_this_time_shape,
    const std::vector<int64_t>& max_dec_len_this_time_shape,
    const std::vector<int64_t>& max_len_kv_shape,
    const paddle::optional<std::vector<int64_t>>& rotary_embs_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_bias_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_out_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_zp_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_zp_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_shifts_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_smooths_shape,    
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_input_length,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const int block_size,
    const int num_head,
    const int kv_lora_rank, 
    const int rope_head_dim,
    const int dim_qk,
    const int dim_v,
    const bool causal,
    const bool speculate_decoder) {
  return {{q_shape[0], num_head * kv_lora_rank}};
}

std::vector<paddle::DataType> MlaDeAttnInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& kv_cache_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& padding_offsets_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& encoder_batch_ids_dtype,
    const paddle::DataType& encoder_tile_ids_per_batch_dtype,
    const paddle::DataType& encoder_num_blocks_dtype,
    const paddle::DataType& kv_batch_ids_dtype,
    const paddle::DataType& kv_tile_ids_per_batch_dtype,
    const paddle::DataType& kv_num_blocks_dtype,
    const paddle::DataType& decoder_batch_ids_dtype,
    const paddle::DataType& decoder_tile_ids_per_batch_dtype,
    const paddle::DataType& decoder_num_blocks_dtype,
    const paddle::DataType& max_enc_len_this_time_dtype,
    const paddle::DataType& max_dec_len_this_time_dtype,
    const paddle::DataType& max_len_kv_dtype,
    const paddle::optional<paddle::DataType>& rotary_embs_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_dtype,
    const paddle::optional<paddle::DataType>& qkv_bias_dtype,
    const paddle::optional<paddle::DataType>& qkv_out_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_zp_dtype,
    const paddle::optional<paddle::DataType>& cache_v_zp_dtype,
    const paddle::optional<paddle::DataType>& out_linear_shifts_dtype,
    const paddle::optional<paddle::DataType>& out_linear_smooths_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_input_length,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const int block_size,
    const int num_head,
    const int kv_lora_rank, 
    const int rope_head_dim,
    const int dim_qk,
    const int dim_v,
    const bool causal,
    const bool speculate_decoder) {
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
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "padding_offsets",
             "cum_offsets",
             "block_tables",
             "encoder_batch_ids",
             "encoder_tile_ids_per_batch",
             "encoder_num_blocks",
             "kv_batch_ids",
             "kv_tile_ids_per_batch",
             "kv_num_blocks",
             "decoder_batch_ids",
             "decoder_tile_ids_per_batch",
             "decoder_num_blocks",
             "max_enc_len_this_time",
             "max_dec_len_this_time",
             "max_len_kv",
             paddle::Optional("rotary_embs"),
             paddle::Optional("attn_mask"),
             paddle::Optional("qkv_bias"),
             paddle::Optional("qkv_out_scales"),
             paddle::Optional("cache_k_quant_scales"),
             paddle::Optional("cache_v_quant_scales"),
             paddle::Optional("cache_k_dequant_scales"),
             paddle::Optional("cache_v_dequant_scales"),
             paddle::Optional("cache_k_zp"),
             paddle::Optional("cache_v_zp"),
             paddle::Optional("out_linear_shifts"),
             paddle::Optional("out_linear_smooths")})
    .Outputs({"fmha_out"})
    .Attrs({"cache_quant_type: std::string",
            "use_neox_rotary_style: bool",
            "max_input_length: int",
            "softmax_scale: float",
            "quant_max_bound: float",
            "quant_min_bound: float",
            "out_linear_in_scale: float",
            "speculate_max_draft_token_num: int",
            "block_size: int",
            "num_head: int",
            "kv_lora_rank: int",
            "rope_head_dim: int",
            "dim_qk: int",
            "dim_v: int",
            "causal: bool",
            "speculate_decoder: bool"})
    .SetKernelFn(PD_KERNEL(MlaDeAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(MlaDeAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MlaDeAttnInferDtype));

