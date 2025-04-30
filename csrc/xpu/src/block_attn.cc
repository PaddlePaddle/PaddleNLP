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
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"
#include "ops.h"

#include <core/ctx_manager.h>
#include <core/xft_check.h>
#include <core/xft_event.h>
#include <core/xft_params.h>
#include <xft/xdnn_plugin.h>
#include <xft/operation/page_attn.h>
#include <xft/operation/fmha.h>
#include <flash_api.h> // link xfa

namespace xftkernel = baidu::xpu::xftkernel;
namespace api = baidu::xpu::api;

std::vector<paddle::Tensor> MlaBlockAttn(
    const paddle::Tensor& attn_input, // [m, 7168]  
    const paddle::Tensor& q_a_proj_weights, // self.q_a_proj_weights[i] [7168, 1536] 【enc】【dec】
    const paddle::Tensor& q_b_proj_weights, // self.q_b_proj_weights[i] [1536, 24576] 【enc】【dec】
    const paddle::Tensor& q_a_layernorm_weights, // self.q_a_layernorm_weights[i]  [1536] 【enc】【dec】
    const paddle::Tensor& kv_a_proj_with_mqa_weights, // self.kv_a_proj_with_mqa_weights[i] [7168, 576] 【enc】
    const paddle::Tensor& kv_a_layernorm_weights, // self.kv_a_layernorm_weights[i] [512] 【enc】【dec】
    const paddle::Tensor& kv_b_proj_weights, // self.kv_b_proj_weights[i] [512, 32768] 【enc】
    const paddle::Tensor& linear_weights, // self.linear_weights[i] [16384, 7168] 【enc】【dec】
    const paddle::Tensor& k_b_proj_weights, // self.k_b_proj_weights[i] [128, 128, 512] 【dec】
    const paddle::Tensor& v_b_proj_weights, // self.v_b_proj_weights[i] [128, 512, 128] 【dec】
    const paddle::Tensor& cos_sin_cache, // self.config.rotary_emb.cos_sin_cache [163840, 64]
    const paddle::Tensor& position_ids, // self.position_ids [m]
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_seq_lod, 
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& start_token_raw,
    const paddle::Tensor& decoder_context_len_cache,
    const paddle::Tensor& decoder_context_len,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& kv_seq_lod_raw,
    const paddle::Tensor& encoder_seq_lod_cpu, 
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& start_token_raw_cpu,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& kv_seq_lod_raw_cpu,
    const paddle::Tensor& enc_batch_tensor,
    const paddle::Tensor& dec_batch_tensor,
    const paddle::Tensor& total_enc_len_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& max_enc_len_this_time,
    const paddle::Tensor& max_dec_len_this_time,
    const std::string& cache_quant_type_str,
    const float softmax_scale,
    const int block_size,
    const int num_heads, // self.num_heads
    const int q_lora_rank, // 传 self.config.mla_config.q_lora_rank = 1536
    const int qk_head_dim, // self.config.mla_config.qk_head_dim 192
    const int qk_nope_head_dim, // self.config.mla_config.qk_nope_head_dim 128
    const int qk_rope_head_dim, // self.config.mla_config.qk_rope_head_dim 64
    const int kv_lora_rank, // self.config.mla_config.kv_lora_rank 512
    const int v_head_dim // self.config.mla_config.v_head_dim 128
    ) { 

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xpu::ctx_guard RAII_GUARD(xpu_ctx->x_context());
  using CType = typename XPUTypeTrait<bfloat16>::Type;
  using CacheType = typename XPUTypeTrait<bfloat16>::Type;
  typedef paddle::bfloat16 data_t, cache_t;

  const int bsz = encoder_batch_map.dims()[0]; // batch map tensor长度都是max_batch_size, 有效长度根据
  // const int block_batch = block_tables.dims()[0]; // TODO参数含义 block_batch_  PageParam page_param_
  const int max_block_per_seq = block_tables.dims()[1];
  const int max_seq_len = block_size * max_block_per_seq;
  const int token_num = attn_input.dims()[0];
  const int hidden_dim = attn_input.dims()[1];
  int enc_batch = enc_batch_tensor.data<int32_t>()[0]; 
  int dec_batch = dec_batch_tensor.data<int32_t>()[0];
  int total_enc_len = total_enc_len_tensor.data<int32_t>()[0];
   
  // 输出tensor
  auto block_attn_out = paddle::full({token_num, hidden_dim}, -1, attn_input.type(), attn_input.place()); 

  // encoder
  if(enc_batch > 0){
    // split and copy
    auto ln_out_encoder = paddle::experimental::slice(attn_input, {0}, {0}, {total_enc_len}, {0}, {});
    auto position_ids_enc = paddle::experimental::slice(position_ids, {0}, {0}, {total_enc_len}, {0}, {});
    // encoder no absorb fc
    if (q_lora_rank <= 0) {
        assert("Not supported in block attn");
    }
    auto query_0 = paddle::matmul(ln_out_encoder, q_a_proj_weights);
    auto query_1 = std::get<0>(paddle::experimental::rms_norm(query_0, NULL, NULL, q_a_layernorm_weights, NULL, 1e-06, 1, -1, 0, 0, 0));
    auto query_2 = paddle::matmul(query_1, q_b_proj_weights);

    // auto query_nope = paddle::full({total_enc_len, num_heads, qk_nope_head_dim}, -1, query_2.type(), query_2.place());
    auto query_pe = paddle::full({total_enc_len, num_heads, qk_rope_head_dim}, -1, query_2.type(), query_2.place()); 
    int r = api::split<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(query_2.data<data_t>()),
      {nullptr, reinterpret_cast<CType*>(query_pe.data<data_t>())},
      {total_enc_len, num_heads, qk_head_dim},
      {qk_nope_head_dim, qk_rope_head_dim},
      2); // axis = 2
    PD_CHECK(r == 0, "api::split failed.");

    auto compressed_kv = paddle::matmul(ln_out_encoder, kv_a_proj_with_mqa_weights);

    auto compressed_kv_0 = paddle::full({total_enc_len, kv_lora_rank}, -1, compressed_kv.type(), compressed_kv.place()); 
    auto key_pe = paddle::full({total_enc_len, qk_rope_head_dim}, -1, compressed_kv.type(), compressed_kv.place()); 
    r = api::split<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(compressed_kv.data<data_t>()),
      {reinterpret_cast<CType*>(compressed_kv_0.data<data_t>()), reinterpret_cast<CType*>(key_pe.data<data_t>())},
      {total_enc_len, kv_lora_rank + qk_rope_head_dim},
      {kv_lora_rank, qk_rope_head_dim},
      1); // axis = 1
    PD_CHECK(r == 0, "api::split failed.");

    auto compressed_kv_1 = std::get<0>(paddle::experimental::rms_norm(compressed_kv_0, NULL, NULL, kv_a_layernorm_weights, NULL, 1e-06, 1, -1, 0, 0, 0));
    FusedRotaryPositionEncoding(query_pe, key_pe, position_ids_enc, cos_sin_cache, qk_rope_head_dim, false);
    PrefillMLAWriteCacheKernel(
      compressed_kv_1,
      key_pe,
      kv_cache,
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
      cache_quant_type_str,
      max_seq_len,
      num_heads);

    auto key_value = paddle::matmul(compressed_kv_1, kv_b_proj_weights);

    auto key_nope = paddle::full({total_enc_len, num_heads, qk_nope_head_dim}, -1, key_value.type(), key_value.place()); 
    auto value = paddle::full({total_enc_len, num_heads, v_head_dim}, -1, key_value.type(), key_value.place()); 
    r = api::split<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(key_value.data<data_t>()),
      {reinterpret_cast<CType*>(key_nope.data<data_t>()), reinterpret_cast<CType*>(value.data<data_t>())},
      {total_enc_len, num_heads, qk_nope_head_dim + v_head_dim},
      {qk_nope_head_dim, v_head_dim},
      2); // axis = 2
    PD_CHECK(r == 0, "api::split failed.");

    r = api::strided_slice_view_update<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<CType*>(query_pe.data<data_t>()),
      reinterpret_cast<CType*>(query_2.data<data_t>()),
      {total_enc_len, num_heads, qk_rope_head_dim},
      {total_enc_len, num_heads, qk_head_dim},
      {0, 0, qk_nope_head_dim}, // start
      {total_enc_len, num_heads, qk_head_dim}, // end
      {1, 1, 1}); // stride
    PD_CHECK(r == 0, "api::strided_slice_view_update failed.");

    auto key = paddle::full({total_enc_len, num_heads, qk_head_dim}, -1, query_2.type(), query_2.place());
    r = api::strided_slice_view_update<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<CType*>(key_nope.data<data_t>()),
      reinterpret_cast<CType*>(key.data<data_t>()),
      {total_enc_len, num_heads, qk_nope_head_dim},
      {total_enc_len, num_heads, qk_head_dim}, 
      {0, 0, 0}, // start
      {total_enc_len, num_heads, qk_nope_head_dim}, // end
      {1, 1, 1}); // stride
    PD_CHECK(r == 0, "api::strided_slice_view_update failed.");

    auto key_pe_broadcast = paddle::full({total_enc_len, num_heads, qk_rope_head_dim}, -1, query_2.type(), query_2.place());
    r = api::broadcast<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<CType*>(key_pe.data<data_t>()),
      reinterpret_cast<CType*>(key_pe_broadcast.data<data_t>()),
      {total_enc_len, 1, qk_rope_head_dim},
      {total_enc_len, num_heads, qk_rope_head_dim});
    PD_CHECK(r == 0, "api::broadcast failed.");

    r = api::strided_slice_view_update<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<CType*>(key_pe_broadcast.data<data_t>()),
      reinterpret_cast<CType*>(key.data<data_t>()),
      {total_enc_len, num_heads, qk_rope_head_dim},
      {total_enc_len, num_heads, qk_head_dim},
      {0, 0, qk_nope_head_dim}, // start
      {total_enc_len, num_heads, qk_head_dim}, // end
      {1, 1, 1}); // stride
    PD_CHECK(r == 0, "api::strided_slice_view_update failed.");

    // attn
    auto fmha_out_prefill = MlaEnAttn(
      query_2,
      key,
      value,
      encoder_seq_lod,
      encoder_batch_map,
      encoder_seq_lod_cpu,
      encoder_batch_map_cpu,
      enc_batch_tensor,
      padding_offsets,
      cum_offsets,
      block_tables,
      max_enc_len_this_time,
      max_dec_len_this_time,
      softmax_scale,
      block_size,
      num_heads,
      qk_head_dim,
      v_head_dim)[0];    

    auto out_linear_out_prefill = paddle::matmul(fmha_out_prefill, linear_weights);

    r = api::copy<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(out_linear_out_prefill.data<data_t>()),
      reinterpret_cast<CType*>(block_attn_out.data<data_t>()),
      {total_enc_len * hidden_dim});
    PD_CHECK(r == 0, "api::copy failed.");

  }

  // decoder
  if(dec_batch > 0){
    // split
    auto ln_out_decoder = paddle::experimental::slice(attn_input, {0}, {total_enc_len}, {token_num}, {0}, {});
    auto position_ids_dec = paddle::experimental::slice(position_ids, {0}, {total_enc_len}, {token_num}, {0}, {});
    // decoder absorb fc
    auto compressed_kv = paddle::matmul(ln_out_decoder, kv_a_proj_with_mqa_weights);

    auto compressed_kv_0 = paddle::full({dec_batch, kv_lora_rank}, -1, compressed_kv.type(), compressed_kv.place()); 
    auto key_pe = paddle::full({dec_batch, qk_rope_head_dim}, -1, compressed_kv.type(), compressed_kv.place()); 
    int r = api::split<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(compressed_kv.data<data_t>()),
      {reinterpret_cast<CType*>(compressed_kv_0.data<data_t>()), reinterpret_cast<CType*>(key_pe.data<data_t>())},
      {dec_batch, kv_lora_rank + qk_rope_head_dim},
      {kv_lora_rank, qk_rope_head_dim},
      1); // axis = 1
    PD_CHECK(r == 0, "api::split failed.");

    auto compressed_kv_1 = std::get<0>(paddle::experimental::rms_norm(compressed_kv_0, NULL, NULL, kv_a_layernorm_weights, NULL, 1e-06, 1, -1, 0, 0, 0)); // TODO inplace

    if (q_lora_rank <= 0) {
        assert("Not supported in block attn");
    }
    auto query_0 = paddle::matmul(ln_out_decoder, q_a_proj_weights);
    auto query_1 = std::get<0>(paddle::experimental::rms_norm(query_0, NULL, NULL, q_a_layernorm_weights, NULL, 1e-06, 1, -1, 0, 0, 0));
    auto query_2 = paddle::matmul(query_1, q_b_proj_weights);

    auto query_nope = paddle::full({dec_batch, num_heads, qk_nope_head_dim}, -1, query_2.type(), query_2.place()); 
    auto query_pe = paddle::full({dec_batch, num_heads, qk_rope_head_dim}, -1, query_2.type(), query_2.place()); 
    r = api::split<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(query_2.data<data_t>()),
      {reinterpret_cast<CType*>(query_nope.data<data_t>()), reinterpret_cast<CType*>(query_pe.data<data_t>())},
      {dec_batch, num_heads, qk_head_dim},
      {qk_nope_head_dim, qk_rope_head_dim},
      2); // axis = 2
    PD_CHECK(r == 0, "api::split failed.");

    auto query_nope_trans = paddle::full({num_heads, dec_batch, qk_nope_head_dim}, -1, query_2.type(), query_2.place()); 
    r = api::transpose<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(query_nope.data<data_t>()),
      reinterpret_cast<CType*>(query_nope_trans.data<data_t>()),
      {dec_batch, num_heads, qk_nope_head_dim},
      {1, 0, 2});
    PD_CHECK(r == 0, "api::transpose failed.");

    auto q_nope_bmm = Bmm(query_nope_trans, k_b_proj_weights)[0];

    auto q_nope_out = paddle::full({dec_batch, num_heads, kv_lora_rank}, -1, query_2.type(), query_2.place());
    r = api::transpose<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(q_nope_bmm.data<data_t>()),
      reinterpret_cast<CType*>(q_nope_out.data<data_t>()),
      {num_heads, dec_batch, kv_lora_rank},
      {1, 0, 2});
    PD_CHECK(r == 0, "api::transpose failed.");

    FusedRotaryPositionEncoding(query_pe, key_pe, position_ids_dec, cos_sin_cache, qk_rope_head_dim, false);
    DecodeMLAWriteCacheKernel(
      compressed_kv_1,
      key_pe,
      kv_cache,
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
      cache_quant_type_str,
      max_seq_len,
      num_heads,
      false); // self.config.speculate_config.speculate_method is not None

    auto q_input = paddle::full({dec_batch, num_heads * (kv_lora_rank + qk_rope_head_dim)}, -1, query_2.type(), query_2.place()); // TODO inplace
    r = api::concat<CType>(
      xpu_ctx->x_context(),
      {reinterpret_cast<const CType*>(q_nope_out.data<data_t>()), reinterpret_cast<const CType*>(query_pe.data<data_t>())},
      reinterpret_cast<CType*>(q_input.data<data_t>()),
      {{dec_batch, num_heads, kv_lora_rank}, {dec_batch, num_heads, qk_rope_head_dim}},
      2); // axis = 2

    // attn 
    auto fmha_out_decode = MlaDeAttn( // [dec_batch, 128*512]
      q_input,
      kv_cache,
      decoder_context_len,
      decoder_batch_map,
      decoder_context_len_cpu,
      decoder_batch_map_cpu,
      dec_batch_tensor,
      padding_offsets,
      cum_offsets,
      block_tables,
      softmax_scale,
      block_size,
      num_heads,
      kv_lora_rank,
      qk_rope_head_dim,
      qk_head_dim,
      v_head_dim)[0];    
  
    auto fmha_out_decode_trans = paddle::full({num_heads, dec_batch, kv_lora_rank}, -1, query_2.type(), query_2.place()); // shape要跟transpose后的对齐，下面的BMM会读取shape做计算
    r = api::transpose<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(fmha_out_decode.data<data_t>()),
      reinterpret_cast<CType*>(fmha_out_decode_trans.data<data_t>()),
      {dec_batch, num_heads, kv_lora_rank},
      {1, 0, 2});
    PD_CHECK(r == 0, "api::transpose failed.");

    auto fmha_out_decode_bmm = Bmm(fmha_out_decode_trans, v_b_proj_weights)[0]; // fmha_out_decode_bmm = [dec_batch, 128*128]

    auto fmha_out_decode_out = paddle::full({dec_batch, num_heads * v_head_dim}, -1, query_2.type(), query_2.place());
    r = api::transpose<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(fmha_out_decode_bmm.data<data_t>()),
      reinterpret_cast<CType*>(fmha_out_decode_out.data<data_t>()),
      {num_heads, dec_batch, v_head_dim},
      {1, 0, 2});
    PD_CHECK(r == 0, "api::transpose failed.");

    auto out_linear_out_decode = paddle::matmul(fmha_out_decode_out, linear_weights);

    r = api::copy<CType>(
      xpu_ctx->x_context(),
      reinterpret_cast<const CType*>(out_linear_out_decode.data<data_t>()),
      reinterpret_cast<CType*>(block_attn_out.data<data_t>()) + total_enc_len * hidden_dim,
      {dec_batch * hidden_dim});
    PD_CHECK(r == 0, "api::copy failed.");
  }
  return {block_attn_out};
    
}


std::vector<std::vector<int64_t>> MlaBlockAttnInferShape(const std::vector<int64_t>& attn_input_shape) {  
  return {attn_input_shape};
}

std::vector<paddle::DataType> MlaBlockAttnInferDtype(const paddle::DataType& attn_input_dtype) {  
  return {attn_input_dtype};
}

PD_BUILD_OP(absorb_mla_block_attention_xpu)
    .Inputs({"attn_input", 
             "q_a_proj_weights", 
             "q_b_proj_weights", 
             "q_a_layernorm_weights", 
             "kv_a_proj_with_mqa_weights", 
             "kv_a_layernorm_weights", 
             "kv_b_proj_weights", 
             "linear_weights", 
             "k_b_proj_weights", 
             "v_b_proj_weights", 
             "cos_sin_cache",  
             "position_ids", 
             "kv_cache",
             "block_tables",
             "encoder_seq_lod", 
             "encoder_batch_map",
             "start_token_raw",
             "decoder_context_len_cache",
             "decoder_context_len",
             "decoder_batch_map",
             "kv_seq_lod_raw",
             "encoder_seq_lod_cpu", 
             "encoder_batch_map_cpu",
             "start_token_raw_cpu",
             "decoder_context_len_cache_cpu",
             "decoder_context_len_cpu",
             "decoder_batch_map_cpu",
             "kv_seq_lod_raw_cpu",
             "enc_batch_tensor",
             "dec_batch_tensor",
             "total_enc_len_tensor",
             "padding_offsets",
             "cum_offsets",
             "max_enc_len_this_time",
             "max_dec_len_this_time",})
    .Outputs({"block_attn_out"})
    // .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string",
            "softmax_scale: float",
            "block_size: int",
            "num_heads: int",
            "q_lora_rank: int",
            "qk_head_dim: int",
            "qk_nope_head_dim: int",
            "qk_rope_head_dim: int",
            "kv_lora_rank: int",
            "v_head_dim: int"})
    .SetKernelFn(PD_KERNEL(MlaBlockAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(MlaBlockAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MlaBlockAttnInferDtype));