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
std::vector<paddle::Tensor> MlaAttn(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    const paddle::Tensor& v,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
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
  const auto& key_cache_dims = key_cache.dims();
  const auto& value_cache_dims = value_cache.dims();
  const int bsz = seq_lens_encoder.dims()[0];
  const int token_num = input_dims[0];
  const int num_head = key_cache_dims[1];
  const int dim_qk = key_cache_dims[3];
  const int dim_v = value_cache_dims[3];
  const int block_batch = block_tables.dims()[0]; // TODO参数含义 block_batch_  PageParam page_param_
  const int max_block_per_seq = block_tables.dims()[1];
  const int block_size = key_cache_dims[2];  
  const int max_seq_len = block_size * max_block_per_seq;
  // 初始化输入：q k v
  auto q_xft = baidu::xpu::xft::xftTensor<QType, 2>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(q.data<qdata_t>())),
      std::array<int64_t, 2>{q.shape()[0],
                             q.shape()[1]});
  auto k_xft = baidu::xpu::xft::xftTensor<QType, 2>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(k.data<qdata_t>())),
      std::array<int64_t, 2>{k.shape()[0],
                             k.shape()[1]});
  auto v_xft = baidu::xpu::xft::xftTensor<QType, 2>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(v.data<qdata_t>())),
      std::array<int64_t, 2>{v.shape()[0],
                             v.shape()[1]});                             
  // 初始化输入：k cache
  auto key_cache_xft = baidu::xpu::xft::xftTensor<CacheType, 4>(
  reinterpret_cast<CacheType*>(const_cast<paddle::bfloat16*>(key_cache.data<cache_t>())),
  std::array<int64_t, 4>{key_cache.shape()[0],
                          key_cache.shape()[1],
                          key_cache.shape()[2],
                          key_cache.shape()[3]});   
  // 初始化输入：v cache                                    
  auto value_cache_xft = baidu::xpu::xft::xftTensor<CacheType, 4>(
  reinterpret_cast<CacheType*>(const_cast<paddle::bfloat16*>(value_cache.data<cache_t>())),
  std::array<int64_t, 4>{value_cache.shape()[0],
                          value_cache.shape()[1],
                          value_cache.shape()[2],
                          value_cache.shape()[3]}); 
  // 初始化输入：block table
  auto block_tables_xft = baidu::xpu::xft::xftTensor<int, 2>(
  reinterpret_cast<int*>(const_cast<int*>(block_tables.data<int>())),
  std::array<int64_t, 2>{block_tables.shape()[0],
                          block_tables.shape()[1]}); 
  // 初始化输出tensor

  auto fmha_out = paddle::full({q.shape()[0], num_head * dim_v}, -2, q.type(), q.place()); 
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
  std::vector<int> decoder_context_len_cache;
  xpu_wait(xpu_ctx->x_context()->xpu_stream); // 是否需要！！！！TODO
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
      decoder_context_len_cache.push_back(seq_lens_decoder_cpu[i]);
    }
    else{
        batch_offset++;
    }
  }

  // encoder
  if(max_enc_len_this_time.data<int>()[0] > 0){
    // q_lod
    baidu::xpu::api::VectorParam<int32_t> context_len_vp =
        baidu::xpu::api::VectorParam<int32_t>{encoder_seq_lod.data(), enc_batch + 1, nullptr}
            .to_xpu(RAII_GUARD);
    // real batch（encoder阶段 kv cache写需要）
    baidu::xpu::api::VectorParam<int32_t> valid_batch_vp =
        baidu::xpu::api::VectorParam<int32_t>{encoder_batch_map.data(), enc_batch, nullptr}
            .to_xpu(RAII_GUARD);
    // prefix (not support)
    baidu::xpu::api::VectorParam<int32_t> prefix_lens_vp = baidu::xpu::api::VectorParam<int32_t>();
    // kv_lod (非prefix cache情况与q_lod一致)
    baidu::xpu::api::VectorParam<int32_t> encoder_kv_lods_vp =
        baidu::xpu::api::VectorParam<int32_t>{encoder_seq_lod.data(), enc_batch + 1, nullptr}
            .to_xpu(RAII_GUARD);  
    // page_param_
    baidu::xpu::xft::PageParam page_param_(block_batch, // block_batch_
                                            max_enc_len_this_time.data<int>()[0], // max_enc_len_this_time.data<int>()[0],max_seq_len // max_context_len_
                                            block_size, // block_size_
                                            max_block_per_seq, // max_num_blocks_per_seq_
                                            false); // v_trans  
    // attn_param
    baidu::xpu::xft::PageAttnParam attn_param_(dim_qk, // head_dim(q,k)
                                              num_head, // q_head_num
                                              num_head, // kv_head_num
                                              false, // vp_lod_flag
                                              context_len_vp, // context_len_vp
                                              valid_batch_vp, // valid_batch_vp
                                              prefix_lens_vp, // prefix_lens_vp
                                              encoder_kv_lods_vp); // encoder_kv_lods_vp
    
    // k v cache write
    std::vector<int> start_tokens(enc_batch, 0);
    baidu::xpu::api::VectorParam<int32_t> start_tokens_vp =
        baidu::xpu::api::VectorParam<int32_t>{start_tokens.data(), enc_batch, nullptr}.to_xpu(RAII_GUARD);
    int ret_cache = xftkernel::xft_reshape_cached_kv<float16, float16, int>(
            xpu_ctx->x_context(),
            reinterpret_cast<float16*>(k_xft.data()),
            reinterpret_cast<float16*>(key_cache_xft.data()),
            block_tables_xft.data(),
            context_len_vp,
            start_tokens_vp,
            valid_batch_vp,
            enc_batch,
            num_head,
            dim_qk,
            block_batch,
            block_size,
            max_block_per_seq,
            "BLHD", // qkv_layout,
            "HLD", // page_layout,
            nullptr, // scale,
            nullptr, // batch_max_ptrs, 
            nullptr); // max_ptrs
    ret_cache = xftkernel::xft_reshape_cached_kv<float16, float16, int>(
            xpu_ctx->x_context(),
            reinterpret_cast<float16*>(v_xft.data()),
            reinterpret_cast<float16*>(value_cache_xft.data()),
            block_tables_xft.data(),
            context_len_vp,
            start_tokens_vp,
            valid_batch_vp,
            enc_batch,
            num_head,
            dim_v,
            block_batch,
            block_size,
            max_block_per_seq,
            "BLHD", // qkv_layout,
            "HLD", // page_layout,
            nullptr, // scale,
            nullptr, // batch_max_ptrs, 
            nullptr); // max_ptrs
    // fmha op
    using FMHA_Type = typename baidu::xpu::xft::FMHA_QBF16_KVBF16;
    // auto fmha_op = baidu::xpu::xft::FMHAOperation<FMHA_Type>(
    //         enc_batch,
    //         attn_param_.head_dim_,
    //         attn_param_.q_head_num_,
    //         attn_param_.kv_head_num_,
    //         page_param_.max_context_len_,  
    //         page_param_.max_context_len_,
    //         attn_param_.context_len_vp_,
    //         attn_param_.encoder_kv_lods_vp_);

    using TQ = typename FMHA_Type::Qtype;
    using TK = typename FMHA_Type::Ktype;
    using TV = typename FMHA_Type::Vtype;
    using TBIAS = typename FMHA_Type::Btype;
    using TO = typename FMHA_Type::Otype;
    using TMASK = typename FMHA_Type::Mtype;
    int ret = baidu::xpu::xfa::flash_attention_context_vllm<
            TQ,
            TK,
            TV,
            TBIAS,
            TO,
            float, // T_GEMM0
            float, // T_GEMM1
            float, // T_EW
            int,
            0>(
            xpu_ctx->x_context(),
            q_xft.data(), // q
            std::is_same<TK, int8_t>::value ? nullptr : reinterpret_cast<TK*>(k_xft.data()), // k
            std::is_same<TK, int8_t>::value ? nullptr : reinterpret_cast<TV*>(v_xft.data()), // v
            nullptr, // p_bias == nullptr ? nullptr : p_bias->data(), // mask_bias
            fmha_out_xft.data(), //o
            enc_batch, // batch_size
            page_param_.max_context_len_,      // max_seq_q
            page_param_.max_context_len_,  // max_seq_kv, （prefix cache是会不一样）
            attn_param_.q_head_num_,
            attn_param_.head_dim_,
            attn_param_.kv_head_num_,
            nullptr,
            attn_param_.context_len_vp_,
            attn_param_.encoder_kv_lods_vp_,
            {}, // mask_shape_
            {}, // alibi_slopes_shape
            true, // is_causal_mask_
            false, // is_qkv_fusion_
            0x0010,  //     param.qkv_layout_ = AttnQKVLayout_t::ATTN_BLHD
            0x10, // vsl_flag_ = VslConverter::FMHA_LVSL
            nullptr,  // q_maxptr
            nullptr,  // k_maxptr
            nullptr,  // v_maxptr
            nullptr,
            dim_v, // vo_head_dim
            nullptr, // p_cache_k->data(),
            nullptr, // p_cache_v->data(),
            nullptr, // std::is_same<TK, int8_t>::value ? p_kcache_perhead_scale->data() : p_cache_k->max_data(),
            nullptr, // std::is_same<TK, int8_t>::value ? p_vcache_perhead_scale->data() : p_cache_v->max_data(),
            block_size, // block_size
            max_block_per_seq, // max_blocks_per_seq (prefix cache)
            page_param_.max_context_len_, // prefill_len
            nullptr); // block_tables (prefix cache)

    // std::cout << "fmha kernel done " <<std::endl;
  }

  // decoder
  if(max_dec_len_this_time.data<int>()[0] > 0){
    // context_len
    baidu::xpu::api::VectorParam<int32_t> context_len_vp =
        baidu::xpu::api::VectorParam<int32_t>{decoder_context_len.data(), dec_batch, nullptr}
            .to_xpu(RAII_GUARD);
    baidu::xpu::api::VectorParam<int32_t> context_len_vp_cache =
        baidu::xpu::api::VectorParam<int32_t>{decoder_context_len_cache.data(), dec_batch, nullptr}
            .to_xpu(RAII_GUARD);
    // real batch     
    baidu::xpu::api::VectorParam<int32_t> valid_batch_vp =
        baidu::xpu::api::VectorParam<int32_t>{decoder_batch_map.data(), dec_batch, nullptr}
            .to_xpu(RAII_GUARD);
    // prefix (not support)
    baidu::xpu::api::VectorParam<int32_t> prefix_lens_vp = baidu::xpu::api::VectorParam<int32_t>();
    // kv_lod (decoder 不需要)
    baidu::xpu::api::VectorParam<int32_t> encoder_kv_lods_vp = baidu::xpu::api::VectorParam<int32_t>();
    // page_param_
    baidu::xpu::xft::PageParam page_param_(block_batch, // block_batch_
                                            max_seq_len, // max_dec_len_this_time.data<int>()[0],max_seq_len // max_context_len_ // max_dec_len_this_time.data<int>()[0]
                                            block_size, // block_size_
                                            max_block_per_seq, // max_num_blocks_per_seq_
                                            false); // v_trans     
    // attn_param
    baidu::xpu::xft::PageAttnParam attn_param_(dim_qk, // head_dim(q,k)
                                              num_head, // q_head_num
                                              num_head, // kv_head_num
                                              false, // vp_lod_flag
                                              context_len_vp, // context_len_vp
                                              valid_batch_vp, // valid_batch_vp
                                              prefix_lens_vp, // prefix_lens_vp
                                              encoder_kv_lods_vp); // encoder_kv_lods_vp
    // k v cache write
    std::vector<int> kv_seq_lod(dec_batch + 1);
    std::iota(kv_seq_lod.begin(), kv_seq_lod.end(), 0);
    baidu::xpu::api::VectorParam<int32_t> kv_seq_lod_vp =
        baidu::xpu::api::VectorParam<int32_t>{kv_seq_lod.data(), dec_batch + 1, nullptr}.to_xpu(RAII_GUARD);
    int ret_cache = xftkernel::xft_reshape_cached_kv<float16, float16, int>(
            xpu_ctx->x_context(),
            reinterpret_cast<float16*>(k_xft.data() + total_enc_len * dim_qk * num_head),
            reinterpret_cast<float16*>(key_cache_xft.data()),
            block_tables_xft.data(),
            kv_seq_lod_vp,
            context_len_vp_cache,
            valid_batch_vp,
            dec_batch,
            num_head,
            dim_qk,
            block_batch,
            block_size,
            max_block_per_seq,
            "BLHD", // qkv_layout,
            "HLD", // page_layout,
            nullptr, // scale,
            nullptr, // batch_max_ptrs, 
            nullptr); // max_ptrs
    ret_cache = xftkernel::xft_reshape_cached_kv<float16, float16, int>(
            xpu_ctx->x_context(),
            reinterpret_cast<float16*>(v_xft.data() + total_enc_len * dim_v * num_head),
            reinterpret_cast<float16*>(value_cache_xft.data()),
            block_tables_xft.data(),
            kv_seq_lod_vp,
            context_len_vp_cache,
            valid_batch_vp,
            dec_batch,
            num_head,
            dim_v,
            block_batch,
            block_size,
            max_block_per_seq,
            "BLHD", // qkv_layout,
            "HLD", // page_layout,
            nullptr, // scale,
            nullptr, // batch_max_ptrs, 
            nullptr); // max_ptrs
    // paged attention op
    using PA_Type = typename baidu::xpu::xft::PA_QBF16_KVBF16;
    using TQ = typename PA_Type::Qtype; // bfloat16
    using TKV = typename PA_Type::KVtype; // bfloat16
    using TO = typename PA_Type::Otype; // bfloat16
    using TID = typename PA_Type::TIDtype; // int
    using TL = float;
    using TGEMM0 = float;
    using TGEMM1 = float;
    using TEW = float;

    auto pa_func = &baidu::xpu::xfa::paged_attention_xft<TQ, TKV, TO, TL, TGEMM0, TGEMM1, TEW, TID, false>;

    int ret = pa_func(
            xpu_ctx->x_context(),
            fmha_out_xft.data() + total_enc_len * dim_v * num_head,
            q_xft.data() + total_enc_len * dim_qk * num_head,
            nullptr,   /* k_cur */
            nullptr, /* v_cur */
            const_cast<TKV*>(key_cache_xft.data()),
            const_cast<TKV*>(value_cache_xft.data()),
            attn_param_.kv_head_num_,
            attn_param_.scale_,
            block_tables_xft.data(),
            attn_param_.context_len_vp_,
            attn_param_.valid_batch_vp_,
            page_param_.block_size_,
            page_param_.max_context_len_,
            nullptr, // (TBIAS*)alibi_slopes,
            block_batch, // page_param_.block_batch_,
            attn_param_.q_head_num_,
            attn_param_.head_dim_,
            page_param_.max_num_blocks_per_seq_,
            nullptr, // shift,
            nullptr, // smooth,
            nullptr, // query_maxptr,
            nullptr, // key_cache_maxptr,
            nullptr, // value_cache_maxptr,
            nullptr, // p_k_scales_inv,
            nullptr, // p_k_zeros,
            nullptr, // p_v_scales_inv,
            nullptr, // p_v_zeros,
            nullptr, // out_maxptr,
            dim_v); // v_head_dim
  }
    return {fmha_out};   
}

std::vector<std::vector<int64_t>> MlaAttnInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& k_shape,
    const std::vector<int64_t>& v_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
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
    const paddle::optional<std::vector<int64_t>>& out_linear_smooths_shape) {
  const int token_num = q_shape[0];
  const int all_v_dim = value_cache_shape[3];
  return {{token_num, all_v_dim}};
}

std::vector<paddle::DataType> MlaAttnInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& k_dtype,
    const paddle::DataType& v_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
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

PD_BUILD_OP(mla_block_multihead_attention_xpu)
    .Inputs({"q",
             "k",
             "v",
             "key_cache",
             "value_cache",
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
    // .SetInplaceMap({{"key_cache", "key_cache_out"},
    //                 {"value_cache", "value_cache_out"}})
    .Attrs({"cache_quant_type: std::string",
            "use_neox_rotary_style: bool",
            "max_input_length: int",
            "softmax_scale: float",
            "quant_max_bound: float",
            "quant_min_bound: float",
            "out_linear_in_scale: float",
            "speculate_max_draft_token_num: int",
            "causal: bool",
            "speculate_decoder: bool"})
    .SetKernelFn(PD_KERNEL(MlaAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(MlaAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MlaAttnInferDtype));

