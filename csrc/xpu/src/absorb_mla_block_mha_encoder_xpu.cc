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
std::vector<paddle::Tensor> MlaEnAttn(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    const paddle::Tensor& v,
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& enc_batch_tensor,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& max_enc_len_this_time,
    const paddle::Tensor& max_dec_len_this_time,
    const float softmax_scale,
    const int block_size,
    const int num_head,
    const int dim_qk,
    const int dim_v) {
        
  baidu::xpu::api::plugin::print_times("[TIME BEGIN] MlaEnAttn");
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
  int enc_batch = enc_batch_tensor.data<int32_t>()[0]; 
  // 初始化输入：q k v
  auto q_xft = baidu::xpu::xft::xftTensor<QType, 3>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(q.data<qdata_t>())),
      std::array<int64_t, 3>{q.shape()[0],
                             q.shape()[1],
                             q.shape()[2]});
  auto k_xft = baidu::xpu::xft::xftTensor<QType, 3>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(k.data<qdata_t>())),
      std::array<int64_t, 3>{k.shape()[0],
                             k.shape()[1],
                             k.shape()[2]});
  auto v_xft = baidu::xpu::xft::xftTensor<QType, 3>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(v.data<qdata_t>())),
      std::array<int64_t, 3>{v.shape()[0],
                             v.shape()[1],
                             v.shape()[2]});       
                                                   
  // 初始化输出tensor
  auto fmha_out = paddle::full({q.shape()[0], num_head * dim_v}, -2, q.type(), q.place()); 
  auto fmha_out_xft = baidu::xpu::xft::xftTensor<QType, 2>(
      reinterpret_cast<QType*>(const_cast<paddle::bfloat16*>(fmha_out.data<qdata_t>())),
      std::array<int64_t, 2>{fmha_out.shape()[0],
                             fmha_out.shape()[1]});

  // encoder
  if(max_enc_len_this_time.data<int>()[0] > 0){
    // q_lod
    baidu::xpu::api::VectorParam<int32_t> context_len_vp{const_cast<int32_t*>(encoder_seq_lod_cpu.data<int32_t>()), enc_batch + 1, const_cast<int32_t*>(encoder_seq_lod.data<int32_t>())};
    // real batch（encoder阶段 kv cache写需要）
    baidu::xpu::api::VectorParam<int32_t> valid_batch_vp{const_cast<int32_t*>(encoder_batch_map_cpu.data<int32_t>()), enc_batch, const_cast<int32_t*>(encoder_batch_map.data<int32_t>())};
    // prefix (not support)
    baidu::xpu::api::VectorParam<int32_t> prefix_lens_vp = baidu::xpu::api::VectorParam<int32_t>();
    // kv_lod (非prefix cache情况与q_lod一致)
    baidu::xpu::api::VectorParam<int32_t> encoder_kv_lods_vp{const_cast<int32_t*>(encoder_seq_lod_cpu.data<int32_t>()), enc_batch + 1, const_cast<int32_t*>(encoder_seq_lod.data<int32_t>())};
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
    // fmha op
    using FMHA_Type = typename baidu::xpu::xft::FMHA_QBF16_KVBF16;
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
            nullptr,
            softmax_scale * sqrt(dim_qk)); // block_tables (prefix cache)

    // std::cout << "fmha kernel done " <<std::endl;
  }
  baidu::xpu::api::plugin::print_times("[TIME END] MlaEnAttn:");
    return {fmha_out};   
}

std::vector<std::vector<int64_t>> MlaEnAttnInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& k_shape,
    const std::vector<int64_t>& v_shape,
    const std::vector<int64_t>& encoder_seq_lod_shape,
    const std::vector<int64_t>& encoder_batch_map_shape,
    const std::vector<int64_t>& encoder_seq_lod_cpu_shape,
    const std::vector<int64_t>& encoder_batch_map_cpu_shape,
    const std::vector<int64_t>& enc_batch_tensor_shape,
    const std::vector<int64_t>& padding_offsets_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& max_enc_len_this_time_shape,
    const std::vector<int64_t>& max_dec_len_this_time_shape) {

  return {v_shape};
}

std::vector<paddle::DataType> MlaEnAttnInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& k_dtype,
    const paddle::DataType& v_dtype,
    const paddle::DataType& encoder_seq_lod_dtype,
    const paddle::DataType& encoder_batch_map_dtype,
    const paddle::DataType& encoder_seq_lod_cpu_dtype,
    const paddle::DataType& encoder_batch_map_cpu_dtype,
    const paddle::DataType& enc_batch_tensor_dtype,
    const paddle::DataType& padding_offsets_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& max_enc_len_this_time_dtype,
    const paddle::DataType& max_dec_len_this_time_dtype) {
    if (q_dtype == paddle::DataType::FLOAT16) {
        return {paddle::DataType::FLOAT16};
    } else if(q_dtype == paddle::DataType::BFLOAT16){
        return {paddle::DataType::BFLOAT16};
    } 
    else {
    PD_THROW("Only supported attr of compute_dtype in ['fp16','bfp16'].");
    }
}

PD_BUILD_OP(absorb_mla_block_mha_encoder_xpu)
    .Inputs({"q",
             "k",
             "v",
             "encoder_seq_lod",
             "encoder_batch_map",
             "encoder_seq_lod_cpu",
             "encoder_batch_map_cpu",
             "enc_batch_tensor",
             "padding_offsets",
             "cum_offsets",
             "block_tables",
             "max_enc_len_this_time",
             "max_dec_len_this_time"})
    .Outputs({"fmha_out"})
    .Attrs({"softmax_scale: float",
            "block_size: int",
            "num_head: int",
            "dim_qk: int",
            "dim_v: int"})
    .SetKernelFn(PD_KERNEL(MlaEnAttn))
    .SetInferShapeFn(PD_INFER_SHAPE(MlaEnAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MlaEnAttnInferDtype));

