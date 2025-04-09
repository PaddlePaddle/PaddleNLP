#pragma once
#include "paddle/extension.h"

std::vector<paddle::Tensor> per_warp_int8_cuda(paddle::Tensor& q,
                                            paddle::Tensor& k,
                                            paddle::Tensor& km,
                                            int BLKQ,
                                            int WARPQ,
                                            int BLKK,
                                            int tensor_layout);
                    
std::vector<paddle::Tensor> per_channel_fp8(paddle::Tensor& v,
                                            int tensor_layout,
                                            float scale_max,
                                            bool smooth_v);

std::vector<paddle::Tensor> sub_mean(paddle::Tensor& v,
                                    paddle::Tensor& vm,
                                    int tensor_layout);

std::vector<paddle::Tensor> sage_attention_fwd(paddle::Tensor& q,
                                               paddle::Tensor& k,
                                               paddle::Tensor& v,
                                               paddle::Tensor& km,
                                               const paddle::Tensor& seq_len_this_time,
                                               const paddle::optional<paddle::Tensor>& vm,
                                               float sm_scale,
                                               std::string qk_quant_gran,
                                               std::string pv_accum_dtype,
                                               int tensor_layout,
                                               bool is_causal,
                                               bool smooth_k,
                                               bool smooth_v,
                                               bool return_lse);
