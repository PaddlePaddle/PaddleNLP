#pragma once
#include "paddle/extension.h"

std::vector<paddle::Tensor> per_warp_int8_varlen_cuda_fwd(paddle::Tensor& q,    // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& k,          // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& cu_seqlen_q,
                                                    paddle::Tensor& km,
                                                    int max_seq_len_q,
                                                    int max_seq_len_k,
                                                    int BLKQ,
                                                    int WARPQ,
                                                    int BLKK);

std::vector<paddle::Tensor> per_channel_varlen_fp8(paddle::Tensor& v,                 // total_seqlen x num_head x head_dim
                                                  paddle::Tensor& padded_cu_seqlen,   // padded
                                                  int max_seq_len_v,
                                                  int padded_total_seq_len,
                                                  int tensor_layout,
                                                  float scale_max,
                                                  bool smooth_v);

std::vector<paddle::Tensor> chunked_segment_mean_fwd(paddle::Tensor& input,         // [total_seqlen, num_head, head_dim]
                                                     paddle::Tensor& cu_seqlens,    // [batch_size + 1], prefix-sum array of sequence lengths
                                                     const int max_seqlen);

std::vector<paddle::Tensor> sage_attention_varlen_fwd(paddle::Tensor& q,          // total_seqlen x num_head x head_dim
                                                      paddle::Tensor& k,          // total_seqlen x num_head x head_dim
                                                      paddle::Tensor& v,          // total_seqlen x num_head x head_dim
                                                      paddle::Tensor& cu_seqlen_q,
                                                      paddle::Tensor& cu_seqlen_v_padded,
                                                      const paddle::Tensor& seq_lens_encoder, // length of each segment this time
                                                      paddle::Tensor& km,
                                                      paddle::optional<paddle::Tensor>& vm,
                                                      int max_seqlen_q,
                                                      int max_seqlen_k,
                                                      float sm_scale,
                                                      std::string qk_quant_gran,
                                                      std::string pv_accum_dtype,
                                                      int tensor_layout,
                                                      bool is_causal,
                                                      bool smooth_k,
                                                      bool smooth_v,
                                                      bool return_lse);