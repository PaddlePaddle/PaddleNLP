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
/*
 * copyright (C) 2022 KUNLUNXIN, Inc
 */

#pragma once
#include "xpu/xdnn.h"

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
DLL_EXPORT int set_stop_value_multi_ends(Context* ctx,
                                         bool* stop_flags,
                                         T* topk_ids,
                                         T* next_tokens,
                                         const T* end_ids,
                                         const int* seq_lens,
                                         const int bs,
                                         const int end_length,
                                         const bool beam_search);


DLL_EXPORT int set_value_by_flags_and_idx(Context* ctx,
                                          const bool* stop_flags,
                                          int64_t* pre_ids_all,
                                          const int64_t* input_ids,
                                          const int* seq_lens_encoder,
                                          const int* seq_lens_decoder,
                                          const int64_t* step_idx,
                                          int bs,
                                          int length,
                                          int length_input_ids);

template <typename T>
DLL_EXPORT int token_penalty_multi_scores(Context* ctx,
                                          const int64_t* pre_ids,
                                          T* logits,
                                          const T* penalty_scores,
                                          const T* frequency_scores,
                                          const T* presence_scores,
                                          const float* temperatures,
                                          const int64_t* cur_len,
                                          const int64_t* min_len,
                                          const int64_t* eos_token_id,
                                          const int64_t* bad_words,
                                          const int64_t bs,
                                          const int64_t length,
                                          const int64_t length_id,
                                          const int64_t end_length,
                                          const int64_t length_bad_words);

DLL_EXPORT int get_padding_offset(Context* ctx,
                                  int* padding_offset,
                                  int* cum_offsets_out,
                                  int* cu_seqlens_q,
                                  int* cu_seqlens_k,
                                  int64_t* x_remove_padding,
                                  const int64_t* input_ids,
                                  const int* cum_offsets,
                                  const int* seq_lens,
                                  const int max_seq_len,
                                  const int bs);

DLL_EXPORT int update_inputs(Context* ctx,
                             bool* not_need_stop,
                             int* seq_lens_this_time,
                             int* seq_lens_encoder,
                             int* seq_lens_decoder,
                             int64_t* input_ids,
                             const int64_t* stop_nums,
                             const bool* stop_flags,
                             const bool* is_block_step,
                             const int64_t* next_tokens,
                             const int bsz,
                             const int max_bsz,
                             const int input_ids_stride);

template <typename T>
DLL_EXPORT int rebuild_padding(Context *ctx,
                    T *output_data, // [bs, dim_embed]
                    const T *input_data, // [token_num, dim_embed]
                    const int *cum_offsets, // [bs]
                    const int *seq_len_decoder, // [bs]
                    const int *seq_len_encoder, // [bs]
                    const int seq_len,
                    const int dim_embed,
                    const int elem_nums);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
