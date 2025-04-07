// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <numeric>
#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace xpu2 {
namespace plugin {

__attribute__((global)) void recover_block(int* recover_block_list,  // [bsz]
                                           int* recover_len,
                                           bool* stop_flags,
                                           int* seq_lens_this_time,
                                           const int* ori_seq_lens_encoder,
                                           int* seq_lens_encoder,
                                           const int* seq_lens_decoder,
                                           int* block_tables,
                                           int* free_list,
                                           int* free_list_len,
                                           int64_t* input_ids,
                                           const int64_t* pre_ids,
                                           const int64_t* step_idx,
                                           const int* encoder_block_lens,
                                           const int* used_list_len,
                                           const int64_t* next_tokens,
                                           const int64_t* first_token_ids,
                                           const int bsz,
                                           const int block_num_per_seq,
                                           const int length,
                                           const int pre_id_length);

}  // namespace plugin
}  // namespace xpu2

namespace xpu3 {
namespace plugin {

__attribute__((global)) void recover_block(int* recover_block_list,  // [bsz]
                                           int* recover_len,
                                           bool* stop_flags,
                                           int* seq_lens_this_time,
                                           const int* ori_seq_lens_encoder,
                                           int* seq_lens_encoder,
                                           const int* seq_lens_decoder,
                                           int* block_tables,
                                           int* free_list,
                                           int* free_list_len,
                                           int64_t* input_ids,
                                           const int64_t* pre_ids,
                                           const int64_t* step_idx,
                                           const int* encoder_block_lens,
                                           const int* used_list_len,
                                           const int64_t* next_tokens,
                                           const int64_t* first_token_ids,
                                           const int bsz,
                                           const int block_num_per_seq,
                                           const int length,
                                           const int pre_id_length);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       int* recover_block_list,  // [bsz]
                       int* recover_len,
                       bool* stop_flags,
                       int* seq_lens_this_time,
                       const int* ori_seq_lens_encoder,
                       int* seq_lens_encoder,
                       const int* seq_lens_decoder,
                       int* block_tables,
                       int* free_list,
                       int* free_list_len,
                       int64_t* input_ids,
                       const int64_t* pre_ids,
                       const int64_t* step_idx,
                       const int* encoder_block_lens,
                       const int* used_list_len,
                       const int64_t* next_tokens,
                       const int64_t* first_token_ids,
                       const int bsz,
                       const int block_num_per_seq,
                       const int length,
                       const int pre_id_length) {
  int ori_free_list_len;
  for (int bid = 0; bid < recover_len[0]; bid++) {
    const int recover_id = recover_block_list[bid];
    const int ori_seq_len_encoder = ori_seq_lens_encoder[recover_id];
    const int step_idx_now = step_idx[recover_id];
    const int seq_len = ori_seq_len_encoder + step_idx_now;
    const int encoder_block_len = encoder_block_lens[recover_id];
    const int decoder_used_len = used_list_len[recover_id];
    int* block_table_now = block_tables + recover_id * block_num_per_seq;
    int64_t* input_ids_now = input_ids + recover_id * length;
    const int64_t* pre_ids_now = pre_ids + recover_id * pre_id_length;

    seq_lens_this_time[recover_id] = seq_len;
    seq_lens_encoder[recover_id] = seq_len;
    stop_flags[recover_id] = false;
    input_ids_now[seq_len - 1] = next_tokens[recover_id];  // next tokens
    input_ids_now[0] = first_token_ids[recover_id];  // set first prompt token
    ori_free_list_len = free_list_len[0];
    free_list_len[0] -= decoder_used_len;

    // 恢复block table
    for (int i = 0; i < decoder_used_len; i++) {
      block_table_now[encoder_block_len + i] =
          free_list[ori_free_list_len - i - 1];
    }
    // 恢复input_ids
    for (int i = 0; i < step_idx_now - 1; i++) {
      input_ids_now[ori_seq_len_encoder + i] = pre_ids_now[i + 1];
    }
  }
  recover_len[0] = 0;
  return api::SUCCESS;
}

static int xpu2or3_wrapper(Context* ctx,
                        int* recover_block_list,  // [bsz]
                        int* recover_len,
                        bool* stop_flags,
                        int* seq_lens_this_time,
                        const int* ori_seq_lens_encoder,
                        int* seq_lens_encoder,
                        const int* seq_lens_decoder,
                        int* block_tables,
                        int* free_list,
                        int* free_list_len,
                        int64_t* input_ids,
                        const int64_t* pre_ids,
                        const int64_t* step_idx,
                        const int* encoder_block_lens,
                        const int* used_list_len,
                        const int64_t* next_tokens,
                        const int64_t* first_token_ids,
                        const int bsz,
                        const int block_num_per_seq,
                        const int length,
                        const int pre_id_length) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  bool is_xpu2 = ctx->dev().type() == api::kXPU2;
  auto recover_block_kernel = is_xpu2 ? xpu2::plugin::recover_block : xpu3::plugin::recover_block;
  recover_block_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      recover_block_list,  // [bsz]
      recover_len,
      stop_flags,
      seq_lens_this_time,
      ori_seq_lens_encoder,
      seq_lens_encoder,
      seq_lens_decoder,
      block_tables,
      free_list,
      free_list_len,
      reinterpret_cast<XPU_INT64*>(input_ids),
      reinterpret_cast<const XPU_INT64*>(pre_ids),
      reinterpret_cast<const XPU_INT64*>(step_idx),
      encoder_block_lens,
      used_list_len,
      reinterpret_cast<const XPU_INT64*>(next_tokens),
      reinterpret_cast<const XPU_INT64*>(first_token_ids),
      bsz,
      block_num_per_seq,
      length,
      pre_id_length);
  return api::SUCCESS;
}

int recover_block(Context* ctx,
                  int* recover_block_list,  // [bsz]
                  int* recover_len,
                  bool* stop_flags,
                  int* seq_lens_this_time,
                  const int* ori_seq_lens_encoder,
                  int* seq_lens_encoder,
                  const int* seq_lens_decoder,
                  int* block_tables,
                  int* free_list,
                  int* free_list_len,
                  int64_t* input_ids,
                  const int64_t* pre_ids,
                  const int64_t* step_idx,
                  const int* encoder_block_lens,
                  const int* used_list_len,
                  const int64_t* next_tokens,
                  const int64_t* first_token_ids,
                  const int bsz,
                  const int block_num_per_seq,
                  const int length,
                  const int pre_id_length) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "recover_block", float);
  WRAPPER_DUMP_PARAM6(ctx,
                      recover_block_list,
                      recover_len,
                      stop_flags,
                      seq_lens_this_time,
                      ori_seq_lens_encoder,
                      seq_lens_encoder);
  WRAPPER_DUMP_PARAM6(ctx,
                      seq_lens_decoder,
                      block_tables,
                      free_list,
                      free_list_len,
                      input_ids,
                      pre_ids);
  WRAPPER_DUMP_PARAM5(ctx,
                      step_idx,
                      encoder_block_lens,
                      used_list_len,
                      next_tokens,
                      first_token_ids);
  WRAPPER_DUMP_PARAM4(ctx, bsz, block_num_per_seq, length, pre_id_length);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       recover_block_list,  // [bsz]
                       recover_len,
                       stop_flags,
                       seq_lens_this_time,
                       ori_seq_lens_encoder,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       block_tables,
                       free_list,
                       free_list_len,
                       input_ids,
                       pre_ids,
                       step_idx,
                       encoder_block_lens,
                       used_list_len,
                       next_tokens,
                       first_token_ids,
                       bsz,
                       block_num_per_seq,
                       length,
                       pre_id_length);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                        recover_block_list,  // [bsz]
                        recover_len,
                        stop_flags,
                        seq_lens_this_time,
                        ori_seq_lens_encoder,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        block_tables,
                        free_list,
                        free_list_len,
                        input_ids,
                        pre_ids,
                        step_idx,
                        encoder_block_lens,
                        used_list_len,
                        next_tokens,
                        first_token_ids,
                        bsz,
                        block_num_per_seq,
                        length,
                        pre_id_length);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
