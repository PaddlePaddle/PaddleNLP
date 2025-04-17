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

__attribute__((global)) void free_and_dispatch_block(
    bool *stop_flags,
    int *seq_lens_this_time,
    int *seq_lens_decoder,
    int *block_tables,
    int *encoder_block_lens,
    bool *is_block_step,
    int *step_block_list,  // [bsz]
    int *step_len,
    int *recover_block_list,
    int *recover_len,
    int *need_block_list,
    int *need_block_len,
    int *used_list_len,
    int *free_list,
    int *free_list_len,
    int64_t *first_token_ids,
    const int bsz,
    const int block_size,
    const int block_num_per_seq,
    const int max_decoder_block_num);

}  // namespace plugin
}  // namespace xpu2

namespace xpu3 {
namespace plugin {

__attribute__((global)) void free_and_dispatch_block(
    bool *stop_flags,
    int *seq_lens_this_time,
    int *seq_lens_decoder,
    int *block_tables,
    int *encoder_block_lens,
    bool *is_block_step,
    int *step_block_list,  // [bsz]
    int *step_len,
    int *recover_block_list,
    int *recover_len,
    int *need_block_list,
    int *need_block_len,
    int *used_list_len,
    int *free_list,
    int *free_list_len,
    int64_t *first_token_ids,
    const int bsz,
    const int block_size,
    const int block_num_per_seq,
    const int max_decoder_block_num);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context *ctx,
                       bool *stop_flags,
                       int *seq_lens_this_time,
                       int *seq_lens_decoder,
                       int *block_tables,
                       int *encoder_block_lens,
                       bool *is_block_step,
                       int *step_block_list,  // [bsz]
                       int *step_len,
                       int *recover_block_list,
                       int *recover_len,
                       int *need_block_list,
                       int *need_block_len,
                       int *used_list_len,
                       int *free_list,
                       int *free_list_len,
                       int64_t *first_token_ids,
                       const int bsz,
                       const int block_size,
                       const int block_num_per_seq,
                       const int max_decoder_block_num) {
  for (int i = 0; i < bsz; i++) {
    int *block_table_now = block_tables + i * block_num_per_seq;
    if (stop_flags[i] && !is_block_step[i]) {
      // 回收block块
      const int encoder_block_len = encoder_block_lens[i];
      const int decoder_used_len = used_list_len[i];
      if (decoder_used_len > 0) {
        const int ori_free_list_len = free_list_len[0];
        free_list_len[0] += decoder_used_len;
        for (int j = 0; j < decoder_used_len; j++) {
          free_list[ori_free_list_len + j] =
              block_table_now[encoder_block_len + j];
          block_table_now[encoder_block_len + j] = -1;
        }
        encoder_block_lens[i] = 0;
        used_list_len[i] = 0;
      }
    } else if (block_table_now[seq_lens_decoder[i] / block_size] == -1) {
      // 统计需要分配block的位置和总数
      const int ori_need_block_len = need_block_len[0];
      need_block_len[0] += 1;
      need_block_list[ori_need_block_len] = i;
    }
  }

  while (need_block_len[0] > free_list_len[0]) {
    // 调度block，根据used_list_len从大到小回收block，直到满足need_block_len
    int max_used_list_len_id = 0;
    int max_used_list_len = 0;
    for (int i = 0; i < bsz; i++) {
      const int used_block_num = !is_block_step[i] ? used_list_len[i] : 0;
      if (used_block_num > max_used_list_len) {
        max_used_list_len_id = i;
        max_used_list_len = used_block_num;
      }
    }

    const int encoder_block_len = encoder_block_lens[max_used_list_len_id];
    int *block_table_now =
        block_tables + max_used_list_len_id * block_num_per_seq;
    for (int i = 0; i < max_used_list_len; i++) {
      free_list[free_list_len[0] + i] = block_table_now[encoder_block_len + i];
      block_table_now[encoder_block_len + i] = -1;
    }
    step_block_list[step_len[0]] = max_used_list_len_id;
    step_len[0] += 1;
    free_list_len[0] += max_used_list_len;
    stop_flags[max_used_list_len_id] = true;
    is_block_step[max_used_list_len_id] = true;
    seq_lens_this_time[max_used_list_len_id] = 0;
    seq_lens_decoder[max_used_list_len_id] = 0;
  }

  // 为需要block的位置分配block，每个位置分配一个block
  for (int i = 0; i < bsz; i++) {
    if (i < need_block_len[0]) {
      const int need_block_id = need_block_list[i];
      if (!stop_flags[need_block_id]) {
        // 如果需要的位置正好是上一步中被释放的位置，不做处理
        used_list_len[need_block_id] += 1;
        const int ori_free_list_len = free_list_len[0];
        free_list_len[0]--;
        int *block_table_now = block_tables + need_block_id * block_num_per_seq;
        block_table_now[seq_lens_decoder[need_block_id] / block_size] =
            free_list[ori_free_list_len - 1];
      }
      need_block_list[i] = -1;
    }
  }

  // 计算可以复原的query id
  int ori_step_len = step_len[0];
  if (ori_step_len > 0) {
    int ori_free_list_len = free_list_len[0];
    int ori_step_block_id = step_block_list[ori_step_len - 1];
    int tmp_used_len = used_list_len[ori_step_block_id];
    // 比之前调度时多分配一个block，防止马上恢复刚调度的query(比如回收的seq_id在need_block_list中）
    int used_len =
        tmp_used_len < max_decoder_block_num ? tmp_used_len + 1 : tmp_used_len;
    while (ori_step_len > 0 && ori_free_list_len >= used_len) {
      recover_block_list[recover_len[0]] = ori_step_block_id;
      is_block_step[ori_step_block_id] = false;
      used_list_len[ori_step_block_id] = used_len;
      ori_free_list_len -= used_len;
      step_block_list[ori_step_len - 1] = -1;
      step_len[0] -= 1;
      recover_len[0] += 1;
      ori_step_len = step_len[0];
      if (ori_step_len > 0) {
        ori_step_block_id = step_block_list[ori_step_len - 1];
        tmp_used_len = used_list_len[ori_step_block_id];
        used_len = tmp_used_len < max_decoder_block_num ? tmp_used_len + 1
                                                        : tmp_used_len;
      }
    }
    need_block_len[0] = 0;
  }
  return api::SUCCESS;
}

static int xpu2or3_wrapper(Context *ctx,
                        bool *stop_flags,
                        int *seq_lens_this_time,
                        int *seq_lens_decoder,
                        int *block_tables,
                        int *encoder_block_lens,
                        bool *is_block_step,
                        int *step_block_list,  // [bsz]
                        int *step_len,
                        int *recover_block_list,
                        int *recover_len,
                        int *need_block_list,
                        int *need_block_len,
                        int *used_list_len,
                        int *free_list,
                        int *free_list_len,
                        int64_t *first_token_ids,
                        const int bsz,
                        const int block_size,
                        const int block_num_per_seq,
                        const int max_decoder_block_num) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  bool is_xpu2 = ctx->dev().type() == api::kXPU2;
  auto free_and_dispatch_block_kernel = is_xpu2
                                     ? xpu2::plugin::free_and_dispatch_block
                                     : xpu3::plugin::free_and_dispatch_block;
  free_and_dispatch_block_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          stop_flags,
          seq_lens_this_time,
          seq_lens_decoder,
          block_tables,
          encoder_block_lens,
          is_block_step,
          step_block_list,
          step_len,
          recover_block_list,
          recover_len,
          need_block_list,
          need_block_len,
          used_list_len,
          free_list,
          free_list_len,
          reinterpret_cast<XPU_INT64 *>(first_token_ids),
          bsz,
          block_size,
          block_num_per_seq,
          max_decoder_block_num);
  return api::SUCCESS;
}

int free_and_dispatch_block(Context *ctx,
                            bool *stop_flags,
                            int *seq_lens_this_time,
                            int *seq_lens_decoder,
                            int *block_tables,
                            int *encoder_block_lens,
                            bool *is_block_step,
                            int *step_block_list,  // [bsz]
                            int *step_len,
                            int *recover_block_list,
                            int *recover_len,
                            int *need_block_list,
                            int *need_block_len,
                            int *used_list_len,
                            int *free_list,
                            int *free_list_len,
                            int64_t *first_token_ids,
                            const int bsz,
                            const int block_size,
                            const int block_num_per_seq,
                            const int max_decoder_block_num) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "free_and_dispatch_block", float);
  WRAPPER_DUMP_PARAM6(ctx,
                      stop_flags,
                      seq_lens_this_time,
                      seq_lens_decoder,
                      block_tables,
                      encoder_block_lens,
                      is_block_step);
  WRAPPER_DUMP_PARAM6(ctx,
                      step_block_list,
                      step_len,
                      recover_block_list,
                      recover_len,
                      need_block_list,
                      need_block_len);
  WRAPPER_DUMP_PARAM4(
      ctx, used_list_len, free_list, free_list_len, first_token_ids);
  WRAPPER_DUMP_PARAM4(
      ctx, bsz, block_size, block_num_per_seq, max_decoder_block_num);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       stop_flags,
                       seq_lens_this_time,
                       seq_lens_decoder,
                       block_tables,
                       encoder_block_lens,
                       is_block_step,
                       step_block_list,
                       step_len,
                       recover_block_list,
                       recover_len,
                       need_block_list,
                       need_block_len,
                       used_list_len,
                       free_list,
                       free_list_len,
                       first_token_ids,
                       bsz,
                       block_size,
                       block_num_per_seq,
                       max_decoder_block_num);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                        stop_flags,
                        seq_lens_this_time,
                        seq_lens_decoder,
                        block_tables,
                        encoder_block_lens,
                        is_block_step,
                        step_block_list,
                        step_len,
                        recover_block_list,
                        recover_len,
                        need_block_list,
                        need_block_len,
                        used_list_len,
                        free_list,
                        free_list_len,
                        first_token_ids,
                        bsz,
                        block_size,
                        block_num_per_seq,
                        max_decoder_block_num);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
