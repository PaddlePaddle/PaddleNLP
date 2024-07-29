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

#include <algorithm>
#include <numeric>
#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace xpu2 {
namespace plugin {

__attribute__((global)) void update_inputs(bool *not_need_stop,
                                           int *seq_lens_this_time,
                                           int *seq_lens_encoder,
                                           int *seq_lens_decoder,
                                           int64_t *input_ids,
                                           const int64_t *stop_nums,
                                           const bool *stop_flags,
                                           const bool *is_block_step,
                                           const int64_t *next_tokens,
                                           const int bsz,
                                           const int max_bsz,
                                           const int input_ids_stride);

}  // namespace plugin
}  // namespace xpu2

namespace xpu3 {
namespace plugin {

__attribute__((global)) void update_inputs(bool *not_need_stop,
                                           int *seq_lens_this_time,
                                           int *seq_lens_encoder,
                                           int *seq_lens_decoder,
                                           int64_t *input_ids,
                                           const int64_t *stop_nums,
                                           const bool *stop_flags,
                                           const bool *is_block_step,
                                           const int64_t *next_tokens,
                                           const int bsz,
                                           const int max_bsz,
                                           const int input_ids_stride);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context *ctx,
                       bool *not_need_stop,
                       int *seq_lens_this_time,
                       int *seq_lens_encoder,
                       int *seq_lens_decoder,
                       int64_t *input_ids,
                       const int64_t *stop_nums,
                       const bool *stop_flags,
                       const bool *is_block_step,
                       const int64_t *next_tokens,
                       const int bsz,
                       const int max_bsz,
                       const int input_ids_stride) {
  std::vector<int64_t> stop_flag_now_int(max_bsz, 1);
  for (int i = 0; i < bsz; i++) {
    bool stop_flags_now = stop_flags[i];
    stop_flag_now_int[i] = is_block_step[i] ? 0 : stop_flags_now;
    const int seq_len_encoder = seq_lens_encoder[i];
    const int seq_len_decoder = seq_lens_decoder[i];

    seq_lens_decoder[i] =
        stop_flags[i]
            ? 0
            : (seq_len_decoder == 0 ? seq_len_encoder : seq_len_decoder + 1);

    seq_lens_this_time[i] = stop_flags[i] ? 0 : 1;
    seq_lens_encoder[i] = 0;
    int64_t *input_ids_now = input_ids + i * input_ids_stride;
    input_ids_now[0] = next_tokens[i];
  }
  int64_t stop_sum = 0;
  for (size_t i = 0; i < stop_flag_now_int.size(); i++) {
    stop_sum += stop_flag_now_int[i];
  }
  not_need_stop[0] = stop_sum < stop_nums[0];
  return api::SUCCESS;
}

static int xpu2or3_wrapper(Context *ctx,
                        bool *not_need_stop,
                        int *seq_lens_this_time,
                        int *seq_lens_encoder,
                        int *seq_lens_decoder,
                        int64_t *input_ids,
                        const int64_t *stop_nums,
                        const bool *stop_flags,
                        const bool *is_block_step,
                        const int64_t *next_tokens,
                        const int bsz,
                        const int max_bsz,
                        const int input_ids_stride) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto update_inputs = xpu2::plugin::update_inputs;
  update_inputs<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      not_need_stop,
      seq_lens_this_time,
      seq_lens_encoder,
      seq_lens_decoder,
      reinterpret_cast<XPU_INT64 *>(input_ids),
      reinterpret_cast<const XPU_INT64 *>(stop_nums),
      stop_flags,
      is_block_step,
      reinterpret_cast<const XPU_INT64 *>(next_tokens),
      bsz,
      max_bsz,
      input_ids_stride);
  return api::SUCCESS;
}

int update_inputs(Context *ctx,
                  bool *not_need_stop,
                  int *seq_lens_this_time,
                  int *seq_lens_encoder,
                  int *seq_lens_decoder,
                  int64_t *input_ids,
                  const int64_t *stop_nums,
                  const bool *stop_flags,
                  const bool *is_block_step,
                  const int64_t *next_tokens,
                  const int bsz,
                  const int max_bsz,
                  const int input_ids_stride) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "update_inputs", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      not_need_stop,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      input_ids);
  WRAPPER_DUMP_PARAM4(ctx, stop_nums, stop_flags, is_block_step, next_tokens);
  WRAPPER_DUMP_PARAM3(ctx, bsz, max_bsz, input_ids_stride);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       not_need_stop,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       input_ids,
                       stop_nums,
                       stop_flags,
                       is_block_step,
                       next_tokens,
                       bsz,
                       max_bsz,
                       input_ids_stride);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                        not_need_stop,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        input_ids,
                        stop_nums,
                        stop_flags,
                        is_block_step,
                        next_tokens,
                        bsz,
                        max_bsz,
                        input_ids_stride);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
