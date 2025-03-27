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

namespace xpu3 {
namespace plugin {

__attribute__((global)) void update_inputs_v2(bool *not_need_stop,
                                              int64_t* step_idx,
                                              bool* stop_flags,
                                              int* seq_lens_this_time,
                                              int* seq_lens_encoder,
                                              int* seq_lens_decoder,
                                              int64_t* next_tokens,
                                              int64_t* kwargs_next_tokens,
                                              int64_t* input_ids,
                                              const int64_t* end_ids,
                                              const int64_t* stop_nums,
                                              const bool* is_block_step,
                                              const int64_t* max_dec_len,
                                              int bsz,
                                              int max_bsz,
                                              int input_ids_stride,
                                              int end_length);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

bool is_in_end_v3(const int64_t id, const int64_t* end_ids, int length) {
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return false;
}

static int cpu_wrapper(Context *ctx,
                       bool* not_need_stop,
                       int64_t* step_idx,
                       bool* stop_flags,
                       int* seq_lens_this_time,
                       int* seq_lens_encoder,
                       int* seq_lens_decoder,
                       int64_t* next_tokens,
                       int64_t* kwargs_next_tokens,
                       int64_t* input_ids,
                       const int64_t* end_ids,
                       const int64_t* stop_nums,
                       const bool* is_block_step,
                       const int64_t* max_dec_len,
                       int bsz,
                       int max_bsz,
                       int input_ids_stride,
                       int end_length) {
  // part1:
  for (int i = 0; i < max_bsz; i++) {
    bool stop_flag = stop_flags[i];
    if (!stop_flag) {
      step_idx[i] += 1;
    }
    if (step_idx[i] >= max_dec_len[i]) {
      stop_flags[i] = true;
    }
  }
  // part2:
  for (int i = 0; i < bsz; i++) {
    if (stop_flags[i]) {
      if (seq_lens_this_time[i] == 0) {
        next_tokens[i] = -1;
      } else {
        next_tokens[i] = end_ids[0];
        kwargs_next_tokens[i] = end_ids[0];
      }
    } else {
      kwargs_next_tokens[i] = next_tokens[i];
    }
    if (is_in_end_v3(next_tokens[i], end_ids, end_length)) {
      stop_flags[i] = true;
    }
  }
  // part3: same with update_intputs
  std::vector<int64_t> stop_flag_now_int(max_bsz, 1);
  for (int i = 0; i < bsz; i++) {
    bool stop_flags_now = stop_flags[i];
    stop_flag_now_int[i] = is_block_step[i] ? 0 : stop_flags_now;
    const int seq_len_encoder = seq_lens_encoder[i];
    const int seq_len_decoder = seq_lens_decoder[i];

    seq_lens_decoder[i] = stop_flags_now ? 0
        : (seq_len_encoder > 0 ? (seq_len_encoder + seq_len_decoder) : seq_len_decoder + 1);

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
                           bool* not_need_stop,
                           int64_t* step_idx,
                           bool* stop_flags,
                           int* seq_lens_this_time,
                           int* seq_lens_encoder,
                           int* seq_lens_decoder,
                           int64_t* next_tokens,
                           int64_t* kwargs_next_tokens,
                           int64_t* input_ids,
                           const int64_t* end_ids,
                           const int64_t* stop_nums,
                           const bool* is_block_step,
                           const int64_t* max_dec_len,
                           int bsz,
                           int max_bsz,
                           int input_ids_stride,
                           int end_length) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto update_inputs_v2_kernel = xpu3::plugin::update_inputs_v2;
  update_inputs_v2_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      not_need_stop,
      reinterpret_cast<XPU_INT64 *>(step_idx),
      stop_flags,
      seq_lens_this_time,
      seq_lens_encoder,
      seq_lens_decoder,
      reinterpret_cast<XPU_INT64 *>(next_tokens),
      reinterpret_cast<XPU_INT64 *>(kwargs_next_tokens),
      reinterpret_cast<XPU_INT64 *>(input_ids),
      reinterpret_cast<const XPU_INT64 *>(end_ids),
      reinterpret_cast<const XPU_INT64 *>(stop_nums),
      is_block_step,
      reinterpret_cast<const XPU_INT64 *>(max_dec_len),
      bsz,
      max_bsz,
      input_ids_stride,
      end_length);

  return api::SUCCESS;
}

int update_inputs_v2(Context* ctx,
                    bool* not_need_stop,
                    int64_t* step_idx,
                    bool* stop_flags,
                    int* seq_lens_this_time,
                    int* seq_lens_encoder,
                    int* seq_lens_decoder,
                    int64_t* next_tokens,
                    int64_t* kwargs_next_tokens,
                    int64_t* input_ids,
                    const int64_t* end_ids,
                    const int64_t* stop_nums,
                    const bool* is_block_step,
                    const int64_t* max_dec_len,
                    int now_bsz,
                    int max_bsz,
                    int input_ids_stride,
                    int end_length) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "update_inputs_v2", int);
  WRAPPER_DUMP_PARAM5(ctx,
                      not_need_stop,
                      step_idx,
                      stop_flags,
                      seq_lens_this_time,
                      seq_lens_encoder);
  WRAPPER_DUMP_PARAM5(ctx,
                      seq_lens_decoder,
                      next_tokens,
                      kwargs_next_tokens,
                      input_ids,
                      end_ids);
  WRAPPER_DUMP_PARAM3(ctx, stop_nums, is_block_step, max_dec_len);
  WRAPPER_DUMP_PARAM4(ctx, now_bsz, max_bsz, input_ids_stride, end_length);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_LE(ctx, max_bsz, 1024);
  WRAPPER_ASSERT_LE(ctx, now_bsz, max_bsz);
  // TODO(mayang02): check ptrs
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       not_need_stop,
                       step_idx,
                       stop_flags,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       next_tokens,
                       kwargs_next_tokens,
                       input_ids,
                       end_ids,
                       stop_nums,
                       is_block_step,
                       max_dec_len,
                       now_bsz,
                       max_bsz,
                       input_ids_stride,
                       end_length);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                       not_need_stop,
                       step_idx,
                       stop_flags,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       next_tokens,
                       kwargs_next_tokens,
                       input_ids,
                       end_ids,
                       stop_nums,
                       is_block_step,
                       max_dec_len,
                       now_bsz,
                       max_bsz,
                       input_ids_stride,
                       end_length);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
