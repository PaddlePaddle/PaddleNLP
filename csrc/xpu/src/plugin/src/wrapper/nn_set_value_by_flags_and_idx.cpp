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

__attribute__((global)) void set_value_by_flags_and_idx(
    const bool* stop_flags,
    int64_t* pre_ids_all,
    const int64_t* input_ids,
    const int* seq_lens_encoder,
    const int* seq_lens_decoder,
    const int64_t* step_idx,
    int bs,
    int length,
    int length_input_ids);

}  // namespace plugin
}  // namespace xpu2

namespace xpu3 {
namespace plugin {

__attribute__((global)) void set_value_by_flags_and_idx(
    const bool* stop_flags,
    int64_t* pre_ids_all,
    const int64_t* input_ids,
    const int* seq_lens_encoder,
    const int* seq_lens_decoder,
    const int64_t* step_idx,
    int bs,
    int length,
    int length_input_ids);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context* ctx,
                       const bool* stop_flags,
                       int64_t* pre_ids_all,
                       const int64_t* pre_ids,
                       const int64_t* step_idx,
                       const int bs,
                       const int length) {
  for (int i = 0; i < bs; i++) {
    int64_t* pre_ids_all_now = pre_ids_all + i * length;
    if (!stop_flags[i] && step_idx[i] >= 0) {
      pre_ids_all_now[step_idx[i]] = pre_ids[i];
    }
  }
  return api::SUCCESS;
}

static int cpu_wrapper(Context* ctx,
                       const bool* stop_flags,
                       int64_t* pre_ids_all,
                       const int64_t* input_ids,
                       const int* seq_lens_encoder,
                       const int* seq_lens_decoder,
                       const int64_t* step_idx,
                       int bs,
                       int length,
                       int length_input_ids) {
  for (int i = 0; i < bs; i++) {
    if (!stop_flags[i]) {
      int64_t* pre_ids_all_now = pre_ids_all + i * length;
      const int64_t* input_ids_now = input_ids + i * length_input_ids;
      const int seq_len_dec = seq_lens_decoder[i];
      const int seq_len_enc = seq_lens_encoder[i];
      if (seq_len_dec == 0 && seq_len_enc == 0) continue;
      if (step_idx[i] >= 0) {
        if (seq_len_dec ==
            0) {  // encoder, get last token accord to seq_lens_encoder
          pre_ids_all_now[step_idx[i]] = input_ids_now[seq_len_enc - 1];
        } else {  // decoder, get first token
          pre_ids_all_now[step_idx[i]] = input_ids_now[0];
        }
      }
    }
  }
  return api::SUCCESS;
}

static int xpu2or3_wrapper(Context* ctx,
                        const bool* stop_flags,
                        int64_t* pre_ids_all,
                        const int64_t* input_ids,
                        const int* seq_lens_encoder,
                        const int* seq_lens_decoder,
                        const int64_t* step_idx,
                        int bs,
                        int length,
                        int length_input_ids) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto set_value_by_flags_and_idx_kernel = xpu2::plugin::set_value_by_flags_and_idx;
  set_value_by_flags_and_idx_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      stop_flags,
      reinterpret_cast<XPU_INT64*>(pre_ids_all),
      reinterpret_cast<const XPU_INT64*>(input_ids),
      seq_lens_encoder,
      seq_lens_decoder,
      reinterpret_cast<const XPU_INT64*>(step_idx),
      bs,
      length,
      length_input_ids);
  return api::SUCCESS;
}

int set_value_by_flags_and_idx(Context* ctx,
                               const bool* stop_flags,
                               int64_t* pre_ids_all,
                               const int64_t* input_ids,
                               const int* seq_lens_encoder,
                               const int* seq_lens_decoder,
                               const int64_t* step_idx,
                               int bs,
                               int length,
                               int length_input_ids) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "set_value_by_flags_and_idx", int64_t);
  WRAPPER_DUMP_PARAM6(ctx,
                      stop_flags,
                      pre_ids_all,
                      input_ids,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      step_idx);
  WRAPPER_DUMP_PARAM3(ctx, bs, length, length_input_ids);
  WRAPPER_DUMP(ctx);
  int64_t stop_flags_len = -1;
  int64_t pre_ids_all_len = -1;
  int64_t input_ids_len = -1;
  int64_t seq_lens_encoder_len = -1;
  int64_t seq_lens_decoder_len = -1;
  int64_t step_idx_len = -1;
  WRAPPER_CHECK_SHAPE(ctx, &stop_flags_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &pre_ids_all_len, {bs, length});
  WRAPPER_CHECK_SHAPE(ctx, &input_ids_len, {bs, length_input_ids});
  WRAPPER_CHECK_SHAPE(ctx, &seq_lens_encoder_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &seq_lens_decoder_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &step_idx_len, {bs});
  WRAPPER_CHECK_PTR(ctx, int64_t, stop_flags_len, stop_flags);
  WRAPPER_CHECK_PTR(ctx, int64_t, pre_ids_all_len, pre_ids_all);
  WRAPPER_CHECK_PTR(ctx, int64_t, input_ids_len, input_ids);
  WRAPPER_CHECK_PTR(ctx, int, seq_lens_encoder_len, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, seq_lens_decoder_len, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, int64_t, step_idx_len, step_idx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       stop_flags,
                       pre_ids_all,
                       input_ids,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       step_idx,
                       bs,
                       length,
                       length_input_ids);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                        stop_flags,
                        pre_ids_all,
                        input_ids,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        step_idx,
                        bs,
                        length,
                        length_input_ids);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
