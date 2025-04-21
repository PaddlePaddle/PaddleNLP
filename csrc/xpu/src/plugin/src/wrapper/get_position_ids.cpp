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

__attribute__((global)) void get_position_ids(
                                  const int *seq_lens_encoder,
                                  const int *seq_lens_decoder,
                                  const int *seq_lens_this_time,
                                  int *position_ids,
                                  const int bs);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int xpu3_wrapper(Context *ctx,
                        const int *seq_lens_encoder,
                        const int *seq_lens_decoder,
                        const int *seq_lens_this_time,
                        int *position_ids,
                        const int bs) {
  // using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  // bool is_xpu2 = ctx->dev().type() == api::kXPU2;
  // auto get_padding_offset = is_xpu2 ? xpu2::plugin::get_padding_offset
  //                                   : xpu3::plugin::get_padding_offset;
  // auto remove_padding =
  //     is_xpu2 ? xpu2::plugin::remove_padding : xpu3::plugin::remove_padding;
  xpu3::plugin::get_position_ids<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      seq_lens_encoder,
      seq_lens_decoder,
      seq_lens_this_time,
      position_ids,
      bs);
  return api::SUCCESS;
  // get_padding_offset<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(padding_offset,
  //                                                              cum_offsets_out,
  //                                                              cu_seqlens_q,
  //                                                              cu_seqlens_k,
  //                                                              cum_offsets,
  //                                                              seq_lens,
  //                                                              max_seq_len,
  //                                                              bs);
  // remove_padding<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
  //     reinterpret_cast<XPU_INT64 *>(x_remove_padding),
  //     reinterpret_cast<const XPU_INT64 *>(input_ids),
  //     seq_lens,
  //     cum_offsets_out,
  //     max_seq_len,
  //     bs);
  // return api::SUCCESS;
}

int get_position_ids(Context *ctx,
                      const int *seq_lens_encoder,
                      const int *seq_lens_decoder,
                      const int *seq_lens_this_time,
                      int *position_ids,
                      const int bs) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "get_position_ids", int);
  WRAPPER_DUMP_PARAM5(
      ctx, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, position_ids, bs);
  // WRAPPER_DUMP_PARAM4(ctx, x_remove_padding, input_ids, cum_offsets, seq_lens);
  // WRAPPER_DUMP_PARAM2(ctx, max_seq_len, bs);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        seq_lens_this_time,
                        position_ids,
                        bs);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
