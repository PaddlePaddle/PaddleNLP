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

__attribute__((global)) void get_padding_offset(int *padding_offset,
                                                int *cum_offsets_out,
                                                int *cu_seqlens_q,
                                                int *cu_seqlens_k,
                                                const int *cum_offsets,
                                                const int *seq_lens,
                                                const int max_seq_len,
                                                const int bs);
__attribute__((global)) void remove_padding(int64_t *x_remove_padding,
                                            const int64_t *input_data,
                                            const int *seq_lens,
                                            const int *cum_offsets,
                                            const int sequence_length,
                                            const int bs);

}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int get_padding_offset_cpu(int *padding_offset,
                                  int *cum_offsets_out,
                                  int *cu_seqlens_q,
                                  int *cu_seqlens_k,
                                  const int *cum_offsets,
                                  const int *seq_lens,
                                  const int max_seq_len,
                                  const int bs) {
  for (int i = 0; i < bs; i++) {
    int cum_offset = i == 0 ? 0 : cum_offsets[i - 1];
    for (int j = 0; j < seq_lens[i]; j++) {
      padding_offset[i * max_seq_len - cum_offset + j] = cum_offset;
    }
    cum_offsets_out[i] = cum_offset;
    int cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i];
    cu_seqlens_q[i + 1] = cum_seq_len;
    cu_seqlens_k[i + 1] = cum_seq_len;
  }
  return api::SUCCESS;
}

static int remove_padding_cpu(int64_t *x_remove_padding,
                              const int64_t *input_data,
                              const int *seq_lens,
                              const int *cum_offsets,
                              const int sequence_length,
                              const int bs) {
  for (int i = 0; i < bs; i++) {
    for (int j = 0; j < seq_lens[i]; j++) {
      const int tgt_seq_id = i * sequence_length - cum_offsets[i] + j;
      const int src_seq_id = i * sequence_length + j;
      x_remove_padding[tgt_seq_id] = input_data[src_seq_id];
    }
  }
  return api::SUCCESS;
}

static int cpu_wrapper(Context *ctx,
                       int *padding_offset,
                       int *cum_offsets_out,
                       int *cu_seqlens_q,
                       int *cu_seqlens_k,
                       int64_t *x_remove_padding,
                       const int64_t *input_ids,
                       const int *cum_offsets,
                       const int *seq_lens,
                       const int max_seq_len,
                       const int bs) {
  get_padding_offset_cpu(padding_offset,
                         cum_offsets_out,
                         cu_seqlens_q,
                         cu_seqlens_k,
                         cum_offsets,
                         seq_lens,
                         max_seq_len,
                         bs);
  remove_padding_cpu(
      x_remove_padding, input_ids, seq_lens, cum_offsets_out, max_seq_len, bs);
  return api::SUCCESS;
}

static int xpu2or3_wrapper(Context *ctx,
                        int *padding_offset,
                        int *cum_offsets_out,
                        int *cu_seqlens_q,
                        int *cu_seqlens_k,
                        int64_t *x_remove_padding,
                        const int64_t *input_ids,
                        const int *cum_offsets,
                        const int *seq_lens,
                        const int max_seq_len,
                        const int bs) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto get_padding_offset = xpu2::plugin::get_padding_offset;
  auto remove_padding = xpu2::plugin::remove_padding;
  get_padding_offset<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(padding_offset,
                                                               cum_offsets_out,
                                                               cu_seqlens_q,
                                                               cu_seqlens_k,
                                                               cum_offsets,
                                                               seq_lens,
                                                               max_seq_len,
                                                               bs);
  remove_padding<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      reinterpret_cast<XPU_INT64 *>(x_remove_padding),
      reinterpret_cast<const XPU_INT64 *>(input_ids),
      seq_lens,
      cum_offsets_out,
      max_seq_len,
      bs);
  return api::SUCCESS;
}

int get_padding_offset(Context *ctx,
                       int *padding_offset,
                       int *cum_offsets_out,
                       int *cu_seqlens_q,
                       int *cu_seqlens_k,
                       int64_t *x_remove_padding,
                       const int64_t *input_ids,
                       const int *cum_offsets,
                       const int *seq_lens,
                       const int max_seq_len,
                       const int bs) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "get_padding_offset", int);
  WRAPPER_DUMP_PARAM4(
      ctx, padding_offset, cum_offsets_out, cu_seqlens_q, cu_seqlens_k);
  WRAPPER_DUMP_PARAM4(ctx, x_remove_padding, input_ids, cum_offsets, seq_lens);
  WRAPPER_DUMP_PARAM2(ctx, max_seq_len, bs);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       padding_offset,
                       cum_offsets_out,
                       cu_seqlens_q,
                       cu_seqlens_k,
                       x_remove_padding,
                       input_ids,
                       cum_offsets,
                       seq_lens,
                       max_seq_len,
                       bs);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                        padding_offset,
                        cum_offsets_out,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        x_remove_padding,
                        input_ids,
                        cum_offsets,
                        seq_lens,
                        max_seq_len,
                        bs);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
