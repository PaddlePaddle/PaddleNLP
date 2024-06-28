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

template <typename T>
__attribute__((global)) void rebuild_padding(T *output_data, // [bs, dim_embed]
                                            const T *input_data, // [token_num, dim_embed]
                                            const int *cum_offsets, // [bs]
                                            const int *seq_len_decoder, // [bs]
                                            const int *seq_len_encoder, // [bs]
                                            const int seq_len,
                                            const int dim_embed,
                                            const int elem_nums);

}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {
template <typename T>
static int cpu_wrapper(Context *ctx,
                        T *output_data, // [bs, dim_embed]
                        const T *input_data, // [token_num, dim_embed]
                        const int *cum_offsets, // [bs]
                        const int *seq_len_decoder, // [bs]
                        const int *seq_len_encoder, // [bs]
                        const int seq_len,
                        const int dim_embed,
                        const int elem_nums) {
  for (int i=0;i < elem_nums;i++){
    const int bi = i / dim_embed;
    const int bias_idx = i % dim_embed;
    int seq_id = 0;
    // just encoder or stop, get last token; just decoder, get first token.
    if (seq_len_decoder[bi] == 0) {
        if (seq_len_encoder[bi] != 0) {
            seq_id = seq_len_encoder[bi] - 1;
        } else {
            continue;
        }
    }
    const int ori_token_idx = bi * seq_len - cum_offsets[bi] + seq_id;
    const int src_offset = ori_token_idx * dim_embed + bias_idx;
    output_data[i] = input_data[src_offset];
  }
  
  return api::SUCCESS;
}
template <typename T>
static int xpu2or3_wrapper(Context *ctx,
                        T *output_data, // [bs, dim_embed]
                        const T *input_data, // [token_num, dim_embed]
                        const int *cum_offsets, // [bs]
                        const int *seq_len_decoder, // [bs]
                        const int *seq_len_encoder, // [bs]
                        const int seq_len,
                        const int dim_embed,
                        const int elem_nums) {
  xpu2::plugin::rebuild_padding<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(output_data,
                                                               input_data,
                                                               cum_offsets,
                                                               seq_len_decoder,
                                                               seq_len_encoder,
                                                               seq_len,
                                                               dim_embed,
                                                               elem_nums);
  return api::SUCCESS;
}

template <typename T>
int rebuild_padding(Context *ctx,
                    T *output_data, // [bs, dim_embed]
                    const T *input_data, // [token_num, dim_embed]
                    const int *cum_offsets, // [bs]
                    const int *seq_len_decoder, // [bs]
                    const int *seq_len_encoder, // [bs]
                    const int seq_len,
                    const int dim_embed,
                    const int elem_nums) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "rebuild_padding", T);
  WRAPPER_DUMP_PARAM5(
      ctx, output_data, input_data, cum_offsets, seq_len_decoder, seq_len_encoder);
  WRAPPER_DUMP_PARAM3(ctx, seq_len, dim_embed, elem_nums);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       output_data,
                       input_data,
                       cum_offsets,
                       seq_len_decoder,
                       seq_len_encoder,
                       seq_len,
                       dim_embed,
                       elem_nums);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                       output_data,
                       input_data,
                       cum_offsets,
                       seq_len_decoder,
                       seq_len_encoder,
                       seq_len,
                       dim_embed,
                       elem_nums);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}
template int rebuild_padding<float>(Context *ctx,
                    float *output_data, // [bs, dim_embed]
                    const float *input_data, // [token_num, dim_embed]
                    const int *cum_offsets, // [bs]
                    const int *seq_len_decoder, // [bs]
                    const int *seq_len_encoder, // [bs]
                    const int seq_len,
                    const int dim_embed,
                    const int elem_nums);
template int rebuild_padding<float16>(Context *ctx,
                    float16 *output_data, // [bs, dim_embed]
                    const float16 *input_data, // [token_num, dim_embed]
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