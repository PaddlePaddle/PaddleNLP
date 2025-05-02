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

template <typename TI, typename TO, typename TID>
__attribute__((global)) void absorb_mla_cache_write(const TI* src,
                                                    TO* dst,
                                                    const TID* block_table,
                                                    int32_t* kv_seq_lod,
                                                    int32_t* start_token,
                                                    int32_t* real_batch,
                                                    int64_t batch_size,
                                                    int64_t kv_head_num,
                                                    int64_t head_dim,
                                                    int64_t max_seq_num,
                                                    int64_t block_size,
                                                    int64_t max_num_blocks_per_seq);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename TI, typename TO, typename TID>
static int xpu3_wrapper(
        Context* ctx,
        const TI* x,
        TO* y,
        const TID* block_table,
        const VectorParam<int32_t>& kv_seq_lod_xpu,
        const VectorParam<int32_t>& start_tokens_xpu,
        const VectorParam<int32_t>& real_batch_xpu,
        int64_t batch_size,
        int64_t kv_head_num,
        int64_t head_dim,
        int64_t max_seq_num,
        int64_t block_size,
        int64_t max_num_blocks_per_seq) {

        auto func = &xpu3::plugin::absorb_mla_cache_write<TI, TO, TID>;
        func<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
                x,
                y,
                block_table,
                kv_seq_lod_xpu.xpu,
                start_tokens_xpu.xpu,
                real_batch_xpu.xpu,
                batch_size,
                kv_head_num,
                head_dim,
                max_seq_num,
                block_size,
                max_num_blocks_per_seq);

  return api::SUCCESS;

}

template <typename TI, typename TO, typename TID>
int absorb_mla_cache_write_xpu(Context *ctx,
        const TI* x,
        TO* y,
        const TID* block_table,
        const VectorParam<int32_t>& kv_seq_lod,
        const VectorParam<int32_t>& start_tokens,
        const VectorParam<int32_t>& real_batch,
        int64_t batch_size,
        int64_t kv_head_num,
        int64_t head_dim,
        int64_t max_seq_num,
        int64_t block_size,
        int64_t max_num_blocks_per_seq) {
    WRAPPER_CHECK_CTX(ctx);
    WRAPPER_DUMP_FUNCTION_T3(ctx, "absorb_mla_cache_write", TI, TO, TID);
    WRAPPER_DUMP_PARAM6(ctx, x, y, block_table, kv_seq_lod, start_tokens, real_batch);
    WRAPPER_DUMP_PARAM6(ctx, batch_size, kv_head_num, head_dim, max_seq_num, block_size, max_num_blocks_per_seq);
    WRAPPER_DUMP(ctx);
    // check vector param size
    WRAPPER_ASSERT_EQ(ctx, batch_size + 1, kv_seq_lod.len);
    WRAPPER_ASSERT_EQ(ctx, batch_size, start_tokens.len);
    WRAPPER_ASSERT_EQ(ctx, batch_size, real_batch.len);
    WRAPPER_ASSERT_EQ(ctx, kv_head_num, 1);
    WRAPPER_ASSERT_LE(ctx, batch_size, 32);
    // check input ptr shape
    int64_t kv_seqlen_sum = kv_seq_lod.cpu[batch_size];
    int64_t hidden_dim = kv_head_num * head_dim;
    // check kv_cache and block_table ptr shape
    int64_t table_count = max_seq_num * max_num_blocks_per_seq;        // block number in a block table
    int64_t kv_cache_size = table_count * kv_head_num * head_dim * block_size; 
    // x 'shape is supposed to be [kv_seqlen_sum, hidden_dim]
    WRAPPER_CHECK_PTR(ctx, TI, kv_seqlen_sum * hidden_dim, x);
    // y is kv cache ptr, which is supposed to be [table_count, kv_head_num, block_size, head_dim]
    WRAPPER_CHECK_PTR(ctx, TO, kv_cache_size, y);

    // int64_t single_block_occupy = kv_head_num * head_dim * block_size; // single_block_occupy represent how many bytes
                                                                       // occupied in a block
                                                                       // which is different from block size
                                                                       // block_size means the seqlen a block can hold
    // kv_cache 'shape is supposed to be [block_num, hidden_dim, block_size]
    // FIXME: in real cases, kv cache may not be full
    // WRAPPER_CHECK_PTR(ctx, TO, table_count * single_block_occupy, y);
    // block_table 'shape is supposed to be [max_seq_num, max_num_blocks_per_seq]
    WRAPPER_CHECK_PTR(ctx, TID, table_count, block_table);

  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<TI, TO, TID>(ctx,
                                        x,
                                        y,
                                        block_table,
                                        kv_seq_lod,
                                        start_tokens,
                                        real_batch,
                                        batch_size,
                                        kv_head_num,
                                        head_dim,
                                        max_seq_num,
                                        block_size,
                                        max_num_blocks_per_seq);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

#define INSTANTIATION_ABSORB_MLA_CACHE_WRITE(TI, TO, TID)                                                                \
    template int absorb_mla_cache_write_xpu<TI, TO, TID>(api::Context*,                                                     \
                                                const TI*,                                                             \
                                                TO*,                                                                   \
                                                const TID*,                                                            \
                                                const api::VectorParam<int32_t>&,                                      \
                                                const api::VectorParam<int32_t>&,                                      \
                                                const api::VectorParam<int32_t>&,                                      \
                                                int64_t,                                                               \
                                                int64_t,                                                               \
                                                int64_t,                                                               \
                                                int64_t,                                                               \
                                                int64_t,                                                               \
                                                int64_t);

INSTANTIATION_ABSORB_MLA_CACHE_WRITE(float16, float16, int32_t);
INSTANTIATION_ABSORB_MLA_CACHE_WRITE(bfloat16, bfloat16, int32_t);



}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
