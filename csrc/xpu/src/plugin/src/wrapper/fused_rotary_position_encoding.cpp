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
template <typename T, typename TR>
__attribute__((global)) void rotary_embedding_neox(
        const int* positions,  // [num_tokens]
        T* query,                  // [num_tokens, num_heads, head_size]
        T* key,                    // [num_tokens, num_kv_heads, head_size]
        const TR* cos_sin_cache,   // [max_position, 2, rot_dim // 2]
        const int rot_dim,
        const int num_heads,
        const int num_kv_heads,
        const int head_size,
        const int32_t num_tokens);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

// template <typename T>
// static int cpu_wrapper() {
// }

template <typename T, typename TR>
static int xpu3_wrapper(Context* ctx,
        const int* positions,  // [num_tokens]
        T* query,                  // [num_tokens, num_heads, head_size]
        T* key,                    // [num_tokens, num_kv_heads, head_size]
        const TR* cos_sin_cache,   // [max_position, 2, rot_dim // 2]
        const int rot_dim,
        const int num_heads,
        const int num_kv_heads,
        const int head_size,
        const int32_t num_tokens) {
  using XPU_TID = typename XPUIndexType<T>::type;
  using XPU_TRID = typename XPUIndexType<TR>::type;
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto rotary_embedding_neox = xpu3::plugin::rotary_embedding_neox<XPU_TID, XPU_TRID>;
  rotary_embedding_neox<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      positions,
      reinterpret_cast<XPU_TID *>(query),
      reinterpret_cast<XPU_TID *>(key),
      reinterpret_cast<const XPU_TRID *>(cos_sin_cache),
      rot_dim,
      num_heads,
      num_kv_heads,
      head_size,
      num_tokens);
  return api::SUCCESS;
}

// template <typename T>
// int set_stop_value_multi_ends(Context* ctx,
//                               bool* stop_flags,
//                               T* topk_ids,
//                               T* next_tokens,
//                               const T* end_ids,
//                               const int* seq_lens,
//                               const int bs,
//                               const int end_length,
//                               const bool beam_search) 
                              
                              
template <typename T, typename TR>
int rotary_embedding_neox(
        Context* ctx,
        const int* positions,  // [num_tokens]
        T* query,                  // [num_tokens, num_heads, head_size]
        T* key,                    // [num_tokens, num_kv_heads, head_size]
        const TR* cos_sin_cache,   // [max_position, 2, rot_dim // 2]
        const int rot_dim,
        const int num_heads,
        const int num_kv_heads,
        const int head_size,
        const int32_t num_tokens)
{
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T2(ctx, "rotary_embedding_neox", T, TR);
  WRAPPER_DUMP_PARAM5(
      ctx, positions, query, key, cos_sin_cache, rot_dim);
  WRAPPER_DUMP_PARAM4(ctx, num_heads, num_kv_heads, head_size,num_tokens);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, num_tokens, positions);
  WRAPPER_CHECK_PTR(ctx, T, num_tokens * num_heads * head_size, query);
  WRAPPER_CHECK_PTR(ctx, T, num_tokens * num_kv_heads * head_size, key);
  WRAPPER_CHECK_PTR(ctx, TR, num_tokens * rot_dim, cos_sin_cache);

  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T, TR>(ctx,
                           positions,
                           query,
                           key,
                           cos_sin_cache,
                           rot_dim,
                           num_heads,
                           num_kv_heads,
                           head_size,
                           num_tokens);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}


template int rotary_embedding_neox<float, float>(
        Context* ctx,
        const int* positions,  // [num_tokens]
        float* query,                  // [num_tokens, num_heads, head_size]
        float* key,                    // [num_tokens, num_kv_heads, head_size]
        const float* cos_sin_cache,   // [max_position, 2, rot_dim // 2]
        const int rot_dim,
        const int num_heads,
        const int num_kv_heads,
        const int head_size,
        const int32_t num_tokens);

template int rotary_embedding_neox<float16, float16>(
        Context* ctx,
        const int* positions,  // [num_tokens]
        float16* query,                  // [num_tokens, num_heads, head_size]
        float16* key,                    // [num_tokens, num_kv_heads, head_size]
        const float16* cos_sin_cache,   // [max_position, 2, rot_dim // 2]
        const int rot_dim,
        const int num_heads,
        const int num_kv_heads,
        const int head_size,
        const int32_t num_tokens);

template int rotary_embedding_neox<bfloat16, bfloat16>(
        Context* ctx,
        const int* positions,  // [num_tokens]
        bfloat16* query,                  // [num_tokens, num_heads, head_size]
        bfloat16* key,                    // [num_tokens, num_kv_heads, head_size]
        const bfloat16* cos_sin_cache,   // [max_position, 2, rot_dim // 2]
        const int rot_dim,
        const int num_heads,
        const int num_kv_heads,
        const int head_size,
        const int32_t num_tokens);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
