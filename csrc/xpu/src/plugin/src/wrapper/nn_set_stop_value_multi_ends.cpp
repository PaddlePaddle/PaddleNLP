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
__attribute__((global)) void set_stop_value_multi_ends(bool* stop_flags,
                                                       T* topk_ids,
                                                       T* next_tokens,
                                                       const T* end_ids,
                                                       const int* seq_lens,
                                                       const int bs,
                                                       const int end_length,
                                                       const bool beam_search);
}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
__inline__ bool is_in_end(const T id, const T* end_ids, int length) {
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return false;
}

template <typename T>
static int cpu_wrapper(Context* ctx,
                       bool* stop_flags,
                       T* topk_ids,
                       T* next_tokens,
                       const T* end_ids,
                       const int* seq_lens,
                       const int bs,
                       const int end_length,
                       const bool beam_search) {
  for (int i = 0; i < bs; i++) {
    if (stop_flags[i]) {
      if (seq_lens[i] == 0) {
        topk_ids[i] = -1;
      } else {
        topk_ids[i] = end_ids[0];
        next_tokens[i] = end_ids[0];
      }
    } else {
      next_tokens[i] = topk_ids[i];
    }
    if (!beam_search && is_in_end(topk_ids[i], end_ids, end_length)) {
      stop_flags[i] = true;
    }
  }
  return api::SUCCESS;
}

template <typename T>
static int xpu2or3_wrapper(Context* ctx,
                        bool* stop_flags,
                        T* topk_ids,
                        T* next_tokens,
                        const T* end_ids,
                        const int* seq_lens,
                        const int bs,
                        const int end_length,
                        const bool beam_search) {
  using XPU_TID = typename XPUIndexType<T>::type;
  auto set_stop_value_multi_ends = xpu2::plugin::set_stop_value_multi_ends<XPU_TID>;
  set_stop_value_multi_ends<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      stop_flags,
      reinterpret_cast<XPU_TID*>(topk_ids),
      reinterpret_cast<XPU_TID*>(next_tokens),
      reinterpret_cast<const XPU_TID*>(end_ids),
      seq_lens,
      bs,
      end_length,
      beam_search);
  return api::SUCCESS;
}

template <typename T>
int set_stop_value_multi_ends(Context* ctx,
                              bool* stop_flags,
                              T* topk_ids,
                              T* next_tokens,
                              const T* end_ids,
                              const int* seq_lens,
                              const int bs,
                              const int end_length,
                              const bool beam_search) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "set_stop_value_multi_ends", T);
  WRAPPER_DUMP_PARAM5(
      ctx, stop_flags, topk_ids, next_tokens, end_ids, seq_lens);
  WRAPPER_DUMP_PARAM3(ctx, bs, end_length, beam_search);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, bool, bs, stop_flags);
  WRAPPER_CHECK_PTR(ctx, T, bs, topk_ids);
  WRAPPER_CHECK_PTR(ctx, T, end_length, end_ids);
  WRAPPER_CHECK_PTR(ctx, T, bs, seq_lens);
  WRAPPER_ASSERT_LE(ctx, end_length, 1024);  // assume end_length <= 1024
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx,
                          stop_flags,
                          topk_ids,
                          next_tokens,
                          end_ids,
                          seq_lens,
                          bs,
                          end_length,
                          beam_search);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper<T>(ctx,
                           stop_flags,
                           topk_ids,
                           next_tokens,
                           end_ids,
                           seq_lens,
                           bs,
                           end_length,
                           beam_search);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int set_stop_value_multi_ends<int64_t>(Context* ctx,
                                                bool* stop_flags,
                                                int64_t* topk_ids,
                                                int64_t* next_tokens,
                                                const int64_t* end_ids,
                                                const int* seq_lens,
                                                const int bs,
                                                const int end_length,
                                                const bool beam_search);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
