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
__attribute__((global)) void min_length_logits_process(
    T* logits,
    const int64_t* cur_len,
    const int64_t* min_len,
    const int64_t* eos_token_id,
    const int64_t bs,
    const int64_t length,
    const int64_t length_id,
    const int64_t end_length);
__attribute__((global)) void update_repeat_times(const int64_t* pre_ids,
                                                 const int64_t* cur_len,
                                                 int* repeat_times,
                                                 const int64_t bs,
                                                 const int64_t length,
                                                 const int64_t length_id);
template <typename T>
__attribute__((global)) void update_value_by_repeat_times(
    const int* repeat_times,
    const T* penalty_scores,
    const T* frequency_score,
    const T* presence_score,
    const float* temperatures,
    T* logits,
    const int64_t bs,
    const int64_t length);
template <typename T>
__attribute__((global)) void update_value_by_repeat_times_simd(
    const int* repeat_times,
    const T* penalty_scores,
    const T* frequency_score,
    const T* presence_score,
    const float* temperatures,
    T* logits,
    const int64_t bs,
    const int64_t length);
template <typename T>
__attribute__((global)) void ban_bad_words(T* logits,
                                           const int64_t* bad_words_list,
                                           const int64_t bs,
                                           const int64_t length,
                                           const int64_t bad_words_length);

}  // namespace plugin
}  // namespace xpu2

namespace xpu3 {
namespace plugin {

template <typename T>
__attribute__((global)) void min_length_logits_process(
    T* logits,
    const int64_t* cur_len,
    const int64_t* min_len,
    const int64_t* eos_token_id,
    const int64_t bs,
    const int64_t length,
    const int64_t length_id,
    const int64_t end_length);
__attribute__((global)) void update_repeat_times(const int64_t* pre_ids,
                                                 const int64_t* cur_len,
                                                 int* repeat_times,
                                                 const int64_t bs,
                                                 const int64_t length,
                                                 const int64_t length_id);
template <typename T>
__attribute__((global)) void update_value_by_repeat_times(
    const int* repeat_times,
    const T* penalty_scores,
    const T* frequency_score,
    const T* presence_score,
    const float* temperatures,
    T* logits,
    const int64_t bs,
    const int64_t length);
template <typename T>
__attribute__((global)) void update_value_by_repeat_times_simd(
    const int* repeat_times,
    const T* penalty_scores,
    const T* frequency_score,
    const T* presence_score,
    const float* temperatures,
    T* logits,
    const int64_t bs,
    const int64_t length);
template <typename T>
__attribute__((global)) void ban_bad_words(T* logits,
                                           const int64_t* bad_words_list,
                                           const int64_t bs,
                                           const int64_t length,
                                           const int64_t bad_words_length);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

void update_repeat_times_cpu(const int64_t* pre_ids,
                             const int64_t* cur_len,
                             int* repeat_times,
                             const int64_t bs,
                             const int64_t length,
                             const int64_t length_id) {
  for (int64_t i = 0; i < bs; i++) {
    if (cur_len[i] >= 0) {
      for (int64_t j = 0; j < length_id; j++) {
        int64_t id = pre_ids[i * length_id + j];
        if (id < 0 || id >= length) continue;
        repeat_times[i * length + id] += 1;
      }
    }
  }
}

void ban_bad_words_cpu(float* logits,
                       const int64_t* bad_words_list,
                       const int64_t bs,
                       const int64_t length,
                       const int64_t bad_words_length) {
  for (int64_t i = 0; i < bs; i++) {
    float* logits_now = logits + i * length;
    for (int64_t j = 0; j < bad_words_length; j++) {
      int64_t bad_words_token_id = bad_words_list[j];
      if (bad_words_token_id >= length || bad_words_token_id < 0) continue;
      logits_now[bad_words_token_id] = -1e10;
    }
  }
}

template <typename T>
static int cpu_wrapper(Context* ctx,
                       const int64_t* pre_ids,
                       T* logits,
                       const T* penalty_scores,
                       const T* frequency_scores,
                       const T* presence_scores,
                       const float* temperatures,
                       const int64_t* cur_len,
                       const int64_t* min_len,
                       const int64_t* eos_token_id,
                       const int64_t* bad_words,
                       const int64_t bs,
                       const int64_t length,
                       const int64_t length_id,
                       const int64_t end_length,
                       const int64_t length_bad_words) {
  std::vector<float> logitsfp32(bs * length);
  std::vector<float> penalty_scoresfp32(bs);
  std::vector<float> frequency_scoresfp32(bs);
  std::vector<float> presence_scoresfp32(bs);
  std::vector<int> repeat_times_buffer(bs * length, 0);
  int ret = api::cast<T, float>(ctx, logits, logitsfp32.data(), bs * length);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::cast<T, float>(ctx, penalty_scores, penalty_scoresfp32.data(), bs);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::cast<T, float>(
      ctx, frequency_scores, frequency_scoresfp32.data(), bs);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret =
      api::cast<T, float>(ctx, presence_scores, presence_scoresfp32.data(), bs);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  for (int64_t i = 0; i < bs; i++) {
    if (cur_len[i] >= 0 && cur_len[i] < min_len[i]) {
      for (int64_t j = 0; j < end_length; j++) {
        logitsfp32[i * length + eos_token_id[j]] = -1e4;
      }
    }
  }
  int* repeat_times = &(repeat_times_buffer[0]);
  update_repeat_times_cpu(
      pre_ids, cur_len, repeat_times, bs, length, length_id);
  for (int64_t i = 0; i < bs; i++) {
    float alpha = penalty_scoresfp32[i];
    float beta = frequency_scoresfp32[i];
    float gamma = presence_scoresfp32[i];
    float temperature = temperatures[i];
    for (int64_t j = 0; j < length; j++) {
      int times = repeat_times[i * length + j];
      float logit_now = logitsfp32[i * length + j];
      if (times != 0) {
        logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
        logit_now = logit_now - times * beta - gamma;
      }
      logitsfp32[i * length + j] = logit_now / temperature;
    }
  }
  if (bad_words && length_bad_words > 0) {
    ban_bad_words_cpu(
        logitsfp32.data(), bad_words, bs, length, length_bad_words);
  }
  ret = api::cast<float, T>(ctx, logitsfp32.data(), logits, bs * length);
  return ret;
}

template <typename T>
static int xpu2or3_wrapper(Context* ctx,
                        const int64_t* pre_ids,
                        T* logits,
                        const T* penalty_scores,
                        const T* frequency_scores,
                        const T* presence_scores,
                        const float* temperatures,
                        const int64_t* cur_len,
                        const int64_t* min_len,
                        const int64_t* eos_token_id,
                        const int64_t* bad_words,
                        const int64_t bs,
                        const int64_t length,
                        const int64_t length_id,
                        const int64_t end_length,
                        const int64_t length_bad_words) {
  api::ctx_guard RAII_GUARD(ctx);
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto min_length_logits_process_kernel = xpu2::plugin::min_length_logits_process<T>;
  auto update_repeat_times_kernel = xpu2::plugin::update_repeat_times;
  auto update_value_by_repeat_times_kernel = xpu2::plugin::update_value_by_repeat_times<T>;
  if(length % 16 == 0) {
    update_value_by_repeat_times_kernel = xpu2::plugin::update_value_by_repeat_times_simd<T>;
  }
  auto ban_bad_words_kernel = xpu2::plugin::ban_bad_words<T>;

  int* repeat_times = RAII_GUARD.alloc_l3_or_gm<int>(bs * length);
  WRAPPER_ASSERT_WORKSPACE(ctx, repeat_times);
  int ret = api::constant<int>(ctx, repeat_times, bs * length, 0);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);

  update_repeat_times_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      reinterpret_cast<const XPU_INT64*>(pre_ids),
      reinterpret_cast<const XPU_INT64*>(cur_len),
      repeat_times,
      bs,
      length,
      length_id);
  min_length_logits_process_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      logits,
      reinterpret_cast<const XPU_INT64*>(cur_len),
      reinterpret_cast<const XPU_INT64*>(min_len),
      reinterpret_cast<const XPU_INT64*>(eos_token_id),
      bs,
      length,
      length_id,
      end_length);
  update_value_by_repeat_times_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      repeat_times,
      penalty_scores,
      frequency_scores,
      presence_scores,
      temperatures,
      logits,
      bs,
      length);

  if (bad_words && length_bad_words > 0) {
    ban_bad_words_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
        logits,
        reinterpret_cast<const XPU_INT64*>(bad_words),
        bs,
        length,
        length_bad_words);
  }
  return api::SUCCESS;
}

template <typename T>
int token_penalty_multi_scores(Context* ctx,
                               const int64_t* pre_ids,
                               T* logits,
                               const T* penalty_scores,
                               const T* frequency_scores,
                               const T* presence_scores,
                               const float* temperatures,
                               const int64_t* cur_len,
                               const int64_t* min_len,
                               const int64_t* eos_token_id,
                               const int64_t* bad_words,
                               const int64_t bs,
                               const int64_t length,
                               const int64_t length_id,
                               const int64_t end_length,
                               const int64_t length_bad_words) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "token_penalty_multi_scores", T);
  WRAPPER_DUMP_PARAM6(ctx,
                      pre_ids,
                      logits,
                      penalty_scores,
                      frequency_scores,
                      presence_scores,
                      temperatures);
  WRAPPER_DUMP_PARAM3(ctx, cur_len, min_len, eos_token_id);
  WRAPPER_DUMP_PARAM4(ctx, bs, length, length_id, end_length);
  WRAPPER_DUMP(ctx);
  // TODO(mayang02) shape check
  int64_t pre_ids_len = -1;
  int64_t logits_len = -1;
  int64_t penalty_scores_len = -1;
  int64_t frequency_scores_len = -1;
  int64_t presence_scores_len = -1;
  int64_t temperatures_len = -1;
  int64_t cur_len_len = -1;
  int64_t min_len_len = -1;
  int64_t eos_token_id_len = -1;
  int64_t bad_words_len = -1;
  WRAPPER_CHECK_SHAPE(ctx, &pre_ids_len, {bs, length_id});
  WRAPPER_CHECK_SHAPE(ctx, &logits_len, {bs, length});
  WRAPPER_CHECK_SHAPE(ctx, &penalty_scores_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &frequency_scores_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &presence_scores_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &temperatures_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &cur_len_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &min_len_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &eos_token_id_len, {end_length});
  WRAPPER_CHECK_SHAPE(ctx, &bad_words_len, {length_bad_words});
  WRAPPER_CHECK_PTR(ctx, int64_t, pre_ids_len, pre_ids);
  WRAPPER_CHECK_PTR(ctx, T, logits_len, logits);
  WRAPPER_CHECK_PTR(ctx, T, penalty_scores_len, penalty_scores);
  WRAPPER_CHECK_PTR(ctx, T, frequency_scores_len, frequency_scores);
  WRAPPER_CHECK_PTR(ctx, T, presence_scores_len, presence_scores);
  WRAPPER_CHECK_PTR(ctx, float, temperatures_len, temperatures);
  WRAPPER_CHECK_PTR(ctx, int64_t, cur_len_len, cur_len);
  WRAPPER_CHECK_PTR(ctx, int64_t, min_len_len, min_len);
  WRAPPER_CHECK_PTR(ctx, int64_t, eos_token_id_len, eos_token_id);
  WRAPPER_CHECK_PTR(ctx, int64_t, bad_words_len, bad_words);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx,
                          pre_ids,
                          logits,
                          penalty_scores,
                          frequency_scores,
                          presence_scores,
                          temperatures,
                          cur_len,
                          min_len,
                          eos_token_id,
                          bad_words,
                          bs,
                          length,
                          length_id,
                          end_length,
                          length_bad_words);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper<T>(ctx,
                           pre_ids,
                           logits,
                           penalty_scores,
                           frequency_scores,
                           presence_scores,
                           temperatures,
                           cur_len,
                           min_len,
                           eos_token_id,
                           bad_words,
                           bs,
                           length,
                           length_id,
                           end_length,
                           length_bad_words);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int token_penalty_multi_scores<float>(Context* ctx,
                                               const int64_t* pre_ids,
                                               float* logits,
                                               const float* penalty_scores,
                                               const float* frequency_scores,
                                               const float* presence_scores,
                                               const float* temperatures,
                                               const int64_t* cur_len,
                                               const int64_t* min_len,
                                               const int64_t* eos_token_id,
                                               const int64_t* bad_words,
                                               const int64_t bs,
                                               const int64_t length,
                                               const int64_t length_id,
                                               const int64_t end_length,
                                               const int64_t length_bad_words);
template int token_penalty_multi_scores<float16>(
    Context* ctx,
    const int64_t* pre_ids,
    float16* logits,
    const float16* penalty_scores,
    const float16* frequency_scores,
    const float16* presence_scores,
    const float* temperatures,
    const int64_t* cur_len,
    const int64_t* min_len,
    const int64_t* eos_token_id,
    const int64_t* bad_words,
    const int64_t bs,
    const int64_t length,
    const int64_t length_id,
    const int64_t end_length,
    const int64_t length_bad_words);
}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
