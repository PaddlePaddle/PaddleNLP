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

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "paddle/extension.h"

int sum(const int *value, int num) {
    int sum_value = 0;
    for (int i = 0; i <= num; i++) {
        sum_value += value[i];
    }
    return sum_value;
}

void find_candidate_pred_tokens(const int64_t *input_ids,
        const int64_t *input_ids_len,
        const int64_t *pre_ids,
        const int64_t *step_idx,
        const int *draft_token_num,
        int64_t *draft_tokens,
        int32_t *seq_lens_this_time,
        int32_t *seq_lens_encoder,
        int32_t *seq_lens_decoder,
        int64_t input_ids_stride,
        int64_t pre_ids_stride,
        int64_t draft_tokens_stride,
        const int real_batch_size,
        int max_ngram_size = 3,
        int max_draft_tokens = 10) {
    int threshold = 128;
    char *env_var = getenv("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD");
    if (env_var) {
        threshold = std::stoi(env_var);
    }
    bool is_insert = false;
    for (int batch_idx = 0; batch_idx < real_batch_size; batch_idx++) {
        if (seq_lens_encoder[batch_idx] > 0) {
            is_insert = true;
        }
    }
    for (int batch_idx = 0; batch_idx < real_batch_size; batch_idx++) {
        max_draft_tokens = draft_token_num[batch_idx];
        // int local_draft_tokens = max_draft_tokens;
        if (seq_lens_encoder[batch_idx] > 0) {
            continue;
        } else if (seq_lens_decoder[batch_idx] == 0) {
            seq_lens_this_time[batch_idx] = 0;
            continue;
        }
        const int64_t *cur_input_ids = input_ids + batch_idx * input_ids_stride;
        int64_t *cur_draft_tokens = draft_tokens + batch_idx * draft_tokens_stride;
        const int64_t *cur_pre_ids = pre_ids + batch_idx * pre_ids_stride;
        const int64_t cur_step_idx = step_idx[batch_idx];
        const int64_t cur_input_ids_len = input_ids_len[batch_idx];
        seq_lens_this_time[batch_idx] = 1;
        if (!is_insert) {
            auto sum_token_num = sum(seq_lens_this_time, batch_idx);
            int left_min_token_num = real_batch_size - batch_idx;

            if (sum_token_num + max_draft_tokens + left_min_token_num > threshold) {
                int tmp_max_draft_tokens = threshold - sum_token_num - left_min_token_num;
                max_draft_tokens = tmp_max_draft_tokens < max_draft_tokens ? tmp_max_draft_tokens : max_draft_tokens;
            }

            if (sum_token_num + left_min_token_num >= threshold - 1) {
                continue;
            }
        }


        for (int ngram_size = max_ngram_size; ngram_size > 0; --ngram_size) {
            // Extract the last n tokens as our search ngram
            if (cur_step_idx < ngram_size) {
                continue;
            }
            const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);
#ifdef _DEBUG
            if (batch_idx == 0) {
                for (int mm = 0; mm < ngram_size; mm++) {
                    printf("idx %d: %lld\n", mm, ngram[mm]);
                }
            }
            printf("cur_input_ids_len %d\n", cur_input_ids_len);
#endif
            // Iterate through sliding windows of size ngram_size
            bool match_input = false;
            for (int64_t i = 0; i <= cur_input_ids_len - ngram_size; ++i) {
                // Check if the current window matches the ngram
                bool match = true;
                for (int j = 0; j < ngram_size; j++) {
                    if (ngram[j] != cur_input_ids[i + j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    int64_t start_idx = i + ngram_size;
                    int64_t end_idx = std::min(start_idx + max_draft_tokens, cur_input_ids_len);
                    if (start_idx >= end_idx)
                        continue;
#ifdef _DEBUG
                    printf("batch_idx:%d. ngram_size:%d. idx:%lld. \n", batch_idx, ngram_size, i);
                    printf("start:%d. end:%d.\n", start_idx, end_idx);
#endif
                    // Ensure we don't go beyond the length of input_ids and avoid self-match
                    // if (end_idx <= cur_input_ids_len && start_idx < cur_input_ids_len - ngram_size) {
                    // Return a pointer to the next num_pred_tokens
                    int64_t cur_draft_token_num = end_idx - start_idx;

                    seq_lens_this_time[batch_idx] = cur_draft_token_num + 1;
                    memcpy(cur_draft_tokens + 1, cur_input_ids + start_idx, sizeof(int64_t) * cur_draft_token_num);
                    // To break the current batch_idx for-loop
                    ngram_size = 0;
                    match_input = true;
                    break;
                    // }
                }
            }
            if (!match_input) {
#ifdef _DEBUG
                printf("match_input is false so match output\n");
#endif
                for (int64_t i = 0; i <= cur_step_idx - ngram_size; ++i) {
                    // Check if the current window matches the ngram
                    bool match = true;
#ifdef _DEBUG
                    printf("search %d token in pre_ids\n", i);
#endif
                    for (int j = 0; j < ngram_size; j++) {
                        if (ngram[j] != cur_pre_ids[i + j]) {
                            match = false;
                            break;
                        }
                    }

                    if (match) {
#ifdef _DEBUG
                        printf("%d token in pre_ids matched\n", i);
#endif
                        int64_t start_idx = i + ngram_size;
                        int64_t end_idx = std::min(start_idx + max_draft_tokens, cur_step_idx);
                        int64_t cur_draft_token_num = end_idx - start_idx;
                        if (start_idx >= end_idx)
                            continue;

#ifdef _DEBUG
                        printf("cur_step_idx %d, start_idx %lld, end_idx %lld, cur_draft_token_num is %lld\n",
                                cur_step_idx,
                                start_idx,
                                end_idx,
                                cur_draft_token_num);
#endif

                        seq_lens_this_time[batch_idx] = cur_draft_token_num + 1;
                        memcpy(cur_draft_tokens + 1, cur_pre_ids + start_idx, sizeof(int64_t) * cur_draft_token_num);
                        // To break the current batch_idx for-loop
                        ngram_size = 0;
                        break;
                    }
                }
            }
        }
    }
}

void NgramMatch(const paddle::Tensor &input_ids,
        const paddle::Tensor &input_ids_len,
        const paddle::Tensor &pre_ids,
        const paddle::Tensor &step_idx,
        const paddle::Tensor &draft_token_num,
        const paddle::Tensor &draft_tokens,
        const paddle::Tensor &seq_lens_this_time,
        const paddle::Tensor &seq_lens_encoder,
        const paddle::Tensor &seq_lens_decoder,
        const int real_batch_size,
        const int max_ngram_size,
        const int max_draft_tokens) {

    auto input_ids_shape = input_ids.shape();
    const int64_t input_ids_stride = input_ids_shape[1];

    auto pre_ids_shape = pre_ids.shape();
    const int64_t pre_ids_stride = pre_ids_shape[1];

    auto draft_tokens_shape = draft_tokens.shape();
    const int64_t draft_tokens_stride = draft_tokens_shape[1];

    find_candidate_pred_tokens(input_ids.data<int64_t>(),
            input_ids_len.data<int64_t>(),
            pre_ids.data<int64_t>(),
            step_idx.data<int64_t>(),
            draft_token_num.data<int>(),
            const_cast<int64_t *>(draft_tokens.data<int64_t>()),
            const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
            const_cast<int32_t *>(seq_lens_encoder.data<int32_t>()),
            const_cast<int32_t *>(seq_lens_decoder.data<int32_t>()),
            input_ids_stride,
            pre_ids_stride,
            draft_tokens_stride,
            real_batch_size,
            max_ngram_size,
            max_draft_tokens);
}

PD_BUILD_OP(ngram_match)
        .Inputs({"input_ids",
                "input_ids_len",
                "pre_ids",
                "step_idx",
                "draft_token_num",
                "draft_tokens",
                "seq_lens_this_time",
                "seq_lens_encoder",
                "seq_lens_decoder"})
        .Attrs({"real_batch_size: int", "max_ngram_size: int", "max_draft_tokens: int"})
        .Outputs({"draft_tokens_out", "seq_lens_this_time_out"})
        .SetKernelFn(PD_KERNEL(NgramMatch))
        .SetInplaceMap({{"draft_tokens", "draft_tokens_out"}, {"seq_lens_this_time", "seq_lens_this_time_out"}});
