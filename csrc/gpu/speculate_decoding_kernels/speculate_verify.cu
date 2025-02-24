// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <curand_kernel.h>
#include <cstdlib>
#include <string>
#include "helper.h"

__device__ bool is_in(const int64_t *candidates,
                      const int64_t draft,
                      const int candidate_len) {
    for (int i = 0; i < candidate_len; i++) {
        if (draft == candidates[i]) {
            return true;
        }
    }
    return false;
}

static uint64_t seed = 0;
static uint64_t offset = 0;

__device__ int64_t topp_sampling_kernel(const int64_t *candidate_ids,
                                        const float *candidate_scores,
                                        curandState_t *dev_curand_states,
                                        const int candidate_len,
                                        const float topp) {
    const int tid = threadIdx.x;

    float sum_scores = 0.0f;
    float rand_top_p = curand_uniform(dev_curand_states + tid) * topp;
    for (int i = 0; i < candidate_len; i++) {
        sum_scores += candidate_scores[i];
        if (rand_top_p <= sum_scores) {
            return candidate_ids[i];
        }
    }
    return candidate_ids[0];
}

__global__ void setup_kernel(curandState_t *state,
                             const uint64_t seed,
                             const uint64_t offset,
                             const int bs,
                             const bool need_batch_random) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < bs; i += gridDim.x * blockDim.x) {
        if (need_batch_random) {
            curand_init(seed, i, offset, &state[i]);
        } else {
            curand_init(seed, 0, offset, &state[i]);
        }
    }
}

__global__ void speculate_verify(int64_t *accept_tokens,
                                 int *accept_num,
                                 int64_t *step_idx,
                                 bool *stop_flags,
                                 const int *seq_lens_encoder,
                                 const int *seq_lens_decoder,
                                 const int64_t *draft_tokens,
                                 const int *actual_draft_token_nums,
                                 curandState_t *dev_curand_states,
                                 const float *topp,
                                 const int *seq_lens_this_time,
                                 const int64_t *verify_tokens,
                                 const float *verify_scores,
                                 const int64_t *max_dec_len,
                                 const int64_t *end_tokens,
                                 const bool *is_block_step,
                                 const int *output_cum_offsets,
                                 const int *actual_candidate_len,
                                 const int real_bsz,
                                 const int max_draft_tokens,
                                 const int end_length,
                                 const int max_seq_len,
                                 const int max_candidate_len,
                                 const int verify_window) {
    const int bid = threadIdx.x;
    const int start_token_id = bid * max_seq_len - output_cum_offsets[bid];
    int accept_num_now = 1;
    int stop_flag_now_int = 0;

    if (!(is_block_step[bid] || bid >= real_bsz)) {
        if (stop_flags[bid]) {
            stop_flag_now_int = 1;
        // 这里 prefill 阶段也会进入，但是因为 draft tokens 会置零，因此会直接到最后的采样阶段
        } else {
            auto *verify_tokens_now =
                verify_tokens + start_token_id * max_candidate_len;
            auto *draft_tokens_now = draft_tokens + bid * max_draft_tokens;
            auto *actual_candidate_len_now =
                actual_candidate_len + start_token_id;

            int i = 0;
            if (seq_lens_encoder[bid] == 0) {
                for (; i < seq_lens_this_time[bid] - 1; i++) {
                    if (verify_tokens_now[i * max_candidate_len] == draft_tokens_now[i + 1]) {
                        step_idx[bid]++;
                        auto accept_token = draft_tokens_now[i + 1];
                        accept_tokens[bid * max_draft_tokens + i] =
                            accept_token;
                        if (is_in_end(accept_token, end_tokens, end_length) ||
                            step_idx[bid] >= max_dec_len[bid]) {
                            stop_flags[bid] = true;
                            stop_flag_now_int = 1;
                            if (step_idx[bid] >= max_dec_len[bid])
                                accept_tokens[bid * max_draft_tokens + i] =
                                    end_tokens[0];
                            break;
                        } else {
                            accept_num_now++;
                        }
                    } else {
                        break;
                    }
                }
            }
            // sampling 阶段
            // 第一种，draft_token[i+1]被拒绝，需要从 verify_tokens_now[i] 中选一个
            // 第二种，i == seq_lens_this_time[bid]-1,
            // 也是从verify_tokens_now[i]中选一个 但是停止的情况不算
            if (!stop_flag_now_int) {
                int64_t accept_token;
                const float *verify_scores_now =
                    verify_scores + start_token_id * max_candidate_len;
                step_idx[bid]++;
                // sampling
                auto actual_candidate_len_value =
                    actual_candidate_len_now[i] > max_candidate_len
                        ? max_candidate_len
                        : actual_candidate_len_now[i];

                accept_token = topp_sampling_kernel(
                    verify_tokens_now + i * max_candidate_len,
                    verify_scores_now + i * max_candidate_len,
                    dev_curand_states,
                    actual_candidate_len_value,
                    topp[bid]);

                accept_tokens[bid * max_draft_tokens + i] = accept_token;
                if (is_in_end(accept_token, end_tokens, end_length) ||
                    step_idx[bid] >= max_dec_len[bid]) {
                    stop_flags[bid] = true;
                    stop_flag_now_int = 1;
                    if (step_idx[bid] >= max_dec_len[bid])
                        accept_tokens[bid * max_draft_tokens + i] =
                            end_tokens[0];
                }
            }
            accept_num[bid] = accept_num_now;
        }
    }
}

void SpeculateVerify(const paddle::Tensor &accept_tokens,
                     const paddle::Tensor &accept_num,
                     const paddle::Tensor &step_idx,
                     const paddle::Tensor &stop_flags,
                     const paddle::Tensor &seq_lens_encoder,
                     const paddle::Tensor &seq_lens_decoder,
                     const paddle::Tensor &draft_tokens,
                     const paddle::Tensor &seq_lens_this_time,
                     const paddle::Tensor &verify_tokens,
                     const paddle::Tensor &verify_scores,
                     const paddle::Tensor &max_dec_len,
                     const paddle::Tensor &end_tokens,
                     const paddle::Tensor &is_block_step,
                     const paddle::Tensor &output_cum_offsets,
                     const paddle::Tensor &actual_candidate_len,
                     const paddle::Tensor &actual_draft_token_nums,
                     const paddle::Tensor &topp,
                     int max_seq_len,
                     int verify_window) {
    //   printf("Enter speculate update\n");
    auto bsz = accept_tokens.shape()[0];
    int real_bsz = seq_lens_this_time.shape()[0];
    auto max_draft_tokens = draft_tokens.shape()[1];
    auto end_length = end_tokens.shape()[0];
    auto max_candidate_len = verify_tokens.shape()[1];

    constexpr int BlockSize = 512;

    curandState_t *dev_curand_states;
    cudaMalloc(&dev_curand_states, sizeof(curandState_t) * bsz);
    setup_kernel<<<1, BlockSize, 0, accept_tokens.stream()>>>(
        dev_curand_states, seed, offset, bsz, true);
    seed++;
    offset++;

    speculate_verify<<<1, BlockSize, 0, accept_tokens.stream()>>>(
            const_cast<int64_t *>(accept_tokens.data<int64_t>()),
            const_cast<int *>(accept_num.data<int>()),
            const_cast<int64_t *>(step_idx.data<int64_t>()),
            const_cast<bool *>(stop_flags.data<bool>()),
            seq_lens_encoder.data<int>(),
            seq_lens_decoder.data<int>(),
            draft_tokens.data<int64_t>(),
            actual_draft_token_nums.data<int>(),
            dev_curand_states,
            topp.data<float>(),
            seq_lens_this_time.data<int>(),
            verify_tokens.data<int64_t>(),
            verify_scores.data<float>(),
            max_dec_len.data<int64_t>(),
            end_tokens.data<int64_t>(),
            is_block_step.data<bool>(),
            output_cum_offsets.data<int>(),
            actual_candidate_len.data<int>(),
            real_bsz,
            max_draft_tokens,
            end_length,
            max_seq_len,
            max_candidate_len,
            verify_window);


    cudaFree(dev_curand_states);
}

PD_BUILD_OP(speculate_verify)
    .Inputs({"accept_tokens",
             "accept_num",
             "step_idx",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "stop_flags",
             "draft_tokens",
             "seq_lens_this_time",
             "verify_tokens",
             "verify_scores",
             "max_dec_len",
             "end_tokens",
             "is_block_step",
             "output_cum_offsets",
             "actual_candidate_len",
             "actual_draft_token_nums",
             "topp"})
    .Outputs({"accept_tokens_out",
              "accept_num_out",
              "step_idx_out",
              "stop_flags_out"})
    .Attrs({"max_seq_len: int", "verify_window: int", "enable_topp: bool"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"accept_num", "accept_num_out"},
                    {"step_idx", "step_idx_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateVerify));