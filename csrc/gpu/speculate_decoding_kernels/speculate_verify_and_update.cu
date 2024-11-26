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

#include "helper.h"
#include <curand_kernel.h>
#include <cstdlib>
#include <string>

__device__ bool is_in(const int64_t* candidates, const int64_t draft, const int candidate_len) {
    for (int i = 0; i < candidate_len; i++) {
        if (draft == candidates[i]) {
            return true;
        }
    }
    return false;
}

static uint64_t seed = 0;
static uint64_t offset = 0;

__device__ int64_t topp_sampling_kernel(const int64_t* candidate_ids,
        const float* candidate_scores,
        curandState_t* dev_curand_states,
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

__global__ void setup_kernel(curandState_t* state,
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

template <int THREADBLOCK_SIZE, bool ENABLE_TOPP, bool USE_TOPK>
__global__ void speculate_verify_and_update_kernel(int64_t* accept_tokens,
        int* accept_num,
        int64_t* step_idx,
        int* seq_lens_encoder,
        int* seq_lens_decoder,
        bool* stop_flags,
        bool* not_need_stop,
        int64_t* draft_tokens,
        int* actual_draft_token_nums,
        curandState_t* dev_curand_states,
        const float* topp,
        const int* seq_lens_this_time,
        const int64_t* verify_tokens,
        const float* verify_scores,
        const int64_t* max_dec_len,
        const int64_t* end_tokens,
        const bool* is_block_step,
        const int* output_cum_offsets,
        const int* actual_candidate_len,
        const int real_bsz,
        const int max_draft_tokens,
        const int end_length,
        const int max_seq_len,
        const int max_candidate_len,
        const int verify_window) {
    const int bid = threadIdx.x;
    // start token's id of bid batch
    const int start_token_id = bid * max_seq_len - output_cum_offsets[bid];
    // verify and set stop flags
    int accept_num_now = 1;
    int stop_flag_now_int = 0;

    if (!(is_block_step[bid] || bid >= real_bsz)) {

        if (stop_flags[bid]) {
            stop_flag_now_int = 1;
        } else { // Here the prefill stage also goes in, but since the draft tokens are zero in prefill stage, it goes straight to the final sampling stage.
            auto* verify_tokens_now = verify_tokens + start_token_id * max_candidate_len;
            auto* draft_tokens_now = draft_tokens + bid * max_draft_tokens;
            auto* actual_candidate_len_now = actual_candidate_len + start_token_id;

            int i = 0;
            for (; i < seq_lens_this_time[bid] - 1; i++) {
                if (USE_TOPK) {
                    if (verify_tokens_now[i * max_candidate_len] == draft_tokens_now[i + 1]) {
                        accept_num_now++;
                        step_idx[bid]++;
                        auto accept_token = draft_tokens_now[i + 1];
                        accept_tokens[bid * max_draft_tokens + i] = accept_token;
                        if (is_in_end(accept_token, end_tokens, end_length) || step_idx[bid] >= max_dec_len[bid]) {
                            stop_flags[bid] = true;
                            stop_flag_now_int = 1;
                            if (step_idx[bid] >= max_dec_len[bid])
                                accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    auto actual_candidate_len_value = actual_candidate_len_now[i] > max_candidate_len
                            ? max_candidate_len
                            : actual_candidate_len_now[i];
                    if (is_in(verify_tokens_now + i * max_candidate_len,
                                draft_tokens_now[i + 1],
                                actual_candidate_len_value)) {
                        // Top P verify
                        accept_num_now++;
                        step_idx[bid]++;
                        auto accept_token = draft_tokens_now[i + 1];
                        accept_tokens[bid * max_draft_tokens + i] = accept_token;
                        if (is_in_end(accept_token, end_tokens, end_length) || step_idx[bid] >= max_dec_len[bid]) {
                            stop_flags[bid] = true;
                            stop_flag_now_int = 1;
                            if (step_idx[bid] >= max_dec_len[bid])
                                accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
                            break;
                        }
                    } else {
                        // TopK verify
                        int ii = i;
                        if (max_candidate_len >= 2 &&
                                verify_tokens_now[ii * max_candidate_len + 1] == draft_tokens_now[ii + 1]) { // top-2
                            int j = 0;
                            ii += 1;
                            for (; j < verify_window && ii < seq_lens_this_time[bid] - 1; j++, ii++) {
                                if (verify_tokens_now[ii * max_candidate_len] != draft_tokens_now[ii + 1]) {
                                    break;
                                }
                            }
                            if (j >= verify_window) { // accept all
                                accept_num_now += verify_window + 1;
                                step_idx[bid] += verify_window + 1;
                                for (; i < ii; i++) {
                                    auto accept_token = draft_tokens_now[i + 1];
                                    accept_tokens[bid * max_draft_tokens + i] = accept_token;
                                    if (is_in_end(accept_token, end_tokens, end_length) ||
                                            step_idx[bid] >= max_dec_len[bid]) {
                                        stop_flags[bid] = true;
                                        stop_flag_now_int = 1;
                                        if (step_idx[bid] >= max_dec_len[bid])
                                            accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
                                        break;
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
            }

            if (!stop_flag_now_int) {
                int64_t accept_token;
                const float* verify_scores_now = verify_scores + start_token_id * max_candidate_len;
                if (ENABLE_TOPP) {
                    auto actual_candidate_len_value = actual_candidate_len_now[i] > max_candidate_len
                            ? max_candidate_len
                            : actual_candidate_len_now[i];
                    accept_token = topp_sampling_kernel(verify_tokens_now + i * max_candidate_len,
                            verify_scores_now + i * max_candidate_len,
                            dev_curand_states,
                            actual_candidate_len_value,
                            topp[bid]);
                } else {
                    accept_token = verify_tokens_now[i * max_candidate_len];
                }
                accept_tokens[bid * max_draft_tokens + i] = accept_token;
                if (is_in_end(accept_token, end_tokens, end_length) || step_idx[bid] >= max_dec_len[bid]) {
                    stop_flags[bid] = true;
                    stop_flag_now_int = 1;
                    if (step_idx[bid] >= max_dec_len[bid])
                        accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
                }
                step_idx[bid]++;
            }

            seq_lens_decoder[bid] += accept_num_now;

            // For append mode, determine whether to reduce the number of draft tokens depending on whether they are received or not.
            if (seq_lens_this_time[bid] > 1 && seq_lens_encoder[bid] == 0) {
                auto current_actual_draft_token_num = actual_draft_token_nums[bid];
                if (accept_num_now - 1 == current_actual_draft_token_num) {
                    if (current_actual_draft_token_num + 2 <= max_draft_tokens - 1) {
                        actual_draft_token_nums[bid] = current_actual_draft_token_num + 2;
                    } else if (current_actual_draft_token_num + 1 <= max_draft_tokens - 1) {
                        actual_draft_token_nums[bid] = current_actual_draft_token_num + 1;
                    } else {
                        actual_draft_token_nums[bid] = max_draft_tokens - 1;
                    }
                } else {
                    actual_draft_token_nums[bid] =
                            actual_draft_token_nums[bid] - 1 >= 1 ? actual_draft_token_nums[bid] - 1 : 1;
                }
            }

            if (seq_lens_encoder[bid] != 0) {
                seq_lens_decoder[bid] = seq_lens_encoder[bid];
                seq_lens_encoder[bid] = 0;
            }

            accept_num[bid] = accept_num_now;
            draft_tokens[bid * max_draft_tokens] = accept_tokens[bid * max_draft_tokens + accept_num_now - 1];
        }
    }
    if (stop_flag_now_int) {
        seq_lens_decoder[bid] = 0;
    }

    __syncthreads();
    typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int64_t stop_sum = BlockReduce(temp_storage).Sum(stop_flag_now_int);

    if (threadIdx.x == 0) {
        not_need_stop[0] = stop_sum < real_bsz;
    }
}

void SpeculateVerifyAndUpdate(const paddle::Tensor& accept_tokens,
        const paddle::Tensor& accept_num,
        const paddle::Tensor& step_idx,
        const paddle::Tensor& seq_lens_encoder,
        const paddle::Tensor& seq_lens_decoder,
        const paddle::Tensor& stop_flags,
        const paddle::Tensor& not_need_stop,
        const paddle::Tensor& draft_tokens,
        const paddle::Tensor& seq_lens_this_time,
        const paddle::Tensor& verify_tokens,
        const paddle::Tensor& verify_scores,
        const paddle::Tensor& max_dec_len,
        const paddle::Tensor& end_tokens,
        const paddle::Tensor& is_block_step,
        const paddle::Tensor& output_cum_offsets,
        const paddle::Tensor& actual_candidate_len,
        const paddle::Tensor& actual_draft_token_nums,
        const paddle::Tensor& topp,
        int max_seq_len,
        int verify_window,
        bool enable_topp) {
    auto bsz = accept_tokens.shape()[0];
    int real_bsz = seq_lens_this_time.shape()[0];
    auto max_draft_tokens = draft_tokens.shape()[1];
    auto end_length = end_tokens.shape()[0];
    auto max_candidate_len = verify_tokens.shape()[1];

    constexpr int BlockSize = 512;

    curandState_t* dev_curand_states;
    cudaMalloc(&dev_curand_states, sizeof(curandState_t) * bsz);
    setup_kernel<<<1, BlockSize, 0, accept_tokens.stream()>>>(dev_curand_states, seed, offset, bsz, true);
    seed++;
    offset++;

    auto err = cudaDeviceSynchronize();
    if (err != 0) {
        printf("err %d\n", err);
    }

    err = cudaGetLastError();

    if (err != 0) {
        printf("err %d\n", err);
    }

    bool use_topk = false;
    char* env_var = getenv("SPECULATE_VERIFY_USE_TOPK");
    if (env_var) {
        use_topk = (bool)std::stoi(env_var);
    }
    if (use_topk) {
        if (enable_topp) {
            speculate_verify_and_update_kernel<BlockSize, true, true>
                    <<<1, BlockSize, 0, accept_tokens.stream()>>>(const_cast<int64_t*>(accept_tokens.data<int64_t>()),
                            const_cast<int*>(accept_num.data<int>()),
                            const_cast<int64_t*>(step_idx.data<int64_t>()),
                            const_cast<int*>(seq_lens_encoder.data<int>()),
                            const_cast<int*>(seq_lens_decoder.data<int>()),
                            const_cast<bool*>(stop_flags.data<bool>()),
                            const_cast<bool*>(not_need_stop.data<bool>()),
                            const_cast<int64_t*>(draft_tokens.data<int64_t>()),
                            const_cast<int*>(actual_draft_token_nums.data<int>()),
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
        } else {
            speculate_verify_and_update_kernel<BlockSize, false, true>
                    <<<1, BlockSize, 0, accept_tokens.stream()>>>(const_cast<int64_t*>(accept_tokens.data<int64_t>()),
                            const_cast<int*>(accept_num.data<int>()),
                            const_cast<int64_t*>(step_idx.data<int64_t>()),
                            const_cast<int*>(seq_lens_encoder.data<int>()),
                            const_cast<int*>(seq_lens_decoder.data<int>()),
                            const_cast<bool*>(stop_flags.data<bool>()),
                            const_cast<bool*>(not_need_stop.data<bool>()),
                            const_cast<int64_t*>(draft_tokens.data<int64_t>()),
                            const_cast<int*>(actual_draft_token_nums.data<int>()),
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
        }
    } else {
        if (enable_topp) {
            speculate_verify_and_update_kernel<BlockSize, true, false>
                    <<<1, BlockSize, 0, accept_tokens.stream()>>>(const_cast<int64_t*>(accept_tokens.data<int64_t>()),
                            const_cast<int*>(accept_num.data<int>()),
                            const_cast<int64_t*>(step_idx.data<int64_t>()),
                            const_cast<int*>(seq_lens_encoder.data<int>()),
                            const_cast<int*>(seq_lens_decoder.data<int>()),
                            const_cast<bool*>(stop_flags.data<bool>()),
                            const_cast<bool*>(not_need_stop.data<bool>()),
                            const_cast<int64_t*>(draft_tokens.data<int64_t>()),
                            const_cast<int*>(actual_draft_token_nums.data<int>()),
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
        } else {
            speculate_verify_and_update_kernel<BlockSize, false, false>
                    <<<1, BlockSize, 0, accept_tokens.stream()>>>(const_cast<int64_t*>(accept_tokens.data<int64_t>()),
                            const_cast<int*>(accept_num.data<int>()),
                            const_cast<int64_t*>(step_idx.data<int64_t>()),
                            const_cast<int*>(seq_lens_encoder.data<int>()),
                            const_cast<int*>(seq_lens_decoder.data<int>()),
                            const_cast<bool*>(stop_flags.data<bool>()),
                            const_cast<bool*>(not_need_stop.data<bool>()),
                            const_cast<int64_t*>(draft_tokens.data<int64_t>()),
                            const_cast<int*>(actual_draft_token_nums.data<int>()),
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
        }
    }

    cudaFree(dev_curand_states);
}

PD_BUILD_OP(speculate_verify_and_update)
        .Inputs({"accept_tokens",
                "accept_num",
                "step_idx",
                "seq_lens_encoder",
                "seq_lens_decoder",
                "stop_flags",
                "not_need_stop",
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
                "seq_lens_encoder_out",
                "seq_lens_decoder_out",
                "stop_flags_out",
                "not_need_stop_out",
                "draft_tokens_out"})
        .Attrs({"max_seq_len: int", "verify_window: int", "enable_topp: bool"})
        .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                {"accept_num", "accept_num_out"},
                {"step_idx", "step_idx_out"},
                {"seq_lens_encoder", "seq_lens_encoder_out"},
                {"seq_lens_decoder", "seq_lens_decoder_out"},
                {"stop_flags", "stop_flags_out"},
                {"not_need_stop", "not_need_stop_out"},
                {"draft_tokens", "draft_tokens_out"}})
        .SetKernelFn(PD_KERNEL(SpeculateVerifyAndUpdate));