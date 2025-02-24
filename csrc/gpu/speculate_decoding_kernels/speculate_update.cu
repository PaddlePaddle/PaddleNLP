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

#include "helper.h"

template <int THREADBLOCK_SIZE>
__global__ void speculate_update(int *seq_lens_encoder,
                                 int *seq_lens_decoder,
                                 bool *not_need_stop,
                                 int64_t *draft_tokens,
                                 int *actual_draft_token_nums,
                                 const int64_t *accept_tokens,
                                 const int *accept_num,
                                 const bool *stop_flags,
                                 const int *seq_lens_this_time,
                                 const bool *is_block_step,
                                 const int real_bsz,
                                 const int max_draft_tokens) {
    const int bid = threadIdx.x;
    const int accept_num_now = accept_num[bid];
    int stop_flag_now_int = 0;
    if (!(is_block_step[bid] || bid >= real_bsz)) {
        if (stop_flags[bid]) {
            stop_flag_now_int = 1;
        }
        if (seq_lens_encoder[bid] == 0) {
            seq_lens_decoder[bid] += accept_num_now;
        }

        if (seq_lens_this_time[bid] > 1 &&
            seq_lens_encoder[bid] ==
                0) {  // 对于append模式，需要根据接收与否确定是否要降低下次draft
                      // token的数量
            auto current_actual_draft_token_num = actual_draft_token_nums[bid];
            if (accept_num_now - 1 == current_actual_draft_token_num) {
                if (current_actual_draft_token_num + 2 <=
                    max_draft_tokens - 1) {
                    actual_draft_token_nums[bid] =
                        current_actual_draft_token_num + 2;
                } else if (current_actual_draft_token_num + 1 <=
                           max_draft_tokens - 1) {
                    actual_draft_token_nums[bid] =
                        current_actual_draft_token_num + 1;
                } else {
                    actual_draft_token_nums[bid] = max_draft_tokens - 1;
                }
            } else {
                actual_draft_token_nums[bid] =
                    actual_draft_token_nums[bid] - 1 >= 1
                        ? actual_draft_token_nums[bid] - 1
                        : 1;
            }
        }

        if (seq_lens_encoder[bid] != 0) {
            seq_lens_decoder[bid] += seq_lens_encoder[bid];
            seq_lens_encoder[bid] = 0;
        }
        if (!stop_flags[bid]) {
            draft_tokens[bid * max_draft_tokens] =
                accept_tokens[bid * max_draft_tokens + accept_num_now - 1];
        }
        if (stop_flag_now_int) {
            seq_lens_decoder[bid] = 0;
        }
    }
    __syncthreads();
    typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int64_t stop_sum = BlockReduce(temp_storage).Sum(stop_flag_now_int);

    if (threadIdx.x == 0) {
        not_need_stop[0] = stop_sum < real_bsz;
    }
}

void SpeculateUpdate(const paddle::Tensor &seq_lens_encoder,
                       const paddle::Tensor &seq_lens_decoder,
                       const paddle::Tensor &not_need_stop,
                       const paddle::Tensor &draft_tokens,
                       const paddle::Tensor &actual_draft_token_nums,
                       const paddle::Tensor &accept_tokens,
                       const paddle::Tensor &accept_num,
                       const paddle::Tensor &stop_flags,
                       const paddle::Tensor &seq_lens_this_time,
                       const paddle::Tensor &is_block_step) {
    int real_bsz = seq_lens_this_time.shape()[0];
    auto max_draft_tokens = draft_tokens.shape()[1];

    constexpr int BlockSize = 512;

    speculate_update<BlockSize><<<1, BlockSize, 0, accept_tokens.stream()>>>(
        const_cast<int *>(seq_lens_encoder.data<int>()),
        const_cast<int *>(seq_lens_decoder.data<int>()),
        const_cast<bool *>(not_need_stop.data<bool>()),
        const_cast<int64_t *>(draft_tokens.data<int64_t>()),
        const_cast<int *>(actual_draft_token_nums.data<int>()),
        accept_tokens.data<int64_t>(),
        accept_num.data<int>(),
        stop_flags.data<bool>(),
        seq_lens_this_time.data<int>(),
        is_block_step.data<bool>(),
        real_bsz,
        max_draft_tokens);
}

PD_BUILD_OP(speculate_update)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "not_need_stop",
             "draft_tokens",
             "actual_draft_token_nums",
             "accept_tokens",
             "accept_num",
             "stop_flags",
             "seq_lens_this_time",
             "is_block_step"})
    .Outputs({"seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "not_need_stop_out",
              "draft_tokens_out",
              "actual_draft_token_nums_out"})
    .SetInplaceMap({{"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"draft_tokens", "draft_tokens_out"},
                    {"actual_draft_token_nums", "actual_draft_token_nums_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateUpdate));