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

__device__ bool is_in_end_v3(const int64_t id, const int64_t *end_ids, int length) {
    bool flag = false;
    for (int i = 0; i < length; i++) {
        if (id == end_ids[i]) {
            return true;
        }
    }
    return flag;
}

template <int THREADBLOCK_SIZE>
__global__ void update_inputs_kernel_v2(
    bool *not_need_stop,
    int64_t *step_idx,
    bool *stop_flags,
    int *seq_lens_this_time,
    int *seq_lens_encoder,
    int *seq_lens_decoder,
    int64_t *next_tokens,
    int64_t *kwargs_next_tokens,
    int64_t *input_ids,
    const int64_t *end_ids, 
    const int64_t *stop_nums,
    const bool *is_block_step,
    const int64_t *max_dec_len,
    const int bsz,
    const int max_bsz,
    const int input_ids_stride,
    const int end_length) {
  int thread_idx = threadIdx.x;
  // update step_idx and stop_flags
  if (thread_idx < max_bsz) {
    bool stop_flag = stop_flags[thread_idx];
    if (!stop_flag) {
      step_idx[thread_idx] += 1;
    }
    if (step_idx[thread_idx] >= max_dec_len[thread_idx]) {
      stop_flags[thread_idx] = true;
    }
  }
  __syncthreads();
  // update inputs
  if (thread_idx < bsz) {
    if (stop_flags[thread_idx]) {
      if (seq_lens_this_time[thread_idx] == 0) {
        next_tokens[thread_idx] = -1;
      } else {
        next_tokens[thread_idx] = end_ids[0];
        kwargs_next_tokens[thread_idx] = end_ids[0];
      }
    } else {
      kwargs_next_tokens[thread_idx] = next_tokens[thread_idx];
    }
    if (is_in_end_v3(next_tokens[thread_idx], end_ids, end_length)) {
      stop_flags[thread_idx] = true;
    }
  }

  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  bool stop_flag_now = false;
  int64_t stop_flag_now_int = 0;
  if (thread_idx < max_bsz) {
    if (thread_idx < bsz) {
      stop_flag_now = stop_flags[thread_idx];
      if (is_block_step[thread_idx]) {
        stop_flag_now_int=0;
      } else {
        stop_flag_now_int = static_cast<int64_t>(stop_flag_now);
      }
    } else {
      stop_flag_now_int = 1;
    }
  }
  if (thread_idx < bsz) {
    const int seq_len_this_time = seq_lens_this_time[thread_idx];
    const int seq_len_encoder = seq_lens_encoder[thread_idx];
    const int seq_len_decoder = seq_lens_decoder[thread_idx];

    seq_lens_decoder[thread_idx] = stop_flag_now ? 0 : (seq_len_encoder > 0 ? (seq_len_encoder + seq_len_decoder) : seq_len_decoder + 1);

    seq_lens_this_time[thread_idx] = stop_flag_now ? 0 : 1;
    seq_lens_encoder[thread_idx] = 0;
    int64_t *input_ids_now = input_ids + thread_idx * input_ids_stride;
    input_ids_now[0] = next_tokens[thread_idx];
  }
  __syncthreads();
  int64_t stop_sum = BlockReduce(temp_storage).Sum(stop_flag_now_int);
  if (thread_idx == 0) {
    not_need_stop[0] = stop_sum < stop_nums[0];
  }
}

void UpdateInputesV2(const paddle::Tensor& stop_flags,
               const paddle::Tensor& step_idx,
               const paddle::Tensor& not_need_stop,
               const paddle::Tensor& seq_lens_this_time,
               const paddle::Tensor& seq_lens_encoder,
               const paddle::Tensor& seq_lens_decoder,
               const paddle::Tensor& max_dec_len,
               const paddle::Tensor& input_ids,
               const paddle::Tensor& stop_nums,
               const paddle::Tensor& next_tokens,
               const paddle::Tensor& is_block_step,
               const paddle::Tensor& end_ids,
               const paddle::Tensor& kwargs_next_tokens) {
  const int max_bsz = stop_flags.shape()[0];
  const int now_bsz = seq_lens_this_time.shape()[0];
  const int input_ids_stride = input_ids.shape()[1];
  const int end_length = end_ids.shape()[0];
  update_inputs_kernel_v2<1024><<<1, 1024, 0, input_ids.stream()>>>(
    const_cast<bool*>(not_need_stop.data<bool>()),
    const_cast<int64_t*>(step_idx.data<int64_t>()),
    const_cast<bool*>(stop_flags.data<bool>()),
    const_cast<int*>(seq_lens_this_time.data<int>()),
    const_cast<int*>(seq_lens_encoder.data<int>()),
    const_cast<int*>(seq_lens_decoder.data<int>()),
    const_cast<int64_t*>(next_tokens.data<int64_t>()),
    const_cast<int64_t*>(kwargs_next_tokens.data<int64_t>()),
    const_cast<int64_t*>(input_ids.data<int64_t>()),
    end_ids.data<int64_t>(),
    stop_nums.data<int64_t>(),
    is_block_step.data<bool>(),
    max_dec_len.data<int64_t>(),
    now_bsz,
    max_bsz,
    input_ids_stride,
    end_length
  );
}

PD_BUILD_OP(update_inputs_v2)
    .Inputs({"stop_flags", 
             "step_idx",
             "not_need_stop", 
             "seq_lens_this_time", 
             "seq_lens_encoder", 
             "seq_lens_decoder",
             "max_dec_len",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step",
             "end_ids",
             "kwargs_next_tokens"})
    .Outputs({"not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "input_ids_out",
              "next_tokens_out",
              "kwargs_next_tokens_out",
              "step_idx_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"input_ids", "input_ids_out"},
                    {"next_tokens", "next_tokens_out"},
                    {"kwargs_next_tokens", "kwargs_next_tokens_out"},
                    {"step_idx", "step_idx_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputesV2));