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

__device__ bool is_in_end(const int64_t id,
                          const int64_t *end_ids,
                          int length) {
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return false;
}

template <int THREADBLOCK_SIZE>
__global__ void fused_update_inputs_kernel(bool *not_need_stop,
                                           int *seq_lens_this_time,
                                           int *seq_lens_encoder,
                                           int *seq_lens_decoder,
                                           int64_t *input_ids,
                                           const int64_t *stop_nums,
                                           bool *stop_flags,
                                           const bool *is_block_step,
                                           int64_t *next_tokens,
                                           int64_t *topk_ids,
                                           const int64_t *end_ids,
                                           const int bsz,
                                           const int max_bsz,
                                           const int input_ids_stride,
                                           const int end_length) {
  int thread_idx = threadIdx.x;
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  bool stop_flag_now = false;
  int64_t stop_flag_now_int = 0;

  if (thread_idx < max_bsz) {
    if (thread_idx < bsz) {
      // Begin merging set_value_by_flags_v2 logic
      if (stop_flags[thread_idx]) {
        if (seq_lens_decoder[thread_idx] == 0) {
          topk_ids[thread_idx] = -1;
        } else {
          topk_ids[thread_idx] = end_ids[0];
          next_tokens[thread_idx] = end_ids[0];
        }
      } else {
        next_tokens[thread_idx] = topk_ids[thread_idx];
      }

      if (is_in_end(topk_ids[thread_idx], end_ids, end_length)) {
        stop_flags[thread_idx] = true;
      }

      // Continue with update_inputs_kernel logic
      stop_flag_now = stop_flags[thread_idx];

      if (is_block_step[thread_idx]) {
        stop_flag_now_int = 0;
      } else {
        stop_flag_now_int = static_cast<int64_t>(stop_flag_now);
      }
    } else {
      stop_flag_now_int = 1;
    }
  }

  __syncthreads();

  if (thread_idx < bsz) {
    const int seq_len_this_time = seq_lens_this_time[thread_idx];
    const int seq_len_encoder = seq_lens_encoder[thread_idx];
    const int seq_len_decoder = seq_lens_decoder[thread_idx];

    // seq_lens_decoder[thread_idx] = stop_flag_now
    //     ? 0
    //     : (seq_len_decoder == 0 ? seq_len_encoder : seq_len_decoder + 1);
    seq_lens_decoder[thread_idx] =
        stop_flag_now
            ? 0
            : (seq_len_encoder > 0 ? (seq_len_encoder + seq_len_decoder)
                                   : seq_len_decoder + 1);

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

void FusedUpdateInputs(const paddle::Tensor &stop_flags,
                       const paddle::Tensor &not_need_stop,
                       const paddle::Tensor &seq_lens_this_time,
                       const paddle::Tensor &seq_lens_encoder,
                       const paddle::Tensor &seq_lens_decoder,
                       const paddle::Tensor &input_ids,
                       const paddle::Tensor &stop_nums,
                       const paddle::Tensor &next_tokens,
                       const paddle::Tensor &is_block_step,
                       const paddle::Tensor &topk_ids,
                       const paddle::Tensor &end_ids) {
  const int max_bsz = stop_flags.shape()[0];
  const int now_bsz = seq_lens_this_time.shape()[0];
  const int input_ids_stride = input_ids.shape()[1];
  const int64_t end_length = end_ids.shape()[0];

  int threads_per_block = 1024;
  int blocks_per_grid = (max_bsz + threads_per_block - 1) / threads_per_block;

  fused_update_inputs_kernel<1024>
      <<<blocks_per_grid, threads_per_block, 0, input_ids.stream()>>>(
          const_cast<bool *>(not_need_stop.data<bool>()),
          const_cast<int *>(seq_lens_this_time.data<int>()),
          const_cast<int *>(seq_lens_encoder.data<int>()),
          const_cast<int *>(seq_lens_decoder.data<int>()),
          const_cast<int64_t *>(input_ids.data<int64_t>()),
          stop_nums.data<int64_t>(),
          const_cast<bool *>(stop_flags.data<bool>()),
          is_block_step.data<bool>(),
          const_cast<int64_t *>(next_tokens.data<int64_t>()),
          const_cast<int64_t *>(topk_ids.data<int64_t>()),
          end_ids.data<int64_t>(),
          now_bsz,
          max_bsz,
          input_ids_stride,
          end_length);
}

PD_BUILD_OP(fused_update_inputs)
    .Inputs({"stop_flags",
             "not_need_stop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step",
             "topk_ids",
             "end_ids"})
    .Outputs({"not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "input_ids_out",
              "stop_flags_out",
              "next_tokens_out",
              "topk_ids_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"input_ids", "input_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"},
                    {"topk_ids", "topk_ids_out"}})
    .SetKernelFn(PD_KERNEL(FusedUpdateInputs));
