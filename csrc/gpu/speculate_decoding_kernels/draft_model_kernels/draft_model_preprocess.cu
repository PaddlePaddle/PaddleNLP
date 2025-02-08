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
#include "paddle/extension.h"

template <int THREADBLOCK_SIZE, bool EAGLE>
__global__ void draft_model_preprocess_kernel(
    int64_t* draft_tokens,
    int64_t* input_ids,
    bool* stop_flags,
    int* seq_lens_this_time,
    int* seq_lens_encoder,
    int* seq_lens_decoder,
    int64_t* step_idx,
    int* first_token_record,
    bool* not_need_stop,
    const int64_t* accept_tokens,
    const int* accept_num,
    const int* base_model_seq_lens_encoder,
    const int* base_model_seq_lens_decoder,
    const int64_t* base_model_step_idx,
    const bool* base_model_stop_flags,
    int64_t* base_model_draft_tokens,
    const int bsz,
    const int max_draft_token,
    const int accept_tokens_len,
    const int draft_tokens_len,
    const int input_ids_len,
    const int base_model_draft_tokens_len) {
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int64_t not_stop_flag = 0;

  int tid = threadIdx.x;

  if (tid < bsz) {
    auto base_model_step_idx_now = base_model_step_idx[tid];
    auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
    auto* draft_tokens_now = draft_tokens + tid * draft_tokens_len;
    auto accept_num_now = accept_num[tid];
    auto* input_ids_now = input_ids + tid * input_ids_len;
    auto* base_model_draft_tokens_now =
        base_model_draft_tokens + tid * base_model_draft_tokens_len;
#pragma unroll
    for (int i = 1; i < base_model_draft_tokens_len; i++) {
      base_model_draft_tokens_now[i] = -1;
    }

    if (!base_model_stop_flags[tid]) {
      not_stop_flag = 1;
      // 1. first token
      if (base_model_step_idx_now == 0) {
        seq_lens_this_time[tid] = 0;
        not_stop_flag = 0;
      } else if (base_model_step_idx_now == 1 && first_token_record[tid] > 0) {
        // Can be extended to first few tokens
        seq_lens_encoder[tid] = first_token_record[tid];
        first_token_record[tid] = -1;
        stop_flags[tid] = false;
        int64_t base_model_first_token = accept_tokens_now[0];
        int position = base_model_seq_lens_decoder[tid];
        if (EAGLE) {
          input_ids_now[position - 1] = base_model_first_token;
          seq_lens_this_time[tid] = base_model_seq_lens_decoder[tid];
        } else {
          input_ids_now[position] = base_model_first_token;
          seq_lens_this_time[tid] = base_model_seq_lens_decoder[tid] + 1;
        }
      } else if (accept_num_now <=
                 max_draft_token) /*Accept partial draft tokens*/ {
        // Base Model reject stop
        if (stop_flags[tid]) {
          stop_flags[tid] = false;
          seq_lens_decoder[tid] = base_model_seq_lens_decoder[tid];
          step_idx[tid] = base_model_step_idx[tid];
        } else {
          seq_lens_decoder[tid] -= max_draft_token - accept_num_now;
          step_idx[tid] -= max_draft_token - accept_num_now;
        }
        int64_t modified_token = accept_tokens_now[accept_num_now - 1];
        draft_tokens_now[0] = modified_token;
        seq_lens_this_time[tid] = 1;

      } else /*Accept all draft tokens*/ {
        draft_tokens_now[1] = accept_tokens_now[max_draft_token];
        seq_lens_this_time[tid] = 2;
      }
    } else {
      stop_flags[tid] = true;
      seq_lens_this_time[tid] = 0;
      seq_lens_decoder[tid] = 0;
    }
  }
  __syncthreads();
  int64_t not_stop_flag_sum = BlockReduce(temp_storage).Sum(not_stop_flag);
  if (tid == 0) {
    not_need_stop[0] = not_stop_flag_sum > 0;
  }
}


void DraftModelPreprocess(const paddle::Tensor& draft_tokens,
                          const paddle::Tensor& input_ids,
                          const paddle::Tensor& stop_flags,
                          const paddle::Tensor& seq_lens_this_time,
                          const paddle::Tensor& seq_lens_encoder,
                          const paddle::Tensor& seq_lens_decoder,
                          const paddle::Tensor& step_idx,
                          const paddle::Tensor& first_token_record,
                          const paddle::Tensor& not_need_stop,
                          const paddle::Tensor& accept_tokens,
                          const paddle::Tensor& accept_num,
                          const paddle::Tensor& base_model_seq_lens_encoder,
                          const paddle::Tensor& base_model_seq_lens_decoder,
                          const paddle::Tensor& base_model_step_idx,
                          const paddle::Tensor& base_model_stop_flags,
                          const paddle::Tensor& base_model_draft_tokens,
                          const int max_draft_token,
                          const std::string& draft_type) {
  int real_bsz = seq_lens_this_time.shape()[0];
  int accept_tokens_len = accept_tokens.shape()[1];
  int input_ids_len = input_ids.shape()[1];
  int draft_tokens_len = draft_tokens.shape()[1];
  auto cu_stream = seq_lens_this_time.stream();
  constexpr int BlockSize = 256;
  int base_model_draft_tokens_len = base_model_draft_tokens.shape()[1];
  auto not_need_stop_gpu =
      not_need_stop.copy_to(seq_lens_this_time.place(), false);


  if (draft_type == "eagle") {
    draft_model_preprocess_kernel<BlockSize, true>
        <<<1, BlockSize, 0, cu_stream>>>(
            const_cast<int64_t*>(draft_tokens.data<int64_t>()),
            const_cast<int64_t*>(input_ids.data<int64_t>()),
            const_cast<bool*>(stop_flags.data<bool>()),
            const_cast<int*>(seq_lens_this_time.data<int>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_decoder.data<int>()),
            const_cast<int64_t*>(step_idx.data<int64_t>()),
            const_cast<int*>(first_token_record.data<int>()),
            const_cast<bool*>(not_need_stop_gpu.data<bool>()),
            accept_tokens.data<int64_t>(),
            accept_num.data<int>(),
            base_model_seq_lens_encoder.data<int>(),
            base_model_seq_lens_decoder.data<int>(),
            base_model_step_idx.data<int64_t>(),
            base_model_stop_flags.data<bool>(),
            const_cast<int64_t*>(base_model_draft_tokens.data<int64_t>()),
            real_bsz,
            max_draft_token,
            accept_tokens_len,
            draft_tokens_len,
            input_ids_len,
            base_model_draft_tokens_len);
  } else {
    draft_model_preprocess_kernel<BlockSize, false>
        <<<1, BlockSize, 0, cu_stream>>>(
            const_cast<int64_t*>(draft_tokens.data<int64_t>()),
            const_cast<int64_t*>(input_ids.data<int64_t>()),
            const_cast<bool*>(stop_flags.data<bool>()),
            const_cast<int*>(seq_lens_this_time.data<int>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_decoder.data<int>()),
            const_cast<int64_t*>(step_idx.data<int64_t>()),
            const_cast<int*>(first_token_record.data<int>()),
            const_cast<bool*>(not_need_stop_gpu.data<bool>()),
            accept_tokens.data<int64_t>(),
            accept_num.data<int>(),
            base_model_seq_lens_encoder.data<int>(),
            base_model_seq_lens_decoder.data<int>(),
            base_model_step_idx.data<int64_t>(),
            base_model_stop_flags.data<bool>(),
            const_cast<int64_t*>(base_model_draft_tokens.data<int64_t>()),
            real_bsz,
            max_draft_token,
            accept_tokens_len,
            draft_tokens_len,
            input_ids_len,
            base_model_draft_tokens_len);
  }


  auto not_need_stop_cpu =
      not_need_stop_gpu.copy_to(not_need_stop.place(), false);
  bool* not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}


PD_BUILD_OP(draft_model_preprocess)
    .Inputs({"draft_tokens",
             "input_ids",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "first_token_record",
             "not_need_stop",
             "accept_tokens",
             "accept_num",
             "base_model_seq_lens_encoder",
             "base_model_seq_lens_decoder",
             "base_model_step_idx",
             "base_model_stop_flags",
             "base_model_draft_tokens"})
    .Outputs({"draft_tokens_out",
              "input_ids_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_idx_out",
              "not_need_stop_out",
              "first_token_record_out"})
    .Attrs({"max_draft_token: int", "draft_type: std::string"})
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"input_ids", "input_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"step_idx", "step_idx_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"first_token_record", "first_token_record_out"}})
    .SetKernelFn(PD_KERNEL(DraftModelPreprocess));