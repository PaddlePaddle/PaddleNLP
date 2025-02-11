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


__global__ void update_pre_ids_kernel(const int64_t* draft_tokens,
                                      int64_t* pre_ids_all,
                                      const bool* stop_flags,
                                      int* seq_lens_this_time,
                                      const int64_t* step_idx,
                                      int bs,
                                      int pre_id_length,
                                      int max_draft_token) {
  int tid = threadIdx.x;
  if (tid < bs && seq_lens_this_time[tid] != 0 && !stop_flags[tid]) {
    int64_t* pre_ids_all_now = pre_ids_all + tid * pre_id_length;
    const int64_t* draft_token_now = draft_tokens + tid * max_draft_token;
    const int seq_len_this_time = seq_lens_this_time[tid];
    if (step_idx[tid] - 1 > 0 /*Decoder Step*/) {
      for (int i = 0; i < seq_len_this_time; ++i) {
        pre_ids_all_now[step_idx[tid] - i] =
            draft_token_now[seq_len_this_time - 1 - i];
      }
    } else if (step_idx[tid] == 1 /*Encoder Step*/) {
      pre_ids_all_now[1] = draft_token_now[0];
    }
    seq_lens_this_time[tid] = 1;
  }
}


void SpeculateDraftModelUpdate(const paddle::Tensor& draft_tokens,
                               const paddle::Tensor& pre_ids_all,
                               const paddle::Tensor& stop_flags,
                               const paddle::Tensor& seq_lens_this_time,
                               const paddle::Tensor& seq_lens_encoder,
                               const paddle::Tensor& seq_lens_decoder,
                               const paddle::Tensor& step_idx) {
  int64_t real_bs = seq_lens_this_time.shape()[0];
  int64_t pre_id_length = pre_ids_all.shape()[1];
  auto cu_stream = seq_lens_this_time.stream();
  int64_t max_draft_token = draft_tokens.shape()[1];

  int block_size = (real_bs + 32 - 1) / 32 * 32;
  update_pre_ids_kernel<<<1, block_size, 0, cu_stream>>>(
      draft_tokens.data<int64_t>(),
      const_cast<int64_t*>(pre_ids_all.data<int64_t>()),
      stop_flags.data<bool>(),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      step_idx.data<int64_t>(),
      real_bs,
      pre_id_length,
      max_draft_token);
}

PD_BUILD_OP(draft_model_set_value_by_flags)
    .Inputs({"draft_tokens",
             "pre_ids_all",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateDraftModelUpdate));
