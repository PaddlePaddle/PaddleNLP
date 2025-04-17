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

#include "paddle/extension.h"


__global__ void draft_model_update_seq_lens_this_time_kernel(
    const int64_t* base_model_draft_tokens,
    int* base_model_seq_lens_this_time,
    const int* base_model_seq_lens_encoder,
    const bool* base_model_stop_flags,
    int bsz,
    int base_model_draft_token_len) {
  int tid = threadIdx.x;
  if (tid < bsz) {
    if (!base_model_stop_flags[tid] && base_model_seq_lens_encoder[tid] == 0) {
      const int64_t* base_model_draft_tokens_now =
          base_model_draft_tokens + tid * base_model_draft_token_len;
      int token_num = 0;

      for (int i = 0; i < base_model_draft_token_len; ++i) {
        if (base_model_draft_tokens_now[i] != -1) {
          token_num++;
        }
      }
      base_model_seq_lens_this_time[tid] = token_num;
    } else if (base_model_stop_flags[tid]) {
      base_model_seq_lens_this_time[tid] = 0;
    }
  }
}


void DraftModelPostprocess(const paddle::Tensor& base_model_draft_tokens,
                           const paddle::Tensor& base_model_seq_lens_this_time,
                           const paddle::Tensor& base_model_seq_lens_encoder,
                           const paddle::Tensor& base_model_stop_flags) {
  int real_bsz = base_model_seq_lens_this_time.shape()[0];
  auto cu_stream = base_model_seq_lens_this_time.stream();
  constexpr int BlockSize = 512;
  int base_model_draft_token_len = base_model_draft_tokens.shape()[1];
  draft_model_update_seq_lens_this_time_kernel<<<1, BlockSize, 0, cu_stream>>>(
      base_model_draft_tokens.data<int64_t>(),
      const_cast<int*>(base_model_seq_lens_this_time.data<int>()),
      base_model_seq_lens_encoder.data<int>(),
      base_model_stop_flags.data<bool>(),
      real_bsz,
      base_model_draft_token_len);
}


PD_BUILD_OP(draft_model_postprocess)
    .Inputs({"base_model_draft_tokens",
             "base_model_seq_lens_this_time",
             "base_model_seq_lens_encoder",
             "base_model_stop_flags"})
    .Outputs({"base_model_draft_tokens_out",
              "base_model_seq_lens_this_time_out",
              "base_model_stop_flags_out"})
    .SetInplaceMap({{"base_model_draft_tokens", "base_model_draft_tokens_out"},
                    {"base_model_seq_lens_this_time",
                     "base_model_seq_lens_this_time_out"},
                    {"base_model_stop_flags", "base_model_stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(DraftModelPostprocess));