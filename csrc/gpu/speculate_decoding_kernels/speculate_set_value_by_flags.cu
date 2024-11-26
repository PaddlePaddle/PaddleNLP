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

#include "paddle/extension.h"

__global__ void speculate_set_value_by_flag_and_id(int64_t *pre_ids_all, 
                                         const int64_t *accept_tokens, 
                                         const int *accept_num, 
                                         const bool *stop_flags, 
                                         const int *seq_lens_encoder, 
                                         const int *seq_lens_decoder, 
                                         const int64_t *step_idx, 
                                         int bs,
                                         int length,
                                         int max_draft_tokens) {
    int tid = threadIdx.x;
    if (tid < bs && !stop_flags[tid]) {
        int64_t *pre_ids_all_now = pre_ids_all + tid * length;
        const int64_t *accept_tokens_now = accept_tokens + tid * max_draft_tokens;
        const int seq_len_dec = seq_lens_decoder[tid];
        const int seq_len_enc = seq_lens_encoder[tid];
        if (seq_len_dec == 0 && seq_len_enc == 0) return; // stoped
        if (step_idx[tid] >= 0) {
            for (int i = 0; i < accept_num[tid]; i++) {
                pre_ids_all_now[step_idx[tid] - i] = accept_tokens_now[accept_num[tid] - 1 - i];
            }
        }
    }
}

void SpeculateSetValueByFlagsAndIdx(const paddle::Tensor& pre_ids_all, 
                           const paddle::Tensor& accept_tokens,
                           const paddle::Tensor& accept_num,
                           const paddle::Tensor& stop_flags,
                           const paddle::Tensor& seq_lens_this_time,
                           const paddle::Tensor& seq_lens_encoder,
                           const paddle::Tensor& seq_lens_decoder,
                           const paddle::Tensor& step_idx ) {
    auto cu_stream = stop_flags.stream();
    std::vector<int64_t> pre_ids_all_shape = pre_ids_all.shape();
    
    int bs = seq_lens_this_time.shape()[0];
    int length = pre_ids_all_shape[1];
    int max_draft_tokens = accept_tokens.shape()[1];
    int block_size = (bs + 32 - 1) / 32 * 32;
    speculate_set_value_by_flag_and_id<<<1, block_size, 0, cu_stream>>>(const_cast<int64_t*>(pre_ids_all.data<int64_t>()), 
                                                              accept_tokens.data<int64_t>(), 
                                                              accept_num.data<int>(), 
                                                              stop_flags.data<bool>(), 
                                                              seq_lens_encoder.data<int>(),
                                                              seq_lens_decoder.data<int>(),
                                                              step_idx.data<int64_t>(), 
                                                              bs, 
                                                              length,
                                                              max_draft_tokens);
}

PD_BUILD_OP(speculate_set_value_by_flags_and_idx)
    .Inputs({"pre_ids_all", "accept_tokens", "accept_num", "stop_flags", "seq_lens_this_time", "seq_lens_encoder", "seq_lens_decoder", "step_idx"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateSetValueByFlagsAndIdx));