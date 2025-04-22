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

template<typename T>
__global__ void SaveOutputDygraphKernel(
    T* out_pool,
    const T* cur_out,
    const int64_t* step_idx,
    const int* result_idx,
    const int bsz,
    const int result_num,
    const int max_dec_len) {
        for (int bid = blockIdx.x; bid < bsz; bid += gridDim.x) {
            auto src_addr = cur_out + bid;
            auto cur_step_idx = step_idx[bid] - 1;
            for (int i = threadIdx.x; i < result_num; i += blockDim.x) {
                auto tgt_id = result_idx[bid * result_num + i];
                if (tgt_id >= 0) {
                    auto tgt_addr = out_pool + tgt_id * max_dec_len + cur_step_idx;
                    *tgt_addr = *src_addr;
                }
            }
        }
}

void SaveOutputDygraph(
    const paddle::Tensor& all_token_ids,
    const paddle::Tensor& tokens,
    const paddle::Tensor& result_ids,
    const paddle::Tensor& step_idx
) {
    auto cu_stream = all_token_ids.stream();
    
    const int bsz = tokens.shape()[0];
    const int result_num = result_ids.shape()[1];
    const int max_dec_len = all_token_ids.shape()[1];

    SaveOutputDygraphKernel<<<bsz, 256, 0, cu_stream>>>(
        const_cast<int64_t*>(all_token_ids.data<int64_t>()),
        tokens.data<int64_t>(),
        step_idx.data<int64_t>(),
        result_ids.data<int>(),
        bsz,
        result_num,
        max_dec_len
    );

    // SaveOutputDygraphKernel<<<bsz, 256, 0, cu_stream>>>(
    //     const_cast<float*>(all_scores.data<float>()),
    //     scores.data<float>(),
    //     step_idx.data<int64_t>(),
    //     result_ids.data<int>(),
    //     bsz,
    //     result_num,
    //     max_dec_len
    // );
}

PD_BUILD_OP(save_output_dygraph)
    .Inputs({"all_token_ids", "tokens", "result_ids", "step_idx"})
    .Outputs({"all_token_ids_out"})
    .SetInplaceMap({{"all_token_ids", "all_token_ids_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutputDygraph));