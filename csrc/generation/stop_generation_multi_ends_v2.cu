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
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>

__device__ bool is_in_end_v2(const int64_t id, const int64_t *end_ids, int length) {
    bool flag = false;
    for (int i = 0; i < length; i++) {
        if (id == end_ids[i]) {
            return true;
        }
    }
    return flag;
}

__global__ void set_value_by_flags_v2(
        bool *stop_flags,
        int64_t *topk_ids,
        int64_t *next_tokens,
        const int64_t *end_ids, 
        const int *seq_lens,
        const int bs, 
        const int end_length) {
    int tid = threadIdx.x;
    if (tid < bs) {
        if (stop_flags[tid]) {
            if (seq_lens[tid] == 0) {
                topk_ids[tid] = -1;
            } else {
                topk_ids[tid] = end_ids[0];
                next_tokens[tid] = end_ids[0];
            }
        } else {
            next_tokens[tid] = topk_ids[tid];
        }
        if (is_in_end_v2(topk_ids[tid], end_ids, end_length)) {
            stop_flags[tid] = true;
        }
    }
}

void GetStopFlagsMultiV2(const paddle::Tensor& topk_ids, 
                         const paddle::Tensor& stop_flags, 
                         const paddle::Tensor& seq_lens, 
                         const paddle::Tensor& end_ids,
                         const paddle::Tensor& next_tokens) {
    PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
    PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
    
    auto cu_stream = topk_ids.stream();
    std::vector<int64_t> shape = topk_ids.shape();
    int64_t bs_now = shape[0];
    int64_t end_length = end_ids.shape()[0];
    int block_size = (bs_now + 32 - 1) / 32 * 32;
    set_value_by_flags_v2<<<1, block_size, 0, cu_stream>>>(
        const_cast<bool*>(stop_flags.data<bool>()), 
        const_cast<int64_t*>(topk_ids.data<int64_t>()),
        const_cast<int64_t*>(next_tokens.data<int64_t>()),
        end_ids.data<int64_t>(),
        seq_lens.data<int>(),
        bs_now, end_length);
}

PD_BUILD_OP(set_stop_value_multi_ends_v2)
    .Inputs({"topk_ids", "stop_flags", "seq_lens", "end_ids", "next_tokens"})
    .Outputs({"topk_ids_out", "stop_flags_out", "next_tokens_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMultiV2));