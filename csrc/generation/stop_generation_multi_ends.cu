// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

__device__ bool is_in_end(const int64_t id, const int64_t *end_ids, int length) {
    bool flag = false;
    for (int i = 0; i < length; i++) {
        if (id == end_ids[i]) {
            return true;
        }
    }
    return flag;
}

__global__ void set_value_by_flags(
        bool *stop_flags, 
        const int64_t *end_ids, 
        const int64_t *topk_ids, 
        const int bs, 
        const int end_length) {
    int tid = threadIdx.x;
    if (tid < bs) {
        if (is_in_end(topk_ids[tid], end_ids, end_length)) {
            stop_flags[tid] = true;
        }
    }
}

void GetStopFlagsMulti(const paddle::Tensor& topk_ids, const paddle::Tensor& stop_flags, const paddle::Tensor& end_ids) {
    PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
    PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
    
    auto cu_stream = topk_ids.stream();
    std::vector<int64_t> shape = topk_ids.shape();
    int64_t bs_now = shape[0];
    int64_t end_length = end_ids.shape()[0];
    int block_size = (bs_now + 32 - 1) / 32 * 32;
    set_value_by_flags<<<1, block_size, 0, cu_stream>>>(
        const_cast<bool*>(stop_flags.data<bool>()), 
        end_ids.data<int64_t>(), 
        topk_ids.data<int64_t>(), 
        bs_now, end_length);
}

PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids", "stop_flags", "end_ids"})
    .Outputs({"topk_ids_out", "stop_flags_out"})
    .Attrs({"mode: int64_t"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti));
