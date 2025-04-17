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

#include "helper.h"
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>

void set_flags_multi_ends(char *str_flags, bool *res, int length) {
    for (int i = 0; i < length; i++) {
        if (str_flags[i] == '0') {
            res[i] = false;
        } else {
            res[i] = true;
        }
    }
}

__global__ void set_value_by_flags(const bool *stop_flags, const int64_t *end_ids, int64_t *topk_ids, bool *stop_flags_out, const int bs, int end_length) {
    int tid = threadIdx.x;
    if (tid < bs) {
        topk_ids[tid] = stop_flags[tid] ? end_ids[0] : topk_ids[tid];
        __syncthreads();
        if (is_in_end(topk_ids[tid], end_ids, end_length)) {
            stop_flags_out[tid] = true;
        }
    }
}

__global__ void set_value_by_flags_both(const bool *flags, const bool *stop_flags, const int64_t *end_ids, int64_t *topk_ids, bool *stop_flags_out, const int bs, int end_length) {
    int tid = threadIdx.x;
    if (tid < bs) {
        topk_ids[tid] = flags[tid] || stop_flags[tid] ? end_ids[0] : topk_ids[tid];
        __syncthreads();
        if (is_in_end(topk_ids[tid], end_ids, end_length)) {
            stop_flags_out[tid] = true;
        }
    }
}

std::vector<paddle::Tensor> GetStopFlagsMulti(const paddle::Tensor& topk_ids, const paddle::Tensor& stop_flags, const paddle::Tensor& end_ids, int64_t mode) {
    // mode = 0, stop_generation and stop_flags
    // mode = 1, just stop_generation
    // mode = 2, just stop_flags
    PD_CHECK(mode <= 2);
    PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
    PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);

    auto cu_stream = topk_ids.stream();
    std::vector<int64_t> shape = topk_ids.shape();
    int64_t bs_now = shape[0];
    int64_t end_length = end_ids.shape()[0];
    auto topk_ids_out = topk_ids.copy_to(topk_ids.place(), false); // gpu -> gpu
    auto stop_flags_out = stop_flags.copy_to(stop_flags.place(), false); // gpu -> gpu
    if (mode == 0 || mode == 1) {
        constexpr char *path = "/root/paddlejob/workspace/env_run/lzy/ERNIE_ALL/early_stop/ERNIE3.0-fused-fp16/ops/test";
        auto flags = paddle::full({bs_now, 1}, 1, paddle::DataType::BOOL, paddle::CPUPlace());
        int fd = -1;
        int ret = -1;
        void *addr = nullptr;
        fd = open(path, O_RDWR);
        if(-1 == fd){
            perror("open error");
        }
        addr = mmap(NULL, bs_now, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if(addr == MAP_FAILED){
            perror("mmap error");
        }
        close(fd);
        set_flags_multi_ends((char *)addr, flags.data<bool>(), bs_now);
        munmap(addr, bs_now);
        auto flags_gpu = flags.copy_to(topk_ids.place(), false); // cpu -> gpu
        int block_size = (bs_now + 32 - 1) / 32 * 32;
        if (mode == 0) {
            set_value_by_flags_both<<<1, block_size, 0, cu_stream>>>(flags_gpu.data<bool>(), stop_flags.data<bool>(), end_ids.data<int64_t>(), topk_ids_out.data<int64_t>(), stop_flags_out.data<bool>(), bs_now, end_length);
        } else {
            set_value_by_flags<<<1, block_size, 0, cu_stream>>>(flags_gpu.data<bool>(), end_ids.data<int64_t>(), topk_ids_out.data<int64_t>(), stop_flags_out.data<bool>(), bs_now, end_length);
        }
    } else if (mode == 2) {
        int block_size = (bs_now + 32 - 1) / 32 * 32;
        set_value_by_flags<<<1, block_size, 0, cu_stream>>>(stop_flags.data<bool>(), end_ids.data<int64_t>(), topk_ids_out.data<int64_t>(), stop_flags_out.data<bool>(), bs_now, end_length);
    }
    return {topk_ids_out, stop_flags_out};
}

std::vector<std::vector<int64_t>> GetStopFlagsMultiInferShape(const std::vector<int64_t>& topk_ids_shape, const std::vector<int64_t>& stop_flags_shape, const std::vector<int64_t>& end_ids_shape) {
    return {topk_ids_shape, stop_flags_shape};
}

std::vector<paddle::DataType> GetStopFlagsMultiInferDtype(const paddle::DataType& topk_ids_dtype, const paddle::DataType& stop_flags_dtype, const paddle::DataType& end_ids_dtype) {
    return {topk_ids_dtype, stop_flags_dtype};
}

PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids", "stop_flags", "end_ids"})
    .Outputs({"topk_ids_out", "stop_flags_out"})
    .Attrs({"mode: int64_t"})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti))
    .SetInferShapeFn(PD_INFER_SHAPE(GetStopFlagsMultiInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetStopFlagsMultiInferDtype));