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
        const int64_t *end_ids, 
        const int *seq_lens,
        const int bs, 
        const int end_length) {
    int tid = threadIdx.x;
    if (tid < bs) {
        if (stop_flags[tid]) topk_ids[tid] = end_ids[0]; // 长度超限，当前位置置为2
        if (seq_lens[tid] == 0) topk_ids[tid] = -1; // 已终止，当前位置置为-1
        if (is_in_end_v2(topk_ids[tid], end_ids, end_length)) {
            stop_flags[tid] = true;
        }
    }
}

void GetStopFlagsMultiV2(const paddle::Tensor& topk_ids, const paddle::Tensor& stop_flags, const paddle::Tensor& seq_lens, const paddle::Tensor& end_ids) {
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
        end_ids.data<int64_t>(),
        seq_lens.data<int>(),
        bs_now, end_length);
}

PD_BUILD_OP(set_stop_value_multi_ends_v2)
    .Inputs({"topk_ids", "stop_flags", "seq_lens", "end_ids"})
    .Outputs({"topk_ids_out", "stop_flags_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMultiV2));