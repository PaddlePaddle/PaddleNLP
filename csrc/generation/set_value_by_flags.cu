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

__global__ void set_value_by_flag_and_id(const bool *stop_flags, int64_t *pre_ids_all, const int64_t *pre_ids, const int64_t *step_idx, int bs, int length) {
    int tid = threadIdx.x;
    if (tid < bs && !stop_flags[tid]) {
        int64_t *pre_ids_all_now = pre_ids_all + tid * length;
        if (step_idx[tid] >= 0) {
            pre_ids_all_now[step_idx[tid]] = pre_ids[tid];
        }
    }
}

std::vector<paddle::Tensor> SetValueByFlagsAndIdx(const paddle::Tensor& pre_ids_all, const paddle::Tensor& pre_ids_now, const paddle::Tensor& step_idx, const paddle::Tensor& stop_flags) {
    auto cu_stream = stop_flags.stream();
    std::vector<int64_t> pre_ids_all_shape = pre_ids_all.shape();
    auto stop_flags_out = stop_flags.copy_to(stop_flags.place(), false); // gpu -> gpu

    int bs = stop_flags.shape()[0];
    int length = pre_ids_all_shape[1];
    int block_size = (bs + 32 - 1) / 32 * 32;
    set_value_by_flag_and_id<<<1, block_size, 0, cu_stream>>>(stop_flags.data<bool>(), const_cast<int64_t*>(pre_ids_all.data<int64_t>()), pre_ids_now.data<int64_t>(), step_idx.data<int64_t>(), bs, length);
    return {stop_flags_out};
}

std::vector<std::vector<int64_t>> SetValueByFlagsAndIdxInferShape(const std::vector<int64_t>& pre_ids_all_shape, const std::vector<int64_t>& pre_ids_now_shape,
                                                                  const std::vector<int64_t>& step_idx_shape, const std::vector<int64_t>& stop_flags_shape) {
    return {stop_flags_shape};
}

std::vector<paddle::DataType> SetValueByFlagsAndIdxInferDtype(const paddle::DataType& pre_ids_all_dtype,
                                                              const paddle::DataType& pre_ids_now_dtype,
                                                              const paddle::DataType& step_idx_dtype,
                                                              const paddle::DataType& stop_flags_dtype) {
    return {stop_flags_dtype};
}

PD_BUILD_OP(set_value_by_flags_and_idx)
    .Inputs({"pre_ids_all", "pre_ids_now", "step_idx", "stop_flags"})
    .Outputs({"stop_flags_out"})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdx))
    .SetInferShapeFn(PD_INFER_SHAPE(SetValueByFlagsAndIdxInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SetValueByFlagsAndIdxInferDtype));
