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

void UpdateInputIdsCPU(const paddle::Tensor& input_ids_cpu,
        const std::vector<int64_t>& task_input_ids,
        const int bid,
        const int max_seq_len) {
    int64_t* input_ids_cpu_data = const_cast<int64_t*>(input_ids_cpu.data<int64_t>());
    // printf("Input len is %d\n", task_input_ids.size());

    for (int i = 0; i < task_input_ids.size(); i++) {
        // printf("%lld\n", task_input_ids[i]);
        input_ids_cpu_data[bid * max_seq_len + i] = task_input_ids[i];
    }
}

PD_BUILD_OP(speculate_update_input_ids_cpu)
        .Inputs({"input_ids_cpu"})
        .Outputs({"input_ids_cpu_out"})
        .Attrs({"task_input_ids: std::vector<int64_t>", "bid: int", "max_seq_len: int"})
        .SetInplaceMap({{"input_ids_cpu", "input_ids_cpu_out"}})
        .SetKernelFn(PD_KERNEL(UpdateInputIdsCPU));
