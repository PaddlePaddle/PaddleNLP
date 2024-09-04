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
