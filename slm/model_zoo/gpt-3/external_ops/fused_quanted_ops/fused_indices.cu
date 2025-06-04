#include "quant_utils.h"

__global__ void GenerateExpertIndices(
    const int* __restrict__ tokens_per_expert,
    int* __restrict__ indices,
    const int* __restrict__ expert_offsets,
    const int num_experts
) {
    const int expert_idx = blockIdx.x;
    if (expert_idx >= num_experts) return;
    
    const int start_pos = expert_offsets[expert_idx];
    const int count = tokens_per_expert[expert_idx];
    const int tid = threadIdx.x;
    
    for (int i = tid; i < count; i += blockDim.x) {
        indices[start_pos + i] = expert_idx;
    }
}

void cpu_compute_offsets(const int* tokens_per_expert, int* expert_offsets, int num_experts) {
    expert_offsets[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
        expert_offsets[i] = expert_offsets[i - 1] + tokens_per_expert[i - 1];
    }
}

void dispatch_generate_expert_indices(
    const paddle::Tensor &tokens_per_expert,
    paddle::Tensor &indices,
    const std::vector<int>& expert_offsets_host,
    const int num_experts
) {
    int* expert_offsets_gpu;
    cudaMalloc(&expert_offsets_gpu, (num_experts + 1) * sizeof(int));
    cudaMemcpy(expert_offsets_gpu, expert_offsets_host.data(), 
               (num_experts + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 indices_grid(num_experts);
    dim3 indices_block(256);
    
    GenerateExpertIndices<<<indices_grid, indices_block, 0, tokens_per_expert.stream()>>>(
        tokens_per_expert.data<int>(),
        indices.data<int>(),
        expert_offsets_gpu,
        num_experts
    );
    
    cudaFree(expert_offsets_gpu);
}

std::vector<paddle::Tensor> generate_expert_indices(
    const paddle::Tensor &tokens_per_expert
) {
    PD_CHECK(tokens_per_expert.dtype() == paddle::DataType::INT32);
    PD_CHECK(tokens_per_expert.dims().size() == 1);
    
    const int num_experts = tokens_per_expert.shape()[0];
    
    // Get tokens_per_expert data (handle both CPU and GPU cases)
    std::vector<int> tokens_host(num_experts);
    if (tokens_per_expert.place().GetType() == phi::AllocationType::GPU) {
        cudaMemcpy(tokens_host.data(), tokens_per_expert.data<int>(), 
                   num_experts * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        memcpy(tokens_host.data(), tokens_per_expert.data<int>(), 
               num_experts * sizeof(int));
    }
    
    std::vector<int> expert_offsets_host(num_experts + 1);
    cpu_compute_offsets(tokens_host.data(), expert_offsets_host.data(), num_experts);
    
    const int total_tokens = expert_offsets_host[num_experts];
    
    paddle::Tensor indices = paddle::empty(
        {total_tokens}, 
        paddle::DataType::INT32, 
        tokens_per_expert.place()
    );
    
    dispatch_generate_expert_indices(
        tokens_per_expert,
        indices,
        expert_offsets_host,
        num_experts
    );
    
    return {indices};
}

PD_BUILD_OP(generate_expert_indices)
    .Inputs({"tokens_per_expert"})
    .Outputs({"indices"})
    .SetKernelFn(PD_KERNEL(generate_expert_indices));