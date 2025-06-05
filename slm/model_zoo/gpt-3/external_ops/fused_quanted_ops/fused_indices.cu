#include "quant_utils.h"

__global__ void GenerateExpertIndices(
    const int* __restrict__ prefix_sum,
    int* __restrict__ indices,
    const int num_experts,
    const int total_tokens
) {
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_tokens) return;
    
    int expert_idx = 0;
    while (expert_idx < num_experts && token_idx >= prefix_sum[expert_idx]) {
        expert_idx++;
    }
    
    indices[token_idx] = expert_idx;
}

std::vector<paddle::Tensor> generate_expert_indices_fast(
    const std::vector<int> &tokens_per_expert
) {
    const int num_experts = tokens_per_expert.size();
    
    std::vector<int> prefix_sum(num_experts);
    int total_tokens = 0;
    for (int i = 0; i < num_experts; ++i) {
        total_tokens += tokens_per_expert[i];
        prefix_sum[i] = total_tokens;
    }
    
    paddle::Tensor indices = paddle::empty({total_tokens}, paddle::DataType::INT32, paddle::GPUPlace());
    paddle::Tensor prefix_sum_gpu = paddle::empty({num_experts}, paddle::DataType::INT32, paddle::GPUPlace());
    
    cudaMemcpy(
        prefix_sum_gpu.data<int>(),
        prefix_sum.data(),
        num_experts * sizeof(int),
        cudaMemcpyHostToDevice
    );
    
    dim3 grid((total_tokens + 255) / 256);
    dim3 block(256);
    
    GenerateExpertIndices<<<grid, block>>>(
        prefix_sum_gpu.data<int>(),
        indices.data<int>(),
        num_experts,
        total_tokens
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return {indices};
}


PD_BUILD_OP(generate_expert_indices_fast)
    .Attrs({"tokens_per_expert : std::vector<int>"})
    .Outputs({"indices"})
    .SetKernelFn(PD_KERNEL(generate_expert_indices_fast));
