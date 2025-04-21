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

#include "paddle/extension.h"


#define CEILDIV(a,b) (((a+b-1)/b))

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(const scalar_t* __restrict__ topk_ids,
                                                    int32_t* __restrict__ sorted_token_ids,
                                                    int32_t* __restrict__ cumsum_buffer, 
                                                    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t, int num_experts>
__global__ void moe_align_block_size_kernel(const scalar_t* __restrict__ topk_ids,
                                            int32_t* __restrict__ expert_ids,
                                            int32_t* __restrict__ total_tokens_post_pad, 
                                            int32_t block_size, 
                                            size_t numel, 
                                            int32_t* __restrict__ cumsum_buffer) {
  __shared__ int32_t tokens_per_ep[num_experts];

  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
      tokens_per_ep[i] = 0;
  }

  __syncthreads();

  for (int i = threadIdx.x; i < numel; i += blockDim.x) {
    int expert_id = topk_ids[i];
    atomicAdd(&tokens_per_ep[expert_id], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum_buffer[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = tokens_per_ep[i-1];
      cumsum_buffer[i] = cumsum_buffer[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum_buffer[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum_buffer[threadIdx.x]; i < cumsum_buffer[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}


std::vector<std::vector<int64_t>> preprocess_for_moeInferShape(const std::vector<int64_t>& topk_ids, int64_t num_experts, int64_t block_size) {


    int topk_ids_numel = topk_ids[0] * topk_ids[1];
    int max_num_tokens_padded = topk_ids_numel + num_experts * (block_size - 1);

    std::vector<int64_t> sorted_ids = {max_num_tokens_padded};

    int max_num_m_blocks = max_num_tokens_padded / block_size;
    std::vector<int64_t> expert_ids = {max_num_m_blocks};
    std::vector<int64_t> num_tokens_post_pad = {1};

    return {sorted_ids, expert_ids, num_tokens_post_pad};
}

std::vector<paddle::DataType> preprocess_for_moeIferDtype(const paddle::DataType& topk_ids, int64_t num_experts, int64_t block_size) {
    return {paddle::DataType::INT32, paddle::DataType::INT32, paddle::DataType::INT32};
}



std::vector<paddle::Tensor> preprocess_for_moe_kernel(const paddle::Tensor& topk_ids, int64_t num_experts, int64_t block_size) {
    
    int topk_ids_numel = topk_ids.shape()[0] * topk_ids.shape()[1];
    int max_num_tokens_padded = topk_ids_numel + num_experts * (block_size - 1);

    auto sorted_ids = paddle::full(
        {max_num_tokens_padded}, 
        topk_ids_numel,
        paddle::DataType::INT32, 
        topk_ids.place()
    );

    int max_num_m_blocks = max_num_tokens_padded / block_size;
    
    auto expert_ids = paddle::empty(
        {max_num_m_blocks}, paddle::DataType::INT32, 
        topk_ids.place()
    );

    auto num_tokens_post_pad = paddle::empty(
        {1}, 
        paddle::DataType::INT32, 
         topk_ids.place()
    );

    auto cumsum_buffer = paddle::empty(
        {num_experts + 1}, 
        paddle::DataType::INT32, 
        topk_ids.place()
    );

    auto stream = topk_ids.stream();
    using scalar_t = int64_t;

    # define run_align_kernel(num_experts) \
    auto align_kernel = moe_align_block_size_kernel<scalar_t, num_experts>; \
    align_kernel<<<1, 1024, 0, stream>>>( \
    topk_ids.data<scalar_t>(),  \
    expert_ids.data<int32_t>(), \
    num_tokens_post_pad.data<int32_t>(), \
    block_size,  \
    topk_ids_numel, \
    cumsum_buffer.data<int32_t>());
    
    if (num_experts == 8) {
      run_align_kernel(8);
    } else if (num_experts == 256) {
      run_align_kernel(256);
    } else if (num_experts == 2) {
      run_align_kernel(2);
    } else {
      printf("errors");
    }

    const int block_threads = 256;
    const int num_blocks = CEILDIV(topk_ids_numel, block_threads);
    const int max_blocks = 65535;
    const int actual_blocks = std::min(num_blocks, max_blocks);

    auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;

    sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(topk_ids.data<scalar_t>(),
                                                              sorted_ids.data<int32_t>(),
                                                              cumsum_buffer.data<int32_t>(), 
                                                              topk_ids_numel);



    return {sorted_ids, expert_ids, num_tokens_post_pad};
}

PD_BUILD_OP(preprocess_for_moe)
    .Inputs({"topk_ids"})
    .Attrs({"num_experts: int64_t", "block_size: int64_t"})
    .Outputs({"sorted_ids", "expert_ids", "num_tokens_post_pad"})
    .SetKernelFn(PD_KERNEL(preprocess_for_moe_kernel))
    .SetInferShapeFn(PD_INFER_SHAPE(preprocess_for_moeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(preprocess_for_moeIferDtype));






