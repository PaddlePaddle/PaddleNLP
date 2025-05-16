// swiglu_probs_grad_op.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "paddle/extension.h"

#include <vector>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
using BFloat16 = __nv_bfloat16;
#else
struct BFloat16 {
    uint16_t x;
    
    __host__ __device__ BFloat16() : x(0) {}
    
    __host__ __device__ BFloat16(float val) {
        uint32_t* val_bits = reinterpret_cast<uint32_t*>(&val);
        x = static_cast<uint16_t>(*val_bits >> 16);
    }
    
    __host__ __device__ operator float() const {
        uint32_t bits = static_cast<uint32_t>(x) << 16;
        return *reinterpret_cast<float*>(&bits);
    }
};
#endif

__global__ void SwigluProbsGradKernel(
    const BFloat16* o1,               // [seq_len*topk, moe_intermediate_size*2]
    const BFloat16* do2_s,            // [seq_len*topk, moe_intermediate_size]
    const float* unzipped_probs,      // [seq_len*topk, 1]
    BFloat16* do1,                    // [seq_len*topk, moe_intermediate_size*2]
    float* probs_grad,                // [seq_len*topk, 1]
    BFloat16* o2_s,                   // [seq_len*topk, moe_intermediate_size]
    int seq_len_topk,                 // seq_len * topk
    int moe_intermediate_size         
) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const BFloat16* o1_row = o1 + row_idx * moe_intermediate_size * 2;
    const BFloat16* do2_s_row = do2_s + row_idx * moe_intermediate_size;
    BFloat16* do1_row = do1 + row_idx * moe_intermediate_size * 2;
    BFloat16* o2s_row = o2_s + row_idx * moe_intermediate_size;
    
    float prob = unzipped_probs[row_idx];
    
    __shared__ float sum_buffer[1024]; 
    
    float local_probs_grad = 0.0f;
    
    for (int i = tid; i < moe_intermediate_size; i += blockDim.x) {
        float lhs = static_cast<float>(o1_row[i]);
        float rhs = static_cast<float>(o1_row[i + moe_intermediate_size]);
        
        float sig = 1.0f / (1.0f + expf(-lhs));
        float tmp = sig * lhs;
        float o2_val = tmp * rhs; 
        
        float do2_s_val = static_cast<float>(do2_s_row[i]);
        float do2_val = do2_s_val * prob;
        
        float x0_grad = do2_val * rhs * sig * (1.0f + lhs - tmp);
        float x1_grad = do2_val * tmp;
        
        do1_row[i] = BFloat16(x0_grad);
        do1_row[i + moe_intermediate_size] = BFloat16(x1_grad);
        o2s_row[i] = BFloat16(o2_val * prob);
        
        local_probs_grad += do2_s_val * o2_val;
    }
    
    sum_buffer[tid] = local_probs_grad;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_buffer[tid] += sum_buffer[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        probs_grad[row_idx] = sum_buffer[0];
    }
}

std::vector<paddle::Tensor> SwigluProbsGradCUDABackward(
    const paddle::Tensor& o1,
    const paddle::Tensor& do2_s,
    const paddle::Tensor& unzipped_probs) {

    auto o1_dims = o1.dims();
    int seq_len_topk = o1_dims[0];
    int moe_intermediate_size_2 = o1_dims[1];
    int moe_intermediate_size = moe_intermediate_size_2 / 2;
    
    auto do1 = paddle::empty_like(o1);
    auto probs_grad = paddle::empty({seq_len_topk, 1}, paddle::DataType::FLOAT32, o1.place());
    auto o2_s = paddle::empty_like(do2_s);
    
    const BFloat16* o1_ptr = reinterpret_cast<const BFloat16*>(o1.data<phi::bfloat16>());
    const BFloat16* do2_s_ptr = reinterpret_cast<const BFloat16*>(do2_s.data<phi::bfloat16>());
    const float* unzipped_probs_ptr = unzipped_probs.data<float>();
    BFloat16* do1_ptr = reinterpret_cast<BFloat16*>(do1.data<phi::bfloat16>());
    float* probs_grad_ptr = probs_grad.data<float>();
    BFloat16* o2_s_ptr = reinterpret_cast<BFloat16*>(o2_s.data<phi::bfloat16>());
    
    int block_size = 256; 
    
    SwigluProbsGradKernel<<<seq_len_topk, block_size, 0, o1.stream()>>>(
        o1_ptr, do2_s_ptr, unzipped_probs_ptr, do1_ptr, probs_grad_ptr, o2_s_ptr,
        seq_len_topk, moe_intermediate_size);
        
    
    return {do1, probs_grad, o2_s};
}

PD_BUILD_OP(fused_swiglu_probs_bwd)
    .Inputs({"o1", "do2_s", "unzipped_probs"})
    .Outputs({"do1", "probs_grad", "o2_s"})
    .SetKernelFn(PD_KERNEL(SwigluProbsGradCUDABackward));
