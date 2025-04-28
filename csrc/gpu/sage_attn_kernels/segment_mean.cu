#include "sageattn_utils.cuh"
#include "sageattn_fused_varlen.cuh"

#include <cuda_fp16.h>   // for __half and __half2 (float16) intrinsics
#include <cuda_bf16.h>   // for __nv_bfloat16 and __nv_bfloat162 (bfloat16)
#include <cub/cub.cuh>   // for CUB utilities (optional, e.g., for reductions)

#define WARP_SIZE 32

#define BLOCK_SIZE_DIM 64

// grid: (num_head, batch_size)
// block: (head_dim)
template<typename T, uint32_t NUM_HTREADS, uint32_t CHUNK_SIZE, uint32_t ITEMS_PER_THREAD, uint32_t HEAD_DIM>
__global__ void NaiveSegmentMeanKernel(T* __restrict__ input,
                                        T* __restrict__ output,
                                        uint32_t* __restrict__ cu_seqlens,
                                        int stride_i_seqlen,      // num_head x head_dim
                                        int stride_i_h,           //            head_dim
                                        int stride_o_seqlen,      // num_head x head_dim
                                        int stride_o_h,           //            head_dim
                                        int batch_size) 
{
    const int head_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int dim_id = threadIdx.x;

    const int seqlen_this_time = cu_seqlens[batch_id + 1] - cu_seqlens[batch_id];

    T sum = T(0);
    for (int i = 0; i < seqlen_this_time; i ++) {
        T* input_idx = input + (cu_seqlens[batch_id] + i) * stride_i_seqlen + head_id * stride_i_h + dim_id;
            
        sum += *input_idx;
    }
    T mean = sum / T(seqlen_this_time);

    // write back
    T* output_idx = output + batch_id * stride_o_seqlen + head_id * stride_o_h + dim_id;
    *output_idx = mean;

}

template <typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y, int CHUNK_SIZE, int head_dim>
__global__ void SegmentMeanKernel(
    T* __restrict__ input,
    T* __restrict__ output,
    const uint32_t* __restrict__ cu_seqlens,
    int stride_i_seqlen,
    int stride_i_h,
    int stride_o_seqlen,
    int stride_o_h,
    int num_head, 
    int batch_size
) {
    // 一个block 单次计算 chunk_size * head_dim个数据
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared_data = reinterpret_cast<T*>(shared_mem);

    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.x;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;

    const int tid = tid_y * BLOCK_DIM_X + tid_x;

    const int seq_start = cu_seqlens[batch_id];
    const int seq_end = cu_seqlens[batch_id + 1];
    const int seq_len = seq_end - seq_start;

    const int dim_start = tid_x * (head_dim / BLOCK_DIM_X);     // BLOCK_DIM_X = 32, BLOCK_DIM_Y = 4, head_dim / BLOCK_DIM_X = 4
    const int dim_end = (tid_x + 1) * (head_dim / BLOCK_DIM_X);

    T sum[head_dim / BLOCK_DIM_X] = {0};    // 4, 一个thread计算 4 个 head_dim的sum

    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        const int chunk_end = min(chunk_start + CHUNK_SIZE, seq_len);

        for (int s = chunk_start; s < chunk_end; s += BLOCK_DIM_Y) { // loop 32次
            const int seq_idx = s + tid_y;      // 0, 1, 2, 3
            if (seq_idx < chunk_end) {
                const T* src = input + (seq_start + seq_idx) * stride_i_seqlen + head_id * stride_i_h;
                #pragma unroll
                for (int d = dim_start; d < dim_end; ++d) { // loop 4 次
                    shared_data[(seq_idx - chunk_start) * head_dim + d] = src[d];
                }
            }
        }
        __syncthreads();

        // 局部归约


        #pragma unroll
        for (int s = 0; s < chunk_end - chunk_start; ++s) {
            #pragma unroll
            for (int d = dim_start; d < dim_end; ++d) {
                sum[d - dim_start] += shared_data[s * head_dim + d];
            }
        }
        __syncthreads();
    }

    // 写回结果
    if (tid_y == 0) {
        T* dst = output + batch_id * stride_o_seqlen + head_id * stride_o_h;
        #pragma unroll
        for (int d = dim_start; d < dim_end; ++d) {
            dst[d] = sum[d - dim_start] / static_cast<T>(seq_len);
        }
    }
}

std::vector<paddle::Tensor> chunked_segment_mean_fwd(paddle::Tensor& input,         // [total_seqlen, num_head, head_dim]
                                                     paddle::Tensor& cu_seqlens,    // [batch_size + 1], prefix-sum array of sequence lengths
                                                     const int max_seqlen) 
{
    CHECK_CONTIGUOUS(input);

    CHECK_DIMS(input, 3);

    const int batch_size = cu_seqlens.shape()[0] - 1;
    const int num_head = input.shape()[1];
    const int head_dim = input.shape()[2];

    // transpose operator invoke
    paddle::Tensor output = paddle::zeros({batch_size, input.shape()[1], input.shape()[2]}, input.dtype(), paddle::GPUPlace());

    PD_CHECK(input.dtype() == paddle::DataType::FLOAT16 || input.dtype() == paddle::DataType::BFLOAT16, "Only float16 and bfloat16 are supported");

    DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, {
        DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM, {
            constexpr int NUM_THREADS = 256;

            constexpr int BLOCK_SIZE_SEQ = 256;
            constexpr int BLOCK_SIZE_HEAD = 4;

            // for [131, 24, 128], the grid is: (1, 6, 2)
            
            constexpr int BLOCK_DIM_X = 32;
            constexpr int BLOCK_DIM_Y = 4;
            constexpr int CHUNK_SIZE = 128;
            size_t sMemSize = CHUNK_SIZE * HEAD_DIM * 2; // 16 bits - 2 byte

            dim3 grid(num_head, batch_size);
            dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);   // 32 * 4

            SegmentMeanKernel<c_type, BLOCK_DIM_X, BLOCK_DIM_Y, CHUNK_SIZE, HEAD_DIM><<<grid, block, sMemSize>>>(
                reinterpret_cast<c_type*>(input.data()),                // [total_seqlen, num_head, head_dim]
                reinterpret_cast<c_type*>(output.data()),               // [batch_size, num_head, head_dim]
                reinterpret_cast<uint32_t*>(cu_seqlens.data()),
                input.strides()[0], input.strides()[1], 
                output.strides()[0], output.strides()[1],
                num_head,
                batch_size);
        });
    });

    return {output};
}