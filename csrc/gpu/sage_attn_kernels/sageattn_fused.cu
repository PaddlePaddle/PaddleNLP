#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "sageattn_utils.cuh"
#include "paddle/extension.h"

enum class QuantType
{
  kInt8,
  kInt4,
};

template <typename T>
__device__ __forceinline__ float convert_to_float(T val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __half2float(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __bfloat162float(val);
  }
}

template <typename T>
__device__ __forceinline__ T convert_from_float(float val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __float2half_rn(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __float2bfloat16_rn(val);
  }
}


// 
// =========== kernel impl zone ===========
//
// Notice: the address in paddle is aligned in 4 bytes, not 16 bytes
// so theoretically the `float4(xx) = xx` way cannot load data from global memory
// peacefully, instead it will trigger CUDA 719: address misaligned fault.
// Thus, we will use pragma unroll macro and load data in a for loop, which introduces
// extra latency, but works in paddle.
template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, bool has_sm_scale = false, bool sub_mean = false, typename T>
__global__ void QuantInt8Kernel(T *__restrict__ input, T *__restrict__ mean, int8_t *__restrict__ output, float *__restrict__ scale, float sm_scale, const uint32_t num_tokens, 
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T x_val[num_pack_per_thread][8];
  T mean_val[8];
  float x_val_float[num_pack_per_thread][8];
  float mean_val_float[8];

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;
  float *scale_ptr_base = scale + batch_id * stride_bz_scale + head_id * stride_h_scale + bx;

  if constexpr (sub_mean)
  {
    *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base); // 8 elements
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      mean_val_float[j] = convert_to_float(mean_val[j]);
    }
  }

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  // load the data
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = convert_to_float(x_val[i][j]);
      }

      if constexpr (sub_mean)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] -= mean_val_float[j];
        }
      }

      if constexpr (has_sm_scale)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] *= sm_scale;
        }
      }
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = 0.0f;
      }
    }
  }

  float amax_val = 0.0000001f; // prevent from dividing by zero

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
    }
  }

  __shared__ float s_amax;
  const float block_amax_val = sageattn::blockReduceMax(amax_val);
  if (thread_id == 0)
  {
    s_amax = block_amax_val;
    scale_ptr_base[0] = s_amax / 127.0f;
  }

  __syncthreads();

  float tmp_scale = 127.0f / s_amax;

  char4 o_val[num_pack_per_thread][2];

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 2; j += 1)
    {
      o_val[i][j] = make_char4(
        float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale)
      );
    }
  }

  // int8 result
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float2*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float2*>(&o_val[i][0]);
    }
  }
}

template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero=false, typename T>
__global__ void TransposePadPermuteKernel(T *__restrict__ input, T *__restrict__ output, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output)
{

  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  uint32_t num_threads_per_token = head_dim / pack_size;
  uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size + thread_id / num_threads_per_cta * stride_d_output;

  __shared__ T shared_load[CTA_SIZE][head_dim];
  __shared__ T shared_store[head_dim][CTA_SIZE];

  // 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
  // permute on the seq dimension for fp8 mma
  uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
  uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
  uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod  / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);

  constexpr cp_async::SharedMemFillMode fill_mode = pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
  cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size, input_ptr_base, thread_base_token < num_tokens);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  uint32_t smem_row_base = thread_id % CTA_SIZE;
  uint32_t smem_col_base = thread_id / CTA_SIZE;
  uint32_t smem_col_stride = head_dim / 8;

  // TODO: use ldmatrix to do permutation
#pragma unroll
  for (uint32_t i = 0; i < 8; i++)
  {
    shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
  }

  __syncthreads();

  *(float4*)(output_ptr_base) = *(float4*)(&shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);
  // for unable-align reasons, we unroll it manually.
// #pragma unroll
//   for (int i = 0; i < 8; i++) {
//     *(output_ptr_base + i) = shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size + i];  // TODO: not debugged, maybe some problem
//   }
}

template<uint32_t pad_size, bool sub_mean = false, typename T>
__global__ void MeanScaleKernel(T *__restrict__ input, int8_t *__restrict__ output, float *__restrict__ mean, float *__restrict__ scale, const float scale_max, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_d_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16

  uint32_t head_id = blockIdx.x;
  uint32_t batch_id = blockIdx.y;
  uint32_t d_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t num_threads = blockDim.x;
  uint32_t gmem_stride = num_threads * pack_size;
  // pad the number of tokens to 16 to deal with fp8 permute in previous kernel
  uint32_t fp8_padded_num_tokens = (num_tokens + 15) / 16 * 16;
  uint32_t num_iters = fp8_padded_num_tokens / gmem_stride + ((fp8_padded_num_tokens % gmem_stride) > thread_id * pack_size);

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + d_id * stride_d_input + thread_id * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + d_id * stride_d_output + thread_id * pack_size;

  T x_val[8];
  float x_val_float[8];
  uint32_t x_val_fp8[2];

  float max_val = - 1000000.0f;
  float min_val = 1000000.0f;
  float sum_val = 0.0f;

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
// #pragma unroll
//     for (int ii = 0; ii < 8; ii++) {
//       x_val[ii] = *(input_ptr_base + i * gmem_stride + ii); // TODO: not debugged
//     }
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      float x_temp = convert_to_float(x_val[j]);
      max_val = fmaxf(max_val, x_temp);
      min_val = fminf(min_val, x_temp);

      if constexpr (sub_mean)
      {
        sum_val += x_temp;
      }
    }
  }

  // reduce
  __shared__ float s_amax_val;
  __shared__ float s_mean_val;

  float block_max_val = sageattn::blockReduceMax(max_val);
  float block_min_val = sageattn::blockReduceMin(min_val);
  float block_sum_val;

  if constexpr (sub_mean)
  {
    block_sum_val = sageattn::blockReduceSum(sum_val);
  }

  if (thread_id == 0)
  {
    s_mean_val = block_sum_val / fp8_padded_num_tokens;

    if constexpr (sub_mean)
    {
      s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val));
      mean[batch_id * stride_bz_mean + head_id * stride_h_mean + d_id] = s_mean_val;
    }
    else
    {
      s_amax_val = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
    }

    scale[batch_id * stride_bz_scale + head_id * stride_h_scale + d_id] = s_amax_val / scale_max;
  }

  __syncthreads();

  float mean_val = s_mean_val;
  float recp_scale = scale_max / s_amax_val;

  // recalculate num_iters to cover all fp8 output tokens to prevent nan in random initialization
  uint32_t padded_num_tokens = (num_tokens + pad_size - 1) / pad_size * pad_size;
  num_iters = padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
// #pragma unroll
//     for (int ii = 0; ii < 8; ii++) {
//       x_val[ii] = *(input_ptr_base + i * gmem_stride + ii); // TODO: not debugged
//     }
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      x_val_float[j] = convert_to_float(x_val[j]);
      if constexpr (sub_mean)
      {
        x_val_float[j] = (x_val_float[j] - mean_val) * recp_scale;
      }
      else
      {
        x_val_float[j] *= recp_scale;
      }
    }

    floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
    floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);

    *(uint2*)(output_ptr_base + i * gmem_stride) = *(uint2*)(&x_val_fp8[0]);
  }
}


// 
// =========== kernel API zone ===========
// 
void quant_per_block_int8_fuse_sub_mean_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& mean,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(mean);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, paddle::DataType::INT8);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.shape()[0];
  const int head_dim = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.shape()[1];
    num_heads = input.shape()[2];
    stride_seq_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_seq_output = output.strides()[1];
    stride_h_output = output.strides()[2];
  }
  else
  {
    num_tokens = input.shape()[2];
    num_heads = input.shape()[1];
    stride_seq_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_seq_output = output.strides()[2];
    stride_h_output = output.strides()[1];
  }

  auto input_dtype = input.dtype();
  auto mean_dtype = mean.dtype();

  PD_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");
  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM, {
        CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
        CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);
        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, true, c_type><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data()),
          reinterpret_cast<c_type*>(mean.data()),
          output.data<int8_t>(),
          reinterpret_cast<float*>(scale.data()),
          0.0f,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          mean.strides()[0], mean.strides()[1],
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.strides()[0], scale.strides()[1]
        );
      });
    });
  });
}

PD_BUILD_OP(quant_per_block_int8_fuse_sub_mean_cuda)
    .Inputs({"input", "mean", "output", "scale"})
    .Outputs({"out1", "out2", "out3", "out4"})
    .SetInplaceMap({{"input", "out1"}, {"mean", "out2"}, {"output", "out3"}, {"scale", "out4"}}) // Inplace
    .Attrs({"block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_block_int8_fuse_sub_mean_cuda_fwd));

void quant_per_warp_int8_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int warp_block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, paddle::DataType::INT8);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.shape()[0];
  const int head_dim = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.shape()[1];
    num_heads = input.shape()[2];
    stride_seq_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_seq_output = output.strides()[1];
    stride_h_output = output.strides()[2];
  }
  else
  {
    num_tokens = input.shape()[2];
    num_heads = input.shape()[1];
    stride_seq_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_seq_output = output.strides()[2];
    stride_h_output = output.strides()[1];
  }

  auto input_dtype = input.dtype();

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
      DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
        DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM, {
          CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
          CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE));
          dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);
          constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;
          dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

          QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<grid, block>>>(
            reinterpret_cast<c_type*>(input.data()),
            nullptr,
            output.data<int8_t>(),
            reinterpret_cast<float*>(scale.data()),
            0.0,
            num_tokens,
            stride_bz_input, stride_seq_input, stride_h_input,
            0, 0,
            stride_bz_output, stride_seq_output, stride_h_output,
            scale.strides()[0], scale.strides()[1]
          );
        });
      });
    });
  });
}

PD_BUILD_OP(quant_per_warp_int8_cuda)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"block_size: int", "warp_block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_warp_int8_cuda_fwd));

void quant_per_block_int8_cuda_scale_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                float sm_scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, paddle::DataType::INT8);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.shape()[0];
  const int head_dim = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.shape()[1];
    num_heads = input.shape()[2];
    stride_seq_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_seq_output = output.strides()[1];
    stride_h_output = output.strides()[2];
  }
  else
  {
    num_tokens = input.shape()[2];
    num_heads = input.shape()[1];
    stride_seq_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_seq_output = output.strides()[2];
    stride_h_output = output.strides()[1];
  }

  auto input_dtype = input.dtype();

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, true, false, c_type><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data()),
          nullptr,
          output.data<int8_t>(),
          reinterpret_cast<float*>(scale.data()),
          sm_scale,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          0, 0,
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.strides()[0], scale.strides()[1]
        );
      });
    });
  });
}

PD_BUILD_OP(quant_per_block_int8_cuda_scale)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"sm_scale: float", "block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_block_int8_cuda_scale_fwd));

void quant_per_block_int8_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, paddle::DataType::INT8);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.shape()[0];
  const int head_dim = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.shape()[1];
    num_heads = input.shape()[2];
    stride_seq_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_seq_output = output.strides()[1];
    stride_h_output = output.strides()[2];
  }
  else
  {
    num_tokens = input.shape()[2];
    num_heads = input.shape()[1];
    stride_seq_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_seq_output = output.strides()[2];
    stride_h_output = output.strides()[1];
  }

  auto input_dtype = input.dtype();

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data()),
          nullptr,
          output.data<int8_t>(),
          reinterpret_cast<float*>(scale.data()),
          0.0f,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          0, 0,
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.strides()[0], scale.strides()[1]
        );
      });
    });
  });
}

PD_BUILD_OP(quant_per_block_int8_cuda)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"sm_scale: float", "block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_block_int8_cuda_fwd));

// quant v用，但是v不是192，所以可以沿用原来的DISPATCH_HEAD_DIM
void transpose_pad_permute_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);

  constexpr int CTA_SIZE = 64;

  const int batch_size = input.shape()[0];
  const int head_dim = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_tokens, padded_num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.shape()[1];
    num_heads = input.shape()[2];
    stride_seq_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_d_output = output.strides()[1];
    stride_h_output = output.strides()[2];

    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;

    CHECK_SHAPE(output, batch_size, head_dim, num_heads, padded_num_tokens);
  }
  else
  {
    num_tokens = input.shape()[2];
    num_heads = input.shape()[1];
    stride_seq_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_d_output = output.strides()[2];
    stride_h_output = output.strides()[1];

    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;
    CHECK_SHAPE(output, batch_size, num_heads, head_dim, padded_num_tokens);
  }

  auto input_dtype = input.dtype();
  auto output_dtype = output.dtype();

  PD_CHECK(input_dtype == output_dtype, "Input and output must have the same data type");

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 grid(padded_num_tokens / CTA_SIZE, num_heads, batch_size);

      static_assert(CTA_SIZE * HEAD_DIM <= 8192);

      dim3 block(CTA_SIZE * (HEAD_DIM / 8));

      TransposePadPermuteKernel<HEAD_DIM, CTA_SIZE, true, c_type><<<grid, block>>>(
        reinterpret_cast<c_type*>(input.data()),
        reinterpret_cast<c_type*>(output.data()),
        num_tokens,
        stride_bz_input, stride_seq_input, stride_h_input,
        stride_bz_output, stride_d_output, stride_h_output
      );
    });
  });
}

PD_BUILD_OP(transpose_pad_permute_cuda)
    .Inputs({"input", "output"})
    .Outputs({"out1", "out2"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}}) // Inplace
    .Attrs({"tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(transpose_pad_permute_cuda_fwd));

void scale_fuse_quant_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int num_tokens,
                float scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  // CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.shape()[0];
  const int num_tokens_padded = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_heads, head_dim;
  int stride_d_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_heads = input.shape()[2];
    head_dim = input.shape()[1];
    stride_d_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_d_output = output.strides()[1];
    stride_h_output = output.strides()[2];
  }
  else
  {
    num_heads = input.shape()[1];
    head_dim = input.shape()[2];
    stride_d_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_d_output = output.strides()[2];
    stride_h_output = output.strides()[1];
  }

  CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
  CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

  constexpr int CTA_SIZE = 256;

  dim3 grid(num_heads, batch_size, head_dim);
  dim3 block(CTA_SIZE);

  auto input_dtype = input.dtype();

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    MeanScaleKernel<64, false, c_type><<<grid, block>>>(
      reinterpret_cast<c_type*>(input.data()),
      reinterpret_cast<int8_t*>(output.data()),
      nullptr,
      reinterpret_cast<float*>(scale.data()),
      scale_max,
      num_tokens,
      stride_bz_input, stride_d_input, stride_h_input,
      stride_bz_output, stride_d_output, stride_h_output,
      0, 0,
      scale.strides()[0], scale.strides()[1]
    );
  });
}

PD_BUILD_OP(scale_fuse_quant_cuda)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"num_tokens: int", "scale_max: float", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(scale_fuse_quant_cuda_fwd));

// smooth v
void mean_scale_fuse_quant_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& mean,
                paddle::Tensor& scale,
                int num_tokens,
                float scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(mean);
  CHECK_CUDA(scale);

  // CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(mean, paddle::DataType::FLOAT32);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.shape()[0];
  const int num_tokens_padded = input.shape()[3];

  int stride_bz_input = input.strides()[0];
  int stride_bz_output = output.strides()[0];

  int num_heads, head_dim;
  int stride_d_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_heads = input.shape()[2];
    head_dim = input.shape()[1];
    stride_d_input = input.strides()[1];
    stride_h_input = input.strides()[2];
    stride_d_output = output.strides()[1];
    stride_h_output = output.strides()[2];
  }
  else
  {
    num_heads = input.shape()[1];
    head_dim = input.shape()[2];
    stride_d_input = input.strides()[2];
    stride_h_input = input.strides()[1];
    stride_d_output = output.strides()[2];
    stride_h_output = output.strides()[1];
  }

  CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
  CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
  CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

  constexpr int CTA_SIZE = 256;

  dim3 grid(num_heads, batch_size, head_dim);
  dim3 block(CTA_SIZE);

  auto input_dtype = input.dtype();

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    MeanScaleKernel<64, true, c_type><<<grid, block>>>(
      reinterpret_cast<c_type*>(input.data()),
      reinterpret_cast<int8_t*>(output.data()),
      reinterpret_cast<float*>(mean.data()),
      reinterpret_cast<float*>(scale.data()),
      scale_max,
      num_tokens,
      stride_bz_input, stride_d_input, stride_h_input,
      stride_bz_output, stride_d_output, stride_h_output,
      mean.strides()[0], mean.strides()[1],
      scale.strides()[0], scale.strides()[1]
    );
  });
}

PD_BUILD_OP(mean_scale_fuse_quant_cuda)
    .Inputs({"input", "output", "mean", "scale"})
    .Outputs({"out1", "out2", "out3", "out4"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"mean", "out3"}, {"scale", "out4"}}) // Inplace
    .Attrs({"num_tokens: int", "scale_max: float", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(mean_scale_fuse_quant_cuda_fwd));