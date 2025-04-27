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

#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

#include "sageattn_utils.cuh"
#include "sageattn_fused_varlen.cuh"

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

// (Notice) This varlen kernel may introduce extra accuracy loss
template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, bool has_sm_scale = false, bool sub_mean = false, typename T>
__global__ void QuantInt8Kernel_Varlen(T *__restrict__ input, T *__restrict__ mean, 
                                      int8_t *__restrict__ output, float *__restrict__ scale, 
                                      uint32_t *__restrict__ cu_seqlen,
                                      float sm_scale,
                                      const uint32_t stride_seq_input, const uint32_t stride_h_input,
                                      const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                                      const uint32_t stride_seq_output, const uint32_t stride_h_output,
                                      const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T x_val[num_pack_per_thread][8]; // [1][8]
  T mean_val[8];
  float x_val_float[num_pack_per_thread][8];   // [1][8]
  float mean_val_float[8];

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  const uint32_t num_tokens = cu_seqlen[batch_id + 1] - cu_seqlen[batch_id];

  uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;

  uint32_t packed_number = thread_id % num_threads_per_token * pack_size;

  if (bx * BLOCK_SIZE >= num_tokens) return; // let this block finish computing, even if abuntant threads are launched. Because there is a `block Reduce` func in the kernel.

  T *input_ptr_base = input + cu_seqlen[batch_id] * stride_seq_input + head_id * stride_h_input + thread_base_token * stride_seq_input + packed_number;

  T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + packed_number;

  int8_t *output_ptr_base = output + cu_seqlen[batch_id] * stride_seq_output + head_id * stride_h_output + thread_base_token * stride_seq_output + packed_number;

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

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread; // 64 / 1 = 64

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

  // char4 x 2 = 8 bytes
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
  // float2 = 8 bytes = 8 x int8 elements
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
__global__ void TransposePadPermuteVarlenKernel(T *__restrict__ input,  // total_seqlen (not padded) x h_kv x head_dim
                            T *__restrict__ output,                     // head_dim x h_kv x padded_total_seqlen
                            uint32_t *__restrict__ cu_seqlen,
                            uint32_t *__restrict__ padded_cu_seqlen,
                            const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_d_output, const uint32_t stride_h_output)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  uint32_t num_threads_per_token = head_dim / pack_size;  // 128 / 8 = 16 threads per token
  uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

  uint32_t bx = blockIdx.x;         // max_seq_len / 64
  uint32_t head_id = blockIdx.y;    
  uint32_t batch_id = blockIdx.z;   // bsz_id
  uint32_t thread_id = threadIdx.x;

  const uint32_t h_kv = blockDim.y;

  uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;   // 1024 threads -> 64 tokens per block
  const uint32_t bz_seqlen = padded_cu_seqlen[batch_id + 1] - padded_cu_seqlen[batch_id];
  if (thread_base_token >= bz_seqlen) return; // must be >=  !!

  // the problem is, bx ranges from [0, max_seqlen // 64], so thread_base_token can sure cover each seqlen in the batch. And we have done padding
  // so, there is no need to re-check the output bound, when writing into Output.

  // recompute
  // T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;

  T *input_ptr_base = input + 
                      padded_cu_seqlen[batch_id] * stride_seq_input +   // v here is padded by force, so just use `padded_cu_seqlen`
                      head_id * stride_h_input + 
                      thread_base_token * stride_seq_input + 
                      thread_id % num_threads_per_token * pack_size;

  // recompute: [stride_seq_input = h_kv x head_dim]
  // analysis: the original output shape: [b, head_dim, num_head, seq_len]
  // analysis: the original stride_h_output = seq_len
  // analysis: the original stride_d_output = num_head x seq_len
  // analysis: the original stride_bz_output = num_head x seq_len x head_dim

  // analysis: current output shape: [head_dim, num_head, padded_total_seqlen]
  // analysis: current stride_h_output = padded_total_seqlen
  // analysis: current stride_d_output = num_head x padded_total_seqlen

  // T* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size + thread_id / num_threads_per_cta * stride_d_output;

  // 引入更多seqlen之后，是不是后面的会复写前面的
  T* output_ptr_base = output + 
                      padded_cu_seqlen[batch_id] + 
                      head_id * stride_h_output + 
                      bx * CTA_SIZE + 
                      thread_id % num_threads_per_cta * pack_size + 
                      thread_id / num_threads_per_cta * stride_d_output;

  __shared__ T shared_load[CTA_SIZE][head_dim];
  __shared__ T shared_store[head_dim][CTA_SIZE];

  // 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
  // permute on the seq dimension for fp8 mma
  uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
  uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
  uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod  / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);

  constexpr cp_async::SharedMemFillMode fill_mode = pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
  cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size, input_ptr_base, thread_base_token < bz_seqlen);
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
    shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];  // 8 x 16 bit
  }

  __syncthreads();

  *(float4*)(output_ptr_base) = *(float4*)(&shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);  // 4 x 32 bit
}

// this kernel is used to sub mean, and get the v scale, and get fp8 v.
template<uint32_t pad_size, bool sub_mean = false, typename T>
__global__ void MeanScaleVarlenKernel(T *__restrict__ input,  // [head_dim, num_head, total_padded_seqlen]
                            int8_t *__restrict__ output,      // [head_dim, num_head, total_padded_seqlen]
                            float *__restrict__ mean, float *__restrict__ scale, 
                            uint32_t *__restrict__ padded_cu_seqlen,
                            const float scale_max, 
                            const uint32_t num_tokens,  // max_seqlen_v, unpadded
                            const uint32_t stride_d_input, const uint32_t stride_h_input,
                            const uint32_t stride_d_output, const uint32_t stride_h_output,
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

  const uint32_t num_head = gridDim.x;
  const uint32_t head_dim = gridDim.z;
  // pad the number of tokens to 16 to deal with fp8 permute in previous kernel
  uint32_t bz_seqlen = padded_cu_seqlen[batch_id + 1] - padded_cu_seqlen[batch_id];
  uint32_t fp8_padded_num_tokens = (bz_seqlen + 15) / 16 * 16;
  uint32_t num_iters = fp8_padded_num_tokens / gmem_stride + ((fp8_padded_num_tokens % gmem_stride) > thread_id * pack_size);

  // T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + d_id * stride_d_input + thread_id * pack_size;
  T *input_ptr_base = input + 
                      padded_cu_seqlen[batch_id] + 
                      head_id * stride_h_input + 
                      d_id * stride_d_input + 
                      thread_id * pack_size;
  // int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + d_id * stride_d_output + thread_id * pack_size;
  int8_t *output_ptr_base = output + 
                            padded_cu_seqlen[batch_id] + 
                            head_id * stride_h_output + 
                            d_id * stride_d_output + 
                            thread_id * pack_size;

  T x_val[8];
  float x_val_float[8];  // fp32 x 8
  uint32_t x_val_fp8[2]; // fp8  x 8

  float max_val = - 1000000.0f;
  float min_val = 1000000.0f;
  float sum_val = 0.0f;

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
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
  uint32_t padded_num_tokens = (bz_seqlen + pad_size - 1) / pad_size * pad_size;
  num_iters = padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
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
void quant_per_block_int8_fuse_sub_mean_varlen_cuda_fwd(
                paddle::Tensor& input,  // total_seq_len x num_heads x head_dim
                paddle::Tensor& mean,   // bsz x num_heads x head_dim
                paddle::Tensor& output, // total_seq_len x num_heads x head_dim
                paddle::Tensor& scale,  // bsz x num_heads x (total_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE
                paddle::Tensor& cu_seqlen,
                int max_seq_len_q,
                int block_size)  // BLKK: 64
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

  CHECK_DIMS(input, 3);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(output, 3);
  CHECK_DIMS(scale, 3);

  const int batch_size = cu_seqlen.shape()[0] - 1;
  const int head_dim = input.shape()[2];

  int num_tokens = max_seq_len_q;
  int num_heads = input.shape()[1];

  int stride_seq_input = input.strides()[0];
  int stride_h_input = input.strides()[1];
  int stride_seq_output = output.strides()[0];
  int stride_h_output = output.strides()[1];

  auto input_dtype = input.dtype();
  auto mean_dtype = mean.dtype();

  PD_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");
  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM, {
        CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
        CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2]);
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel_Varlen<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, true, c_type><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data()),
          reinterpret_cast<c_type*>(mean.data()),
          output.data<int8_t>(),
          reinterpret_cast<float*>(scale.data()),
          reinterpret_cast<uint32_t*>(cu_seqlen.data()),
          0.0f,
          stride_seq_input, stride_h_input,
          mean.strides()[0], mean.strides()[1],
          stride_seq_output, stride_h_output,
          scale.strides()[0], scale.strides()[1]
        );
      });
    });
  });
}

void quant_per_warp_int8_varlen_cuda_fwd(
                paddle::Tensor& input,  // total_seqlen x num_head x head_dim
                paddle::Tensor& output, // total_seqlen x num_head x head_dim
                paddle::Tensor& scale,  // bsz x num_head x max_seq_len
                paddle::Tensor& cu_seqlen_q,
                int max_seq_len_q,
                int block_size,     // BLKQ: 128
                int warp_block_size) // WARPQ: 32
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, paddle::DataType::INT8);
  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 3);   // not bsz dim now
  CHECK_DIMS(output, 3);  // not bsz dim now
  CHECK_DIMS(scale, 3);

  const int batch_size = cu_seqlen_q.shape()[0] - 1;
  const int head_dim = input.shape()[2];

  const int num_tokens = max_seq_len_q;
  const int num_heads = input.shape()[1];

  const int stride_seq_input = input.strides()[0];
  const int stride_h_input = input.strides()[1];
  const int stride_seq_output = output.strides()[0];
  const int stride_h_output = output.strides()[1];

  auto input_dtype = input.dtype();

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
      DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
        DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM, {
          CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2]);
          CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE));
          dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);  // [num_tokens / 128 x (128 / 32), num_heads, bsz]
          constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024; // 1

          dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

          // printf("Launch params: grid: (%d %d %d), block: %d\n", grid.x, grid.y, grid.z, block.x);
          // printf("Block size: %d, Warp block size: %d\n", BLOCK_SIZE, WARP_BLOCK_SIZE);

          QuantInt8Kernel_Varlen<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<grid, block>>>(
            reinterpret_cast<c_type*>(input.data()),
            nullptr,
            output.data<int8_t>(),
            reinterpret_cast<float*>(scale.data()),
            reinterpret_cast<uint32_t*>(cu_seqlen_q.data()),
            0.0,
            stride_seq_input, stride_h_input,
            0, 0,
            stride_seq_output, stride_h_output,
            scale.strides()[0], scale.strides()[1]
          );
        });
      });
    });
  });
}


// quant v用，但是v不是192，所以可以沿用原来的DISPATCH_HEAD_DIM
// varlen quant v
void transpose_pad_permute_varlen_cuda_fwd(
                paddle::Tensor& input,        // total_seqlen (not padded) x h_kv x head_dim
                paddle::Tensor& output,       // head_dim x h_kv x padded_total_seq_len
                paddle::Tensor& cu_seqlen,
                paddle::Tensor& padded_cu_seqlen,
                int max_seq_len_v,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);

  CHECK_DIMS(input, 3);
  CHECK_DIMS(output, 3);

  constexpr int CTA_SIZE = 64;

  const int batch_size = cu_seqlen.shape()[0] - 1;
  const int head_dim = input.shape()[2];

  int num_tokens = max_seq_len_v;
  int num_heads = input.shape()[1];

  int stride_seq_input = input.strides()[0];  // h_kv x head_dim
  int stride_h_input = input.strides()[1];    // head_dim

  int stride_d_output = output.strides()[0];  // h_kv x padded_total_seq_len
  int stride_h_output = output.strides()[1];  // padded_total_seq_len

  int padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;  // this is aiming at settling the grid size. This value would be OK.

  auto input_dtype = input.dtype();
  auto output_dtype = output.dtype();

  PD_CHECK(input_dtype == output_dtype, "Input and output must have the same data type");

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 grid(padded_num_tokens / CTA_SIZE, num_heads, batch_size); // [max_pad_seqlen // 64, num_head, bsz]

      static_assert(CTA_SIZE * HEAD_DIM <= 8192);

      dim3 block(CTA_SIZE * (HEAD_DIM / 8));    // 64 x (128 / 8) = 64 x 16 = 1024

      TransposePadPermuteVarlenKernel<HEAD_DIM, CTA_SIZE, true, c_type><<<grid, block>>>(
        reinterpret_cast<c_type*>(input.data()),
        reinterpret_cast<c_type*>(output.data()),
        reinterpret_cast<uint32_t*>(cu_seqlen.data()), 
        reinterpret_cast<uint32_t*>(padded_cu_seqlen.data()), 
        stride_seq_input, stride_h_input,
        stride_d_output, stride_h_output
      );
    });
  });
}

// smooth v
void scale_fuse_quant_varlen_cuda_fwd(
                paddle::Tensor& input,  // transpose_permuted_padded_v. [head_dim, num_head, total_padded_seqlen]
                paddle::Tensor& output, // fp8                          [head_dim, num_head, total_padded_seqlen]
                paddle::Tensor& scale,  // [b, num_head, head_dim]
                paddle::Tensor& padded_cu_seqlen,
                int max_seqlen_v, // unpadded max seqlen
                float scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  CHECK_DTYPE(scale, paddle::DataType::FLOAT32);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 3);
  CHECK_DIMS(output, 3);
  CHECK_DIMS(scale, 3);

  const int batch_size = padded_cu_seqlen.shape()[0] - 1;
  const int num_tokens_padded = (max_seqlen_v + 63) / 64 * 64;

  int num_heads, head_dim;
  int stride_d_input, stride_h_input, stride_d_output, stride_h_output;

  num_heads = input.shape()[1];
  head_dim = input.shape()[0];

  stride_d_input = input.strides()[0];
  stride_h_input = input.strides()[1];
  stride_d_output = output.strides()[0];
  stride_h_output = output.strides()[1];

  CHECK_SHAPE(output, input.shape()[0], input.shape()[1], input.shape()[2]);
  CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

  constexpr int CTA_SIZE = 256;

  // this grid & block design, is seqlen-no-aware
  dim3 grid(num_heads, batch_size, head_dim);
  dim3 block(CTA_SIZE);

  auto input_dtype = input.dtype();

  DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    MeanScaleVarlenKernel<64, false, c_type><<<grid, block>>>(
      reinterpret_cast<c_type*>(input.data()),
      reinterpret_cast<int8_t*>(output.data()),
      nullptr,
      reinterpret_cast<float*>(scale.data()),
      reinterpret_cast<uint32_t*>(padded_cu_seqlen.data()), 
      scale_max,
      max_seqlen_v,
      stride_d_input, stride_h_input,
      stride_d_output, stride_h_output,
      0, 0,
      scale.strides()[0], scale.strides()[1]
    );
  });
}

void mean_scale_fuse_quant_varlen_cuda_fwd(
                paddle::Tensor& input,  // transpose_permuted_padded_v. [head_dim, num_head, seqlen]
                paddle::Tensor& output, // same shape
                paddle::Tensor& mean,   // [b, num_head, head_dim]
                paddle::Tensor& scale,  // [b, num_head, head_dim]
                paddle::Tensor& padded_cu_seqlen,
                int max_seqlen_v, // unpadded max seqlen
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

  CHECK_DIMS(input, 3);
  CHECK_DIMS(output, 3);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(scale, 3);

  const int batch_size = padded_cu_seqlen.shape()[0] - 1;
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
    MeanScaleVarlenKernel<64, true, c_type><<<grid, block>>>(
      reinterpret_cast<c_type*>(input.data()),
      reinterpret_cast<int8_t*>(output.data()),
      reinterpret_cast<float*>(mean.data()),
      reinterpret_cast<float*>(scale.data()),
      reinterpret_cast<uint32_t*>(padded_cu_seqlen.data()), 
      scale_max,
      max_seqlen_v,
      stride_d_input, stride_h_input,
      stride_d_output, stride_h_output,
      mean.strides()[0], mean.strides()[1],
      scale.strides()[0], scale.strides()[1]
    );
  });
}

//
//  =========== Exposed to Outside API ===========
//

std::vector<paddle::Tensor> per_warp_int8_varlen_cuda_fwd(paddle::Tensor& q,  // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& k,    // total_seqlen x num_head x head_dim
                                                    paddle::Tensor& cu_seqlen_q,
                                                    paddle::Tensor& km,
                                                    int max_seq_len_q,
                                                    int max_seq_len_k,
                                                    int BLKQ,
                                                    int WARPQ,
                                                    int BLKK) 
{
    paddle::Tensor q_int8 = paddle::empty(q.shape(), paddle::DataType::INT8, paddle::GPUPlace());
    paddle::Tensor k_int8 = paddle::empty(k.shape(), paddle::DataType::INT8, paddle::GPUPlace());

    int b = cu_seqlen_q.shape()[0] - 1;

    int h_qo = q.shape()[1];
    int qo_len = max_seq_len_q;
    int h_kv = k.shape()[1];
    int kv_len = max_seq_len_k;

    int head_dim = q.shape()[2];

    paddle::Tensor q_scale = paddle::empty({b, h_qo, ((qo_len + BLKQ - 1) / BLKQ) * (BLKQ / WARPQ)}, paddle::DataType::FLOAT32, paddle::GPUPlace());
    paddle::Tensor k_scale = paddle::empty({b, h_kv, ((kv_len + BLKK - 1) / BLKK)}, paddle::DataType::FLOAT32, paddle::GPUPlace());

    // quant q -> q_int8
    quant_per_warp_int8_varlen_cuda_fwd(q, q_int8, q_scale, 
                                        cu_seqlen_q,
                                        max_seq_len_q, BLKQ, WARPQ);

    // quant k -> k_int8
    quant_per_block_int8_fuse_sub_mean_varlen_cuda_fwd(k, km, k_int8, k_scale, 
                                                        cu_seqlen_q,
                                                        max_seq_len_k, BLKK);

    return {q_int8, q_scale, k_int8, k_scale};
}

std::vector<paddle::Tensor> per_channel_varlen_fp8(paddle::Tensor& v,                 // total_seqlen x num_head x head_dim
                                                  paddle::Tensor& cu_seqlen_v,        // not padded
                                                  paddle::Tensor& padded_cu_seqlen,   // padded
                                                  int padded_total_seq_len,
                                                  int max_seq_len_v,
                                                  int tensor_layout,
                                                  float scale_max,
                                                  bool smooth_v)
{
    // Notice: this function will pad v to 64-aligned. SM90 arch need to pad 128-align manually.
    int b = cu_seqlen_v.shape()[0] - 1;
    int head_dim = v.shape()[2];
    int h_kv = v.shape()[1];

    int kv_len = max_seq_len_v; // just the max seqlen v, not padded.
    // int padded_len = (kv_len + 63) / 64 * 64;
    PD_CHECK(padded_total_seq_len % 128 == 0 || padded_total_seq_len % 64 == 0, "v must be 64 or 128 padded");

    // note: there may be a bug. We use zeros method instead of empty method.
    paddle::Tensor v_transposed_permutted = paddle::zeros({head_dim, h_kv, padded_total_seq_len}, v.dtype(), paddle::GPUPlace());
    
    transpose_pad_permute_varlen_cuda_fwd(v, v_transposed_permutted, 
                                          cu_seqlen_v, padded_cu_seqlen,
                                          max_seq_len_v, tensor_layout);

    paddle::Tensor v_fp8 = paddle::empty(v_transposed_permutted.shape(), paddle::DataType::FLOAT8_E4M3FN, paddle::GPUPlace());
    paddle::Tensor v_scale = paddle::empty({b, h_kv, head_dim}, paddle::DataType::FLOAT32, paddle::GPUPlace());
    paddle::Tensor vm = paddle::empty({b, h_kv, head_dim}, paddle::DataType::FLOAT32, paddle::GPUPlace());
    if (smooth_v) {
        // not supported.
    } else {
        scale_fuse_quant_varlen_cuda_fwd(v_transposed_permutted, v_fp8, v_scale, 
                                        padded_cu_seqlen, 
                                        kv_len, scale_max, tensor_layout);
    }

    return {v_fp8, v_scale, vm};
}
