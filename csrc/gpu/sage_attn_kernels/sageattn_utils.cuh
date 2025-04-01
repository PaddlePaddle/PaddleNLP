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

/*
 * This file is adapted from
 *   https://github.com/thu-ml/SageAttention/blob/main/csrc/*.cuh.
 *   The original license is kept as-is:
 *
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cuda_pipeline_primitives.h>

#include <cstdint>
#include <type_traits>
#include <sstream>
#include <stdexcept>

#include "paddle/extension.h"

// currently we do not support INT4, so we implement INT8 sage attention inference temporarily

#define FINAL_MASK 0xffffffff
#define WARP_SIZE 32

#define S_FP8_OFFSET 8.807f
#define S_FP8_OFFSET_EXP 6680.8477f
#define S_FP8_OFFSET_EXP_INV 0.0022326917f

#define div_ceil(M, N) (((M) + (N)-1) / (N))

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FP8_CAST_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define RUNTIME_ASSERT(x) __brkpt()
#else
#include <assert.h>
#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

// dispatch_utils.h
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported head dim: " << int(head_dim);       \
    throw std::invalid_argument(err_msg.str());                 \
  }

// add new support to HEAD_DIM = 192 for deepseek
#define DISPATCH_HEAD_DIM_QK(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else if (head_dim == 192) {                                 \
    constexpr int HEAD_DIM = 192;                               \
    __VA_ARGS__                                                 \
  } else if (head_dim == 256) {                                 \
    constexpr int HEAD_DIM = 256;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported head dim: " << int(head_dim);       \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)              \
  if (is_causal == 1) {                                         \
    constexpr bool IS_CAUSAL = true;                            \
    __VA_ARGS__                                                 \
  } else if (is_causal == 0) {                                  \
    constexpr bool IS_CAUSAL = false;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported causal mode: " << int(is_causal);   \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, ...)              \
  if (qk_quant_gran == 2) {                                         \
    constexpr int QK_QUANT_GRAN = 2;                            \
    __VA_ARGS__                                                 \
  } else if (qk_quant_gran == 3) {                                  \
    constexpr int QK_QUANT_GRAN = 3;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported qk_quant_gran: " << int(qk_quant_gran);   \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, ...)             \
  if (return_lse == 1) {                                         \
    constexpr bool RETURN_LSE = true;                            \
    __VA_ARGS__                                                  \
  } else if (return_lse == 0) {                                  \
    constexpr bool RETURN_LSE = false;                           \
    __VA_ARGS__                                                  \
  }  else {                                                      \
    std::ostringstream err_msg;                                  \
    err_msg << "Unsupported causal mode: " << int(return_lse);   \
    throw std::invalid_argument(err_msg.str());                  \
  }

// DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16
// here we will use paddle's DataType
#define DISPATCH_PADDLE_DTYPE_TO_CTYPE_FP16(paddle_dtype, c_type, ...)                \
  if (paddle_dtype == paddle::DataType::FLOAT16) {                                          \
    using c_type = half;                                                                \
    __VA_ARGS__                                                                         \
  } else if (paddle_dtype == paddle::DataType::BFLOAT16) {                               \
    using c_type = nv_bfloat16;                                                         \
    __VA_ARGS__                                                                         \
  } else {                                                                              \
    std::ostringstream oss;                                                             \
    oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << paddle_dtype;    \
    PD_CHECK(false, oss.str());                                                      \
  }

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...)        \
  if (block_size == 64) {                                       \
    constexpr int BLOCK_SIZE = 64;                              \
    __VA_ARGS__                                                 \
  } else if (block_size == 128) {                               \
    constexpr int BLOCK_SIZE = 128;                             \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported block_size " << int(block_size);    \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, ...)  \
  if (warp_block_size == 16) {                                           \
    constexpr int WARP_BLOCK_SIZE = 16;                                  \
    __VA_ARGS__                                                          \
  } else if (warp_block_size == 32) {                                    \
    constexpr int WARP_BLOCK_SIZE = 32;                                  \
    __VA_ARGS__                                                          \
  }  else {                                                              \
    std::ostringstream err_msg;                                          \
    err_msg << "Unsupported warp_block_size " << int(warp_block_size);   \
    throw std::invalid_argument(err_msg.str());                          \
  }

// define the macro for necessary checks, originally in `utils.cuh`
#define CHECK_CUDA(x) \
  PD_CHECK(x.is_gpu(), "Tensor " #x " must be on CUDA") // shift to paddle API: is_gpu()

// CHECK_DTYPE aims at testing the tensor datatype, use paddle::DataType
#define CHECK_DTYPE(x, true_dtype)     \
  PD_CHECK(x.dtype() == true_dtype, \
              "Tensor " #x " must have dtype (" #true_dtype ")")  // DataType dtype() const;
#define CHECK_DIMS(x, true_dim)    \
  PD_CHECK(x.dims().size() == true_dim, \
              "Tensor " #x " must have dimension number (" #true_dim ")") // paddle API: .dims().size()
#define CHECK_NUMEL(x, minimum)     \
  PD_CHECK(x.numel() >= minimum, \
              "Tensor " #x " must have at last " #minimum " elements")
#define CHECK_SHAPE(x, ...)                                   \
  PD_CHECK(x.dims() == common::DDim({__VA_ARGS__}), \
              "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  PD_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")   // TODO: check if valid
#define CHECK_LASTDIM_CONTIGUOUS(x) \
  PD_CHECK(x.strides().at(x.strides().size() - 1) == 1,    \
              "Tensor " #x " must be contiguous at the last dimension")


namespace sageattn {

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T) (0.0f);
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockAllReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (lane < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[i][lane] : (T) (0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T) 0.0f;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}
/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;  // in-warp idx
    int                 wid  = threadIdx.x >> 5;    // warp idx
    val = warpReduceMax(val);  // get maxx in each warp
    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;
    __syncthreads();
    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);
    return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockAllReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5;    // warp idx

    val = warpReduceMax(val);      // get maxx in each warp

    if (lane == 0)                 // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMin(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = min(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}
/* Calculate the minimum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMin(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;  // in-warp idx
    int                 wid  = threadIdx.x >> 5;    // warp idx
    val = warpReduceMin(val);  // get minx in each warp
    if (lane == 0)  // record in-warp minx by warp Idx
        shared[wid] = val;
    __syncthreads();
    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : 1e20f;
    val = warpReduceMin(val);
    return val;
}

} // namespace sageattn

namespace mma{

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define MMA_F16F16F32_M16N8K16_ENABLED
#define MMA_F16F16F16_M16N8K16_ENABLED
#define MMA_S8S8S32_M16N8K32_ENABLED
#define MMA_S4S4S32_M16N8K64_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define MMA_F16F16F32_M16N8K8_ENABLED
#define MMA_F16F16F16_M16N8K8_ENABLED
#define LDMATRIX_M8N8X2_ENABLED
#define LDMATRIX_M8N8X4_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define MMA_F8F8F32_M16N8K16_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define RUNTIME_ASSERT(x) __brkpt()
#else
#include <assert.h>
#define RUNTIME_ASSERT(x) assert(0 && x)
#endif

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x2 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x2(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X2_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R, T* smem_ptr) {
#ifdef LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  // ! only support half dtype now
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  // ! only support half dtype now
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f16.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F16_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k16 instruction for row major and column major f16 matrix
 *   multiplication, accumulated in f16.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#ifdef MMA_F16F16F16_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major int8 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S8S8S32_M16N8K32_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column major int8 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_s8s8s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S8S8S32_M16N8K32_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
          "r"(C[6]), "r"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major int4 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k64_row_col_s4s4s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S4S4S32_M16N8K64_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k64 instruction for row major and column major int4 matrix
 *   multiplication, accumulated in int32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k64_row_col_s4s4s32(int32_t* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_S4S4S32_M16N8K64_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[4]), "r"(C[5]),
          "r"(C[6]), "r"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0),
          "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[4]), "=r"(C[5]), "=r"(C[6]), "=r"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0),
          "r"(0), "r"(0));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n8k32 instruction for row major and column major fp8 e4m3 matrix
 *   multiplication, accumulated in fp32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_f8f8f32(float* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of the mma m16n16k32 instruction for row major and column major fp8 matrix
 *   multiplication, accumulated in fp32.
 * \tparam mma_mode The mode of mma instruction, either kInit or kInplaceUpdate
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f32(float* C, uint32_t* A,
                                                                   uint32_t* B) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  if constexpr (mma_mode == MMAMode::kInplaceUpdate)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
          "f"(C[2]), "f"(C[3]));
    
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
          "f"(C[6]), "f"(C[7]));
  }
  else if constexpr (mma_mode == MMAMode::kInit)
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
          "f"(0.f), "f"(0.f));
  }
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
__device__ __forceinline__ void rowsum_f16f16f32(float* d, uint32_t* s) {
#ifdef MMA_F16F16F32_M16N8K16_ENABLED
  asm volatile(
      "{\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  0.,  %9,  0.};\n"
      "}\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(1006648320), // 1006648320 packs two 1.0f in half precision
        "r"(1006648320), "f"(d[0]), "f"(d[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
__device__ __forceinline__ void rowsum_f8f8f32(float* d, uint32_t* s) {
#ifdef MMA_F8F8F32_M16N8K16_ENABLED
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  0.,  %9,  0.};\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s[0]), "r"(s[1]), "r"(s[2]), "r"(s[3]), "r"(943208504), "r"(943208504), // 943208504 packs four 1.0f in e4m3
        "f"(d[0]), "f"(d[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

} // namespace mma


// namespace cp_async
// intend to wrap the copy asynchronizely operations
namespace cp_async {

enum class SharedMemFillMode {
  kFillZero,  // Fill zero to shared memory when predicate is false
  kNoFill     // Do not fill zero to shared memory when predicate is false
};

enum class PrefetchMode {
  kNoPrefetch,  // Do not fetch additional data from global memory to L2
  kPrefetch     // Fetch additional data from global memory to L2
};

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_ENABLED
#endif
#endif

/*!
 * \brief Wrapper of PTX cp.async.commit_group instruction, commit all prior uncommitted
 *   cp.async instructions to a group
 */
__device__ __forceinline__ void commit_group() {
#ifdef CP_ASYNC_ENABLED
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.wait_group instruction
 * \tparam n Wait till most recent n groups are committed
 */
template <size_t n>
__device__ __forceinline__ void wait_group() {
#ifdef CP_ASYNC_ENABLED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from
 *   global memory to shared memory
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 */
template <PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
#ifdef CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  }
#else
  *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from
 *   global memory to shared memory with predicate.
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is false
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 * \param predicate Predicate value
 * \note fill zero is slower than not fill zero
 */
template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
#ifdef CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 16 : 0;
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    } else {
      asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    }
  } else {
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    } else {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    }
  }
#else
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  } else {
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}

} // namespace cp_async

#ifndef USHORT_TYPE
#define USHORT_TYPE
typedef unsigned short ushort;
#endif

// namespace math
// math operations using ptx
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;
constexpr float log2e_recp = 1.0f / log2e;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return __shfl_xor_sync(0xffffffff, x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}

}  // namespace math


// originally in `permuted_smem.cuh`
enum class SwizzleMode {
  k32B, // for k32B mode, a line of shared memory must have 32B (16 half value)
  k64B, // for k64B mode, a line of shared memory must have 64B (32 half value)
  k128B, // 128B already spans all banks in shared memory. a line of shared memory can have multiple 128B.
};

// Use 128bit as the granularity to fetch/store data per thread to maximize memory bandwidth
using b128_t = uint4;

/*!
 * \brief A stateless shared memory wrapper that uses templates to avoid runtime conditionals. It makes sure
 * that access to consecutive rows idx in the same column idx will make full use of the shared memory bank through
 * permutation in the granularity of 128bit.
 * 
 * This struct treats all offsets to be the number of `b128_t` elements. It is designed to be stateless,
 * meaning it does not maintain any information about the current pointer position. The offset returned by 
 * the struct can be used to access the shared memory through the provided interface.
 * 
 * The struct guarantees that the read to permuted offset (i, j) will be the value stored in permuted offset (i, j).
 * We assume that shared memory operation operates on at least two consecutive 128-bit values in a row within a warp.
 * Under this assumption, we do not permute for k32B mode.
 */
template <SwizzleMode swizzle_mode, uint32_t stride>
struct smem_t {
  // The base pointer.
  b128_t* base;
  // How many b128_t value a row contains
  // uint32_t stride;

  __device__ __forceinline__ smem_t() : base(nullptr) {}
  template <typename T>
  __device__ __forceinline__ smem_t(T* base) : base((b128_t*)base) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(stride % 8 == 0, "Stride must be multiple of 8 for 128B swizzle mode");
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(stride == 4, "Stride must be 4 for 64B swizzle mode");
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      static_assert(stride == 2, "Stride must be 2 for 32B swizzle mode");
    } else {
      static_assert(swizzle_mode != swizzle_mode, "Unsupported swizzle mode");      
    }
  }

  /*!
   * \brief Set the base pointer.
   */
  template <typename T>
  __device__ __forceinline__ void set_base(T* new_base) {
    base = (b128_t*)new_base;
  }

  /*!
   * \brief Compute the element offset given coordinates in a permuted shared memory.
   * \param i The row index.
   * \param j The column index.
   */
  static __device__ __forceinline__ uint32_t get_permuted_offset(const uint32_t &i, const uint32_t &j) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      return i * stride + (j ^ (i % 8));
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      return i * stride + (j ^ ((i / 2) % 4));
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      return i * stride + j;
    }
  }

  /*!
  * \tparam step_size The step size to advance the offset in the permuted shared memory.
  * \param offset The current offset. 
  */
  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(const uint32_t &offset) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size % 8 == 0,
                    "Unsupported step size");
      return offset + step_size;
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(step_size == 4, "Unsupported step size");
      return offset + step_size;
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      static_assert(step_size == 2, "Unsupported step size");
      return offset + step_size;
    }
  }

  // ! use with care
  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(const uint32_t &offset, const uint32_t &step_idx) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + (step_idx % 4 == 3) * 8;
      } else if constexpr (step_size == 4) {
        return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
      } else {
        // step_size % 8 == 0
        return offset + step_size;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(step_size == 2 || step_size == 4, "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ 0x2) + (step_idx % 2 == 1) * 4;
      } else {
        return offset + step_size;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      return offset + step_size;
    }
  }

  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_row(const uint32_t &offset) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x4) + step_size * stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * stride;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x2) + step_size * stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * stride;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k32B) {
      return offset + step_size * stride;
    }
  }

  __device__ __forceinline__ void ldmatrix_m8n8x2(const uint32_t &offset, uint32_t* R) const {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x2(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(const uint32_t &offset, uint32_t* R) const {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(const uint32_t &offset, uint32_t* R) const {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
  }

  template <cp_async::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_128b_async(const uint32_t &offset, const T* gptr, bool predicate) const {
    b128_t* smem_ptr = base + offset;
    cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b128_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void load_128b_async(const uint32_t &offset, const T* gptr) const {
    b128_t* smem_ptr = base + offset;
    cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(smem_ptr, reinterpret_cast<const b128_t*>(gptr));
  }

  template <typename T>
  __device__ __forceinline__ void store_128b(const uint32_t &offset, T* gptr) const {
    *reinterpret_cast<b128_t*>(gptr) = *(base + offset);
  }
};


// numeric conversion
__device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t *dest, float *source0, float *source1)
{
#ifdef FP8_CAST_ENABLED
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void floatx4_to_e5m2x4(uint32_t *dest, float *source0, float *source1)
{
#ifdef FP8_CAST_ENABLED
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e5m2x2.f32   lo, %2, %1;\n" \
      "cvt.rn.satfinite.e5m2x2.f32   hi, %4, %3;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "f"(source0[0]), "f"(source1[1]), "f"(source1[0]), "f"(source1[1]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void halfx4_to_e4m3x4(uint32_t *dest, uint32_t *source0, uint32_t *source1)
{
#ifdef FP8_CAST_ENABLED
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e4m3x2.f16x2   lo, %1;\n" \
      "cvt.rn.satfinite.e4m3x2.f16x2   hi, %2;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "r"(source0[0]), "r"(source1[0]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void halfx4_to_e5m2x4(uint32_t *dest, uint32_t *source0, uint32_t *source1)
{
#ifdef FP8_CAST_ENABLED
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e5m2x2.f16x2   lo, %1;\n" \
      "cvt.rn.satfinite.e5m2x2.f16x2   hi, %2;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "r"(source0[0]), "r"(source1[0]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void e4m3x4_to_halfx4(uint32_t *dest0, uint32_t *dest1, uint32_t *source)
{
#ifdef FP8_CAST_ENABLED
  asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n" \
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n" \
      "}\n" : "=r"(dest0[0]), "=r"(dest1[0]) : "r"(source[0]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ void e5m2x4_to_halfx4(uint32_t *dest0, uint32_t *dest1, uint32_t *source)
{
#ifdef FP8_CAST_ENABLED
  asm volatile( \
      "{\n" \
      ".reg .b16 lo, hi;\n" \
      "mov.b32 {lo, hi}, %2;\n" \
      "cvt.rn.f16x2.e5m2x2 %0, lo;\n" \
      "cvt.rn.f16x2.e5m2x2 %1, hi;\n" \
      "}\n" : "=r"(dest0[0]), "=r"(dest1[0]) : "r"(source[0]));
#else
  RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ int8_t float_to_int8_rn(float x)
{
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t&>(dst);
}


// attn_utils
enum class MaskMode {
    kNone = 0,
    kCausal = 1,
};

// we do not use paddle::DataType, because paddle or torch do not support INT4 natively. We now use our defined DataType.
// To avoid conflict with paddle DataType or something, we rename the `DataType` here to `SADataType`
enum class SADataType {
    kHalf,
    kInt8,
    kInt4,
    kE4M3,
    kE5M2,
};

enum class QuantGranularity {
    kPerTensor = 0,
    kPerBlock = 1,
    kPerWarp = 2,
    kPerThread = 3,
};

enum class ComputeUnit {
  kTensorCore,
  kCudaCore,
};

__device__ __forceinline__ uint32_t get_warp_id()
{
  return threadIdx.y;
}

__device__ __forceinline__ uint32_t get_lane_id()
{
  return threadIdx.x;
}

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_q()
{
  return get_warp_id() / num_warps_k;
}

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_k()
{
  return get_warp_id() % num_warps_k;
}

template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, 
          uint32_t smem_iters_row, uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA, typename T>
__device__ __forceinline__ void load_global_to_share(T **lane_ptr, uint32_t &smem_offset,
                                                    const uint32_t &gmem_stride,
                                                    const smem_t<swizzle_mode, stride> &smem)
{
  static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
  static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

  constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++)
    {
      smem.load_128b_async(smem_offset, *lane_ptr);
      *lane_ptr += (global_to_shared_line_lanes * pack_size);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size));
  }
  smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

// with predicate
template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, 
          uint32_t smem_iters_row, uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA, typename T>
__device__ __forceinline__ void load_global_to_share(T **lane_ptr, uint32_t &smem_offset,
                                                    const uint32_t &gmem_stride,
                                                    const smem_t<swizzle_mode, stride> &smem, uint32_t base_idx, uint32_t max_len)
{
  static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
  static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

  constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++)
    {
      smem.load_128b_async<cp_async::SharedMemFillMode::kNoFill>(smem_offset, *lane_ptr, base_idx < max_len);
      *lane_ptr += (global_to_shared_line_lanes * pack_size);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size));
    base_idx += global_to_shared_copy_lines_per_warp_per_iter;
  }
  smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

template <uint32_t global_to_shared_line_lanes, uint32_t global_to_shared_copy_lines_per_warp_per_iter, 
          uint32_t smem_iters_row, uint32_t smem_iters_col, SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA>
__device__ __forceinline__ void load_fp8_V_global_to_share(int8_t **lane_ptr, uint32_t &smem_offset,
                                                    const uint32_t &gmem_stride,
                                                    const smem_t<swizzle_mode, stride> &smem)
{
  static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
  constexpr uint32_t pack_size_fp8 = 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++)
    {
      smem.load_128b_async(smem_offset, *lane_ptr);
      *lane_ptr += (global_to_shared_line_lanes * pack_size_fp8);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size_fp8));
  }
  smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  // for QK: *lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
  *lane_ptr += CTA; // ! prevent underflow 
  *lane_ptr -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

template <uint32_t num_warps_q, uint32_t num_warps_k, 
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_qk_inner, 
          SwizzleMode swizzle_mode, uint32_t stride, SADataType DTypeQK>
__device__ __forceinline__ void compute_int_qk(const smem_t<swizzle_mode, stride> &smem_Q, const smem_t<swizzle_mode, stride> &smem_K, int32_t RS[][num_tiles_k][8], uint32_t &offset_Q, uint32_t &offset_K)
{
  static_assert(DTypeQK == SADataType::kInt8 || DTypeQK == SADataType::kInt4);

  uint32_t RQ[num_tiles_q][4];
  uint32_t RK[4];

  // the first iteration, mma mode is kInit
#pragma unroll
  for (uint32_t iter = 0; iter < 1; iter++)
  {
    // load RQ
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
    }
    // ! using permutation invariance
    offset_Q = smem_Q.advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        if constexpr (DTypeQK == SADataType::kInt8)
        {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
        }
        else if constexpr (DTypeQK == SADataType::kInt4)
        {
          mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  // following iteration, mma mode is kInplace
#pragma unroll
  for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++)
  {
    // load RQ
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
    }
    offset_Q = smem_Q.advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        if constexpr (DTypeQK == SADataType::kInt8)
        {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS[fq][fk], RQ[fq], RK);
        }
        else if constexpr (DTypeQK == SADataType::kInt4)
        {
          mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInplaceUpdate>(RS[fq][fk], RQ[fq], RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }

  offset_Q -= (2 * num_tiles_qk_inner);
  offset_K -= (2 * num_tiles_qk_inner);
}

// for case when num_tiles_qk_inner = 1
template <uint32_t num_warps_q, uint32_t num_warps_k, 
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_qk_inner, 
          SwizzleMode swizzle_mode, uint32_t stride, SADataType DTypeQK>
__device__ __forceinline__ void compute_int_qk(const smem_t<swizzle_mode, stride> &smem_K, int32_t RS[][num_tiles_k][8], uint32_t RQ[][4], uint32_t offset_K)
{
  static_assert(DTypeQK == SADataType::kInt8 || DTypeQK == SADataType::kInt4);
  static_assert(num_tiles_qk_inner == 1);

  uint32_t RK[4];

  // mma mode is kInit
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++)
  {
    // load RK
    smem_K.ldmatrix_m8n8x4(offset_K, RK);
    offset_K = smem_K.advance_offset_by_row<16>(offset_K);

    // mma
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      if constexpr (DTypeQK == SADataType::kInt8)
      {
        mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
      }
      else if constexpr (DTypeQK == SADataType::kInt4)
      {
        mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename DTypeQKAccum>
__device__ __forceinline__ void apply_causal_mask(const uint32_t &Q_idx_lane_base, const uint32_t &K_idx_lane_base, DTypeQKAccum RS[][num_tiles_k][8])
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        const uint32_t q_idx = Q_idx_lane_base + fq * 16 + 8 * ((k % 4) / 2);
        const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
        const bool out_of_boundary = (kv_idx > q_idx);

        if constexpr (std::is_same<DTypeQKAccum, float>::value)
        {
          RS[fq][fk][k] = (out_of_boundary ? -5000000.0f : RS[fq][fk][k]);
        }
        else if constexpr (std::is_same<DTypeQKAccum, half>::value)
        {
          RS[fq][fk][k] = (out_of_boundary ? __float2half_rn(-50000.0f) : RS[fq][fk][k]);
        }
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename DTypeQKAccum>
__device__ __forceinline__ void apply_out_of_bound_mask(const uint32_t &K_idx_lane_base, DTypeQKAccum RS[][num_tiles_k][8], const uint32_t &kv_len)
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
        const bool out_of_boundary = (kv_idx >= kv_len);

        if constexpr (std::is_same<DTypeQKAccum, float>::value)
        {
          RS[fq][fk][k] = (out_of_boundary ? -5000000.0f : RS[fq][fk][k]);
        }
        else if constexpr (std::is_same<DTypeQKAccum, half>::value)
        {
          RS[fq][fk][k] = (out_of_boundary ? __float2half_rn(-50000.0f) : RS[fq][fk][k]);
        }
      }
    }
  }
}

// for DTypeQKAccum float
template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, bool use_half_o_scale, bool exp_offset, bool fuse_scale=false, typename DTypeSVAccum>
__device__ __forceinline__ void update_mdo(float RS[][num_tiles_k][8], DTypeSVAccum RO[][num_tiles_v][8], float m[][2], float d[][2], const float &sm_scale)
{
  static_assert(std::is_same<DTypeSVAccum, half>::value || (!use_half_o_scale));
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      // assign the smallest value possible
      float m_prev = m[fq][k];
      float m_temp = -5000000.0f;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        float m_local = max(max(RS[fq][fk][k * 2 + 0], RS[fq][fk][k * 2 + 1]),
                                max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
        m_temp = max(m_temp, m_local);
      }

      if constexpr (!fuse_scale)
      {
        if constexpr (exp_offset)
        {
          m_temp = fmaf(m_temp, sm_scale, -S_FP8_OFFSET);
        }
        else
        {
          m_temp *= sm_scale;
        }
      }
      else if constexpr (exp_offset)
      {        
        m_temp += (-S_FP8_OFFSET);        
      }

      // exchange element with the 4 threads in the row
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1)); // 0 exchange with 1, 2 exchange with 3
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2)); // 0 exchange with 2, 1 exchange with 3

      m[fq][k] = max(m[fq][k], m_temp);

      float o_scale = math::ptx_exp2(m_prev - m[fq][k]);

      // update denominator
      d[fq][k] *= o_scale;

      half2 o_scale2;
      if constexpr (use_half_o_scale)
      {  
        o_scale2 = __floats2half2_rn(o_scale, o_scale);
      }

      // update RO
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value)
        {
          RO[fq][fv][k * 2 + 0] *= o_scale;
          RO[fq][fv][k * 2 + 1] *= o_scale;
          RO[fq][fv][k * 2 + 4] *= o_scale;
          RO[fq][fv][k * 2 + 5] *= o_scale;
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          if constexpr (use_half_o_scale)
          {
            ((half2*)RO[fq][fv])[k] = __hmul2(((half2*)RO[fq][fv])[k], o_scale2);
            ((half2*)RO[fq][fv])[k + 2] = __hmul2(((half2*)RO[fq][fv])[k + 2], o_scale2);
          }
          else
          {
            RO[fq][fv][k * 2 + 0] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 0]) * o_scale);
            RO[fq][fv][k * 2 + 1] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 1]) * o_scale);
            RO[fq][fv][k * 2 + 4] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 4]) * o_scale);
            RO[fq][fv][k * 2 + 5] = __float2half_rn(__half2float(RO[fq][fv][k * 2 + 5]) * o_scale);
          }
        }
      }

      // raise RS to exponent
      float negative_m = -m[fq][k];
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        if constexpr (fuse_scale)
        {
          RS[fq][fk][k * 2 + 0] = math::ptx_exp2(RS[fq][fk][k * 2 + 0] + negative_m);
          RS[fq][fk][k * 2 + 1] = math::ptx_exp2(RS[fq][fk][k * 2 + 1] + negative_m);
          RS[fq][fk][k * 2 + 4] = math::ptx_exp2(RS[fq][fk][k * 2 + 4] + negative_m);
          RS[fq][fk][k * 2 + 5] = math::ptx_exp2(RS[fq][fk][k * 2 + 5] + negative_m);
        }
        else
        {
          RS[fq][fk][k * 2 + 0] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 0], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 1] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 1], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 4] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 4], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 5] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 5], sm_scale, negative_m));
        }
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename T>
__device__ __forceinline__ void RS_32_to_16(T RS[][num_tiles_k][8], uint32_t RS_16[][num_tiles_k][4])
{
  static_assert(sizeof(T) == 4);
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      ((half2*)RS_16[fq][fk])[0] = __float22half2_rn(((float2*)RS[fq][fk])[0]);
      ((half2*)RS_16[fq][fk])[1] = __float22half2_rn(((float2*)RS[fq][fk])[1]);
      ((half2*)RS_16[fq][fk])[2] = __float22half2_rn(((float2*)RS[fq][fk])[2]);
      ((half2*)RS_16[fq][fk])[3] = __float22half2_rn(((float2*)RS[fq][fk])[3]);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_32_to_8(float RS[][num_tiles_k][8], uint32_t RS_8[][num_tiles_k / 2][4])
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
    {
      floatx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 4);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 2, RS[fq][fk * 2 + 0] + 6);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 4);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 2, RS[fq][fk * 2 + 1] + 6);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_16_to_8(uint32_t RS[][num_tiles_k][4], uint32_t RS_8[][num_tiles_k / 2][4])
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
    {
      halfx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_8_to_16(uint32_t RS_8[][num_tiles_k / 2][4], uint32_t RS[][num_tiles_k][4])
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
    {
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2, RS_8[fq][fk]);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3, RS_8[fq][fk] + 1);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2, RS_8[fq][fk] + 2);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3, RS_8[fq][fk] + 3);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, ComputeUnit compute_unit = ComputeUnit::kTensorCore, typename T>
__device__ __forceinline__ void accumulate_d(T RS[][num_tiles_k][(compute_unit == ComputeUnit::kTensorCore)? 4 : 8], float d[][2])
{
  // for compute unit cuda core, RS is float
  // for compute unit tensor core, RS is packed half
  static_assert((std::is_same<T, float>::value && compute_unit == ComputeUnit::kCudaCore) || 
                (std::is_same<T, uint32_t>::value && compute_unit == ComputeUnit::kTensorCore));

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      if constexpr (compute_unit == ComputeUnit::kTensorCore)
      {
        // full accumulate with tensor core
        mma::rowsum_f16f16f32(d[fq], (uint32_t*)(RS[fq][fk]));
      }
      else if constexpr (compute_unit == ComputeUnit::kCudaCore)
      { 
        // partial accumulate with cuda core
        d[fq][0] += RS[fq][fk][0] + RS[fq][fk][1] + RS[fq][fk][4] + RS[fq][fk][5];
        d[fq][1] += RS[fq][fk][2] + RS[fq][fk][3] + RS[fq][fk][6] + RS[fq][fk][7];
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void accumulate_d_f8(uint32_t RS[][num_tiles_k / 2][4], float d[][2])
{
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
    {
      mma::rowsum_f8f8f32(d[fq], RS[fq][fk]);
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k, 
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          SwizzleMode swizzle_mode, uint32_t stride, typename DTypeSVAccum>
__device__ __forceinline__ void compute_fp16_sv(const smem_t<swizzle_mode, stride> &smem_V, uint32_t RS_f16[][num_tiles_k][4], DTypeSVAccum RO[][num_tiles_v][8], float d[][2])
{
  uint32_t smem_V_row_base = get_warp_idx_k<num_warps_q, num_warps_k>() * (num_tiles_k * 16) + get_lane_id() % 16;
  uint32_t smem_V_col_base = get_lane_id() / 16;
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      // load RV
      uint32_t RV[4];
      uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fk * 16, smem_V_col_base + fv * 2);
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) 
        {
          mma::mma_sync_m16n16k16_row_col_f16f16f32(RO[fq][fv], RS_f16[fq][fk], RV);
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)RO[fq][fv], RS_f16[fq][fk], RV);
        }
      }
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k, 
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          SwizzleMode swizzle_mode, uint32_t stride, uint32_t RS_width=4, typename T, typename DTypeSVAccum>
__device__ __forceinline__ void compute_fp16_sv_permuted(const smem_t<swizzle_mode, stride> &smem_V, T RS_f16[][num_tiles_k][RS_width], DTypeSVAccum RO[][num_tiles_v][8], float d[][2], uint32_t &offset_V)
{
  static_assert(sizeof(T) == 4);

  // ! be sure you know what you are doing
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      // load RV
      uint32_t RV[4];
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) 
        {
          mma::mma_sync_m16n16k16_row_col_f16f16f32(RO[fq][fv], (uint32_t*)(RS_f16[fq][fk]), RV);
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)RO[fq][fv], (uint32_t*)(RS_f16[fq][fk]), RV);
        }
      }

      offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
    }
    offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
  }

  // make offset_V their original value
  offset_V -= (16 * num_tiles_k * stride);
}

template <uint32_t num_warps_q, uint32_t num_warps_k, 
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          SwizzleMode swizzle_mode, uint32_t stride, uint32_t RS_width=4, typename T, typename DTypeSVAccum>
__device__ __forceinline__ void compute_fp16_sv_permuted_inst_buf(const smem_t<swizzle_mode, stride> &smem_V, T RS_f16[][num_tiles_k][RS_width], DTypeSVAccum RO[][num_tiles_v][8], float d[][2], uint32_t &offset_V)
{
  static_assert(sizeof(T) == 4);
  static_assert(std::is_same<DTypeSVAccum, float>::value);

  uint32_t RO_inst_buf[num_tiles_q][num_tiles_v][4];

  // ! be sure you know what you are doing
#pragma unroll
  for (uint32_t fk = 0; fk < 1; fk++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      // load RV
      uint32_t RV[4];
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        {
          mma::mma_sync_m16n16k16_row_col_f16f16f16<mma::MMAMode::kInit>((uint32_t*)RO_inst_buf[fq][fv], (uint32_t*)(RS_f16[fq][fk]), RV);
        }
      }

      offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
    }
    offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
  }

#pragma unroll
  for (uint32_t fk = 1; fk < num_tiles_k; fk++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      // load RV
      uint32_t RV[4];
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        {
          mma::mma_sync_m16n16k16_row_col_f16f16f16<mma::MMAMode::kInplaceUpdate>((uint32_t*)RO_inst_buf[fq][fv], (uint32_t*)(RS_f16[fq][fk]), RV);
        }
      }

      offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
    }
    offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
  }

  // accumulate into RO
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      RO[fq][fv][0] += __half2float(((half2*)RO_inst_buf[fq][fv])[0].x);
      RO[fq][fv][1] += __half2float(((half2*)RO_inst_buf[fq][fv])[0].y);
      RO[fq][fv][2] += __half2float(((half2*)RO_inst_buf[fq][fv])[1].x);
      RO[fq][fv][3] += __half2float(((half2*)RO_inst_buf[fq][fv])[1].y);
      RO[fq][fv][4] += __half2float(((half2*)RO_inst_buf[fq][fv])[2].x);
      RO[fq][fv][5] += __half2float(((half2*)RO_inst_buf[fq][fv])[2].y);
      RO[fq][fv][6] += __half2float(((half2*)RO_inst_buf[fq][fv])[3].x);
      RO[fq][fv][7] += __half2float(((half2*)RO_inst_buf[fq][fv])[3].y);
    }
  }

  // make offset_V their original value
  offset_V -= (16 * num_tiles_k * stride);
}

template<uint32_t num_tiles_q, uint32_t num_tiles_v,
       ComputeUnit compute_unit = ComputeUnit::kTensorCore, // compute unit for accumulate_d
       typename DTypeQKAccum, typename DTypeSVAccum>
__device__ __forceinline__ void normalize_d(DTypeSVAccum RO[][num_tiles_v][8], DTypeQKAccum m[][2], float d[][2])
{
  if constexpr (compute_unit == ComputeUnit::kCudaCore)
  { 
    // accumulate_d performs partial accumulation with cuda core
    // aggregate d
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 2; k++)
      {
        d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x1); // sum 0 and 1, 2 and 3
        d[fq][k] += __shfl_xor_sync(0xffffffff, d[fq][k], 0x2); // sum 0 and 2, 1 and 3
      }
    }
  }

  // divide O by d
  float d_rcp[num_tiles_q][2];
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      // TODO: check m to prevent nan
      d_rcp[fq][k] = math::ptx_rcp(d[fq][k]);
    }
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value)
        {
          RO[fq][fv][k] *= d_rcp[fq][(k % 4) / 2];
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          RO[fq][fv][k] = __float2half_rn(__half2float(RO[fq][fv][k]) * d_rcp[fq][(k % 4) / 2]);
        }
      }
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k, 
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          SwizzleMode swizzle_mode, uint32_t stride, typename DTypeSVAccum>
__device__ __forceinline__ void compute_fp8_sv(const smem_t<swizzle_mode, stride> &smem_V, uint32_t RS_f8[][num_tiles_k / 2][4], DTypeSVAccum RO[][num_tiles_v][8], float d[][2])
{
  uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
  // uint32_t smem_V_col_base = get_warp_idx_k<num_warps_q, num_warps_k>() * ((16 * num_tiles_k) / 16) + (get_lane_id() / 8) % 2;
  uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
  {
    uint32_t offset_V = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      // load RV
      uint32_t RV[4];
      // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
      smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value)
        {
          mma::mma_sync_m16n16k32_row_col_f8f8f32(RO[fq][fv], RS_f8[fq][fk], RV);
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          // ! Not Implemented
        }
      }
      offset_V = smem_V.advance_offset_by_row<16>(offset_V);
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k, 
          uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          SwizzleMode swizzle_mode, uint32_t stride, typename DTypeSVAccum>
__device__ __forceinline__ void compute_fp8_sv_inst_buf(const smem_t<swizzle_mode, stride> &smem_V, uint32_t RS_f8[][num_tiles_k / 2][4], DTypeSVAccum RO[][num_tiles_v][8], float d[][2])
{
  uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
  // uint32_t smem_V_col_base = get_warp_idx_k<num_warps_q, num_warps_k>() * ((16 * num_tiles_k) / 16) + (get_lane_id() / 8) % 2;
  uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;

  float RO_inst_buf[num_tiles_q][num_tiles_v][8];

#pragma unroll
  for (uint32_t fk = 0; fk < 1; fk++)
  {
    uint32_t offset_V = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      // load RV
      uint32_t RV[4];
      // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
      smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value)
        {
          mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInit>(RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          // ! Not Implemented
        }
      }
      offset_V = smem_V.advance_offset_by_row<16>(offset_V);
    }
  }

#pragma unroll
  for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++)
  {
    uint32_t offset_V = smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      // load RV
      uint32_t RV[4];
      // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
      smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        if constexpr (std::is_same<DTypeSVAccum, float>::value)
        {
          mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInplaceUpdate>(RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
        }
        else if constexpr (std::is_same<DTypeSVAccum, half>::value)
        {
          // ! Not Implemented
        }
      }
      offset_V = smem_V.advance_offset_by_row<16>(offset_V);
    }
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      RO[fq][fv][0] += RO_inst_buf[fq][fv][0];
      RO[fq][fv][1] += RO_inst_buf[fq][fv][1];
      RO[fq][fv][2] += RO_inst_buf[fq][fv][2];
      RO[fq][fv][3] += RO_inst_buf[fq][fv][3];
      RO[fq][fv][4] += RO_inst_buf[fq][fv][4];
      RO[fq][fv][5] += RO_inst_buf[fq][fv][5];
      RO[fq][fv][6] += RO_inst_buf[fq][fv][6];
      RO[fq][fv][7] += RO_inst_buf[fq][fv][7];
    }
  }
}

// paddle converter zone
namespace pd_cvt {

// phi::dtype::xx16 -> half or nv_bfloat16
template <typename T>
struct PD16bitTrait {
  using DataType = T;
};

template <>
struct PD16bitTrait<phi::dtype::float16> {
  // Since LayerNormDirectCUDAFunctor register half type, we need to convert
  // phi::float16 to half.
  using DataType = half;
};

#ifdef PADDLE_CUDA_BF16
template <>
class PD16bitTrait<phi::dtype::bfloat16> {
public:
  using DataType = __nv_bfloat16;
};
#endif

// half or nv_bfloat16 -> phi::dtype::xx16
template <typename T>
struct PD16bitReTrait {
  using DataType = T;
};

template <>
struct PD16bitReTrait<half> {
  using DataType = phi::dtype::float16;
};

#ifdef PADDLE_CUDA_BF16
template<>
class PD16bitReTrait<__nv_bfloat16> {
public:
  using DataType = phi::dtype::bfloat16;
};
#endif

}; // paddle converter zone end

// namespace wgmma
namespace wgmma{
__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

template <int stride, typename T>
__device__ uint64_t make_smem_desc(T* ptr) {
    static_assert(stride == 32 || stride == 64 || stride == 128);
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)(8 * stride)) << 32;
    desc |= ((stride == 128) ? 1llu : (stride == 64) ? 2llu : 3llu) << 62;
    return desc;
}

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k16_f16f16f32(float d[][8], T* sA, T* sB) {
    uint64_t desc_a = make_smem_desc<BK*2>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        " %64,"
        " %65,"
        " %66,  %67,  %68,  %69,  %70;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k16_f16f16f32(float d[][8], T* sA, T* sB) {
    uint64_t desc_a = make_smem_desc<BK*2>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31}, "
        " %32,"
        " %33,"
        " %34,  %35,  %36,  %37,  %38;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k16_f16f16f32(float d[][8], uint32_t RA[], T* sB) {
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        "{%64,  %65,  %66,  %67}, "
        " %68,"
        " %69,  %70,  %71, %72;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB)));
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k16_f16f16f32(float d[][8], uint32_t RA[], T* sB) {
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31}, "
        "{%32,  %33,  %34,  %35}, "
        " %36,"
        " %37,  %38,  %39, %40;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB)));
}

template<int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k32_f8f8f32(float d[][8], uint32_t RA[], T* sB) {
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31}, "
        "{%32,  %33,  %34,  %35}, "
        " %36,"
        " %37,"
        " %38,  %39;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)),
            "n"(1), "n"(1));
}

template<int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k32_f8f8f32(float d[][8], uint32_t RA[], T* sB) {
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        "{%64,  %65,  %66,  %67}, "
        " %68,"
        " %69,"
        " %70,  %71;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)),
            "n"(1), "n"(1));
}

template<int ScaleD, int BK, typename T>
__device__ void wgmma_m64n128k32_s8s8s32(int32_t d[][8], T* sA, T* sB) {
    uint64_t desc_a = make_smem_desc<BK>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63}, "
        " %64,"
        " %65,"
        " %66;\n"
        "}\n"
        : "+r"(d[0][0]), "+r"(d[0][1]), "+r"(d[0][2]), "+r"(d[0][3]), "+r"(d[0][4]), "+r"(d[0][5]), "+r"(d[0][6]), "+r"(d[0][7]),
          "+r"(d[1][0]), "+r"(d[1][1]), "+r"(d[1][2]), "+r"(d[1][3]), "+r"(d[1][4]), "+r"(d[1][5]), "+r"(d[1][6]), "+r"(d[1][7]),
          "+r"(d[2][0]), "+r"(d[2][1]), "+r"(d[2][2]), "+r"(d[2][3]), "+r"(d[2][4]), "+r"(d[2][5]), "+r"(d[2][6]), "+r"(d[2][7]),
          "+r"(d[3][0]), "+r"(d[3][1]), "+r"(d[3][2]), "+r"(d[3][3]), "+r"(d[3][4]), "+r"(d[3][5]), "+r"(d[3][6]), "+r"(d[3][7]),
          "+r"(d[4][0]), "+r"(d[4][1]), "+r"(d[4][2]), "+r"(d[4][3]), "+r"(d[4][4]), "+r"(d[4][5]), "+r"(d[4][6]), "+r"(d[4][7]),
          "+r"(d[5][0]), "+r"(d[5][1]), "+r"(d[5][2]), "+r"(d[5][3]), "+r"(d[5][4]), "+r"(d[5][5]), "+r"(d[5][6]), "+r"(d[5][7]),
          "+r"(d[6][0]), "+r"(d[6][1]), "+r"(d[6][2]), "+r"(d[6][3]), "+r"(d[6][4]), "+r"(d[6][5]), "+r"(d[6][6]), "+r"(d[6][7]),
          "+r"(d[7][0]), "+r"(d[7][1]), "+r"(d[7][2]), "+r"(d[7][3]), "+r"(d[7][4]), "+r"(d[7][5]), "+r"(d[7][6]), "+r"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)));
}

template<int ScaleD, int BK, typename T>
__device__ void wgmma_m64n64k32_s8s8s32(int32_t d[][8], T* sA, T* sB) {
    uint64_t desc_a = make_smem_desc<BK>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34;\n"
        "}\n"
        : "+r"(d[0][0]), "+r"(d[0][1]), "+r"(d[0][2]), "+r"(d[0][3]), "+r"(d[0][4]), "+r"(d[0][5]), "+r"(d[0][6]), "+r"(d[0][7]),
          "+r"(d[1][0]), "+r"(d[1][1]), "+r"(d[1][2]), "+r"(d[1][3]), "+r"(d[1][4]), "+r"(d[1][5]), "+r"(d[1][6]), "+r"(d[1][7]),
          "+r"(d[2][0]), "+r"(d[2][1]), "+r"(d[2][2]), "+r"(d[2][3]), "+r"(d[2][4]), "+r"(d[2][5]), "+r"(d[2][6]), "+r"(d[2][7]),
          "+r"(d[3][0]), "+r"(d[3][1]), "+r"(d[3][2]), "+r"(d[3][3]), "+r"(d[3][4]), "+r"(d[3][5]), "+r"(d[3][6]), "+r"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)));
}

template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename DTypeIn, typename T>
__device__ __forceinline__ void wgmma_f16f16f32(float d[WGMMA_N/16][8], T* sA, T* sB) {
    static_assert(std::is_same<DTypeIn, half>::value);

    static_assert(WGMMA_N == 128 || WGMMA_N == 64);
    if constexpr (WGMMA_N == 128) {
        wgmma_m64n128k16_f16f16f32<ScaleD, ScaleA, ScaleB, TransA, TransB, BK>(d, sA, sB);
    }
    else if constexpr (WGMMA_N == 64) {
        wgmma_m64n64k16_f16f16f32<ScaleD, ScaleA, ScaleB, TransA, TransB, BK>(d, sA, sB);
    }
}

template<int WGMMA_N, int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_s8s8s32(int32_t d[WGMMA_N/16][8], T* sA, T* sB) {
    static_assert(WGMMA_N == 128 || WGMMA_N == 64);
    if constexpr (WGMMA_N == 128) {
        wgmma_m64n128k32_s8s8s32<ScaleD, BK>(d, sA, sB);
    }
    else if constexpr (WGMMA_N == 64) {
        wgmma_m64n64k32_s8s8s32<ScaleD, BK>(d, sA, sB);
    }
}

template<int WGMMA_N, int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_f8f8f32(float d[][8], uint32_t* RA, T* sB) {
    static_assert(WGMMA_N == 128 || WGMMA_N == 64);
    if constexpr (WGMMA_N == 128) {
        wgmma_m64n128k32_f8f8f32<ScaleD, BK>(d, RA, sB);
    }
    else if constexpr (WGMMA_N == 64) {
        wgmma_m64n64k32_f8f8f32<ScaleD, BK>(d, RA, sB);
    }
}

} // namespace wgmma