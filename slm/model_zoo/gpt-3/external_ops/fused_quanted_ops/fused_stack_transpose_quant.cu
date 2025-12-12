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

#include "quant_utils.h"

template <typename T, int VecSize>
struct __align__(sizeof(T) * VecSize) VecType {
  T val[VecSize];
  __host__ __device__ inline T& operator[](size_t i) { return val[i]; }
  __host__ __device__ inline const T& operator[](size_t i) const {
    return val[i];
  }
};

struct FastDiv {
  FastDiv() {}
  FastDiv(uint64_t d) {
    for (shift_val = 0; shift_val < 64; ++shift_val) {
      uint64_t shift_limit = uint64_t(1) << shift_val;
      if (shift_limit >= d) break;
    }

    // quotient = ((uint128_t)n_hi << 64) / d
    uint64_t quotient = 0;
    uint64_t n_hi = (uint64_t(1) << shift_val) - d, n_lo = 0;
    for (int i = 63; i >= 0; --i) {
      uint64_t d_hi = i == 0 ? 0 : d >> (64 - i);
      uint64_t d_lo = d << i;
      if (n_hi == 0 && n_lo == 0) break;
      if ((d_hi < n_hi) || (d_hi <= n_hi && d_lo <= n_lo)) {
        quotient |= uint64_t(1) << i;
        n_hi -= d_hi + (d_lo > n_lo);
        n_lo -= d_lo;
      }
    }
    multiplier = quotient + 1;
  }

  __device__ uint64_t Div(uint64_t n) const {
    uint64_t t = __umul64hi(n, multiplier);
    return (t + n) >> shift_val;
  }

  int shift_val;
  uint64_t multiplier;
};

__device__ void BlockLoad(const int64_t* __restrict__ X_ptrs,
                          __nv_bfloat16 input[4][4],
                          size_t K,
                          size_t block_y,
                          size_t block_x) {
  const __nv_bfloat16* X =
      reinterpret_cast<const __nv_bfloat16*>(X_ptrs[blockIdx.z]);

  for (size_t i = 0; i < 4; i++) {
    size_t idx_m = block_y * 128 + threadIdx.y + i * 32;
    size_t idx_k = block_x * 128 + threadIdx.x * 4;
    size_t idx = idx_m * K + idx_k;

    using LoadT = VecType<__nv_bfloat16, 4>;
    LoadT data = *reinterpret_cast<const LoadT*>(X + idx);
    for (int j = 0; j < 4; j++) {
      input[i][j] = data[j];
    }
  }
}

__device__ __nv_bfloat16 WarpReduceMax(__nv_bfloat16 x) {
  for (int offset = 16; offset > 0; offset /= 2) {
    __nv_bfloat16 t = __shfl_down_sync(0xffffffff, x, offset);
    x = __hmax(x, t);
  }
  return x;
}

__device__ __nv_bfloat16 BlockReduceMax(__nv_bfloat16 input[4][4]) {
  // [(4), 32, 32, (4)] => [32, 32]
  __nv_bfloat16 local_max;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      __nv_bfloat16 t = __habs(input[i][j]);
      local_max = (i == 0 && j == 0) ? t : __hmax(local_max, t);
    }
  }

  // [32, (32)] => [32]
  __nv_bfloat16 warp_max = WarpReduceMax(local_max);

  // [(32)] => [1]
  __shared__ __nv_bfloat16 block_max[32];
  if (threadIdx.x == 0) {
    block_max[threadIdx.y] = warp_max;
  }
  __syncthreads();
  if (threadIdx.y == 0) {
    warp_max = WarpReduceMax(block_max[threadIdx.x]);
    if (threadIdx.x == 0) {
      block_max[0] = warp_max;
    }
  }
  __syncthreads();

  return block_max[0];
}

template <typename OutT>
__global__ void __launch_bounds__(1024)
    FusedStackQuantKernel(const int64_t* __restrict__ X_ptrs,
                          OutT* __restrict__ out,
                          float* __restrict__ scale,
                          size_t M,
                          size_t K,
                          FastDiv K_div_128) {
  size_t block_y = K_div_128.Div(blockIdx.x);
  size_t block_x = blockIdx.x - block_y * (K / 128);

  // Load 128x128 elements from X
  __nv_bfloat16 input[4][4];
  BlockLoad(X_ptrs, input, K, block_y, block_x);

  // Find the maximum in all elements
  __nv_bfloat16 amax = BlockReduceMax(input);

  // Compute scale and store back
  float scale_inv = ComputeScale<__nv_bfloat16, OutT>(amax, 0.0f);
  float scale_out = __frcp_rn(scale_inv);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t idx_n = blockIdx.z;
    size_t idx_m = block_y;
    size_t idx_k = block_x;
    size_t idx = (idx_n * (M / 128) + idx_m) * (K / 128) + idx_k;
    scale[idx] = scale_out;
  }

  // Scale X and store to out
  for (size_t i = 0; i < 4; i++) {
    size_t idx_n = blockIdx.z;
    size_t idx_m = block_y * 128 + threadIdx.y + i * 32;
    size_t idx_k = block_x * 128 + threadIdx.x * 4;
    size_t idx = (idx_n * M + idx_m) * K + idx_k;

    using StoreT = VecType<OutT, 4>;
    StoreT data;
    for (int j = 0; j < 4; j++) {
      float input_fp32 = static_cast<float>(input[i][j]);
      float output_scaled = input_fp32 * scale_inv;
      data[j] = static_cast<OutT>(output_scaled);
    }
    *reinterpret_cast<StoreT*>(out + idx) = data;
  }
}

template <typename OutT>
__global__ void __launch_bounds__(1024)
    FusedStackTransposeQuantKernel(const int64_t* __restrict__ X_ptrs,
                                   OutT* __restrict__ out,
                                   float* __restrict__ scale,
                                   size_t M,
                                   size_t K,
                                   FastDiv K_div_128) {
  size_t block_y = K_div_128.Div(blockIdx.x);
  size_t block_x = blockIdx.x - block_y * (K / 128);

  // Load 128x128 elements from X
  __nv_bfloat16 input[4][4];
  BlockLoad(X_ptrs, input, K, block_y, block_x);

  // Find the maximum in all elements
  __nv_bfloat16 amax = BlockReduceMax(input);

  // Compute scale and store back
  float scale_inv = ComputeScale<__nv_bfloat16, OutT>(amax, 0.0f);
  float scale_out = __frcp_rn(scale_inv);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    size_t idx_n = blockIdx.z;
    size_t idx_k = block_x;
    size_t idx_m = block_y;
    size_t idx = (idx_n * (K / 128) + idx_k) * (M / 128) + idx_m;
    scale[idx] = scale_out;
  }

  // Scale X and transpose in shared memory
  __shared__ OutT shm[128][129];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      float input_fp32 = static_cast<float>(input[i][j]);
      float output_scaled = input_fp32 * scale_inv;
      shm[threadIdx.x * 4 + j][i * 32 + threadIdx.y] =
          static_cast<OutT>(output_scaled);
    }
  }
  __syncthreads();

  // Store X back to out
  for (size_t i = 0; i < 4; i++) {
    size_t idx_n = blockIdx.z;
    size_t idx_k = block_x * 128 + threadIdx.y + i * 32;
    size_t idx_m = block_y * 128 + threadIdx.x * 4;
    size_t idx = (idx_n * K + idx_k) * M + idx_m;

    using StoreT = VecType<OutT, 4>;
    StoreT data;
    for (int j = 0; j < 4; j++) {
      data[j] = shm[i * 32 + threadIdx.y][threadIdx.x * 4 + j];
    }
    *reinterpret_cast<StoreT*>(out + idx) = data;
  }
}

/**
 * Stack tensors in X, optionally transpose dim[-1] and dim[-2], and do
 * quantization on both dim[-1] and dim[-2].
 *
 * Inputs:
 *   X    : N tensors of [M, K], bfloat16
 *
 * Outputs:
 *   if Transpose:
 *     out  : [N * K, M], float8_e4m3fn
 *     scale: [N * K / 128, M / 128], float
 *   else:
 *     out  : [N * M, K], float8_e4m3fn
 *     scale: [N * M / 128, K / 128], float
 *
 * Requirements:
 *   1) N <= 65535
 *   2) M % 128 == 0
 *   3) K % 128 == 0
 */
template <bool Transpose>
std::vector<paddle::Tensor> fused_stack_transpose_quant(
    const std::vector<paddle::Tensor>& X) {
  int64_t N = X.size();
  PD_CHECK(N > 0);
  for (int64_t i = 0; i < N; i++) {
    PD_CHECK(X[i].dtype() == paddle::DataType::BFLOAT16);
  }

  std::vector<int64_t> shape = X[0].shape();
  PD_CHECK(shape.size() == 2);
  int64_t M = shape[0];
  int64_t K = shape[1];

  for (int64_t i = 1; i < N; i++) {
    std::vector<int64_t> shape = X[i].shape();
    PD_CHECK(shape.size() == 2);
    PD_CHECK(shape[0] == M);
    PD_CHECK(shape[1] == K);
  }

  PADDLE_ENFORCE_LE(N,
                    65535,
                    common::errors::InvalidArgument(
                        "The batch size (N) must be no larger than 65535."));
  PADDLE_ENFORCE_EQ(M % 128,
                    0,
                    common::errors::InvalidArgument(
                        "The upper dim (M) must be multiple of 128."));
  PADDLE_ENFORCE_EQ(K % 128,
                    0,
                    common::errors::InvalidArgument(
                        "The lower dim (K) must be multiple of 128."));

  // Allocate for out and scale
  std::vector<int64_t> out_shape, scale_shape;
  if (Transpose) {
    out_shape = {N * K, M};
    scale_shape = {N * K / 128, M / 128};
  } else {
    out_shape = {N * M, K};
    scale_shape = {N * M / 128, K / 128};
  }

  const auto& place = X[0].place();
  paddle::Tensor out =
      paddle::empty(out_shape, paddle::DataType::FLOAT8_E4M3FN, place);
  paddle::Tensor scale =
      paddle::empty(scale_shape, paddle::DataType::FLOAT32, place);

  // Skip 0-size
  if (M == 0 || K == 0) {
    return {out, scale};
  }

  // Copy the pointers in X to device
  paddle::Tensor X_ptrs_cpu = paddle::empty({N}, paddle::DataType::INT64);
  int64_t* X_ptrs_data = X_ptrs_cpu.data<int64_t>();
  for (int64_t i = 0; i < N; i++) {
    X_ptrs_data[i] = reinterpret_cast<int64_t>(X[i].data());
  }
  paddle::Tensor X_ptrs_gpu = X_ptrs_cpu.copy_to(place, /* blocking= */ false);

  // Launch kernel
  dim3 grid((M / 128) * (K / 128), 1, N);
  dim3 block(32, 32);

#define LAUNCH_KERNEL(KERNEL)                                               \
  KERNEL<<<grid, block, 0, X[0].stream()>>>(X_ptrs_gpu.data<int64_t>(),     \
                                            out.data<phi::float8_e4m3fn>(), \
                                            scale.data<float>(),            \
                                            M,                              \
                                            K,                              \
                                            FastDiv(K / 128))
  if (Transpose) {
    LAUNCH_KERNEL(FusedStackTransposeQuantKernel);
  } else {
    LAUNCH_KERNEL(FusedStackQuantKernel);
  }
#undef LAUNCH_KERNEL

  return {out, scale};
}

PD_BUILD_OP(fused_stack_quant)
    .Inputs({paddle::Vec("X")})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(fused_stack_transpose_quant<false>));

PD_BUILD_OP(fused_stack_transpose_quant)
    .Inputs({paddle::Vec("X")})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(fused_stack_transpose_quant<true>));
