// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "fp8_fp8_half_cuda_core_gemm.h"
#include "cutlass/numeric_conversion.h"

template <typename InputType,
          typename OutputType,
          int32_t TILE_M,
          int32_t TILE_N,
          int32_t BLOCK_SIZE,
          bool UseBias>
__global__ void cudaCoreGemm(InputType const* __restrict__ act,
                             InputType const* __restrict__ weight,
                             OutputType const* __restrict__ bias,
                             OutputType* __restrict__ output,
                             int32_t m,
                             int32_t n,
                             int32_t k,
                             float alpha) {
  using VecType = int4;
  static constexpr int32_t kStepK =
      static_cast<int32_t>(128 / (8 * sizeof(InputType)));
  static constexpr int32_t kTileK = kStepK * BLOCK_SIZE;
  auto tileIdM = static_cast<int32_t>(blockIdx.x * TILE_M);
  auto tileIdN = static_cast<int32_t>(blockIdx.y * TILE_N);
  auto tid = static_cast<int32_t>(threadIdx.x);
  float tile_a[kStepK], tile_w[TILE_N * kStepK];
  float acc[TILE_M * TILE_N];

  static_assert(kStepK % 4 == 0);
  using Converter = cutlass::NumericArrayConverter<float, InputType, 4>;
  using CvtSrcType = typename Converter::source_type;
  using CvtResType = typename Converter::result_type;

  static constexpr int32_t kCvtCount =
      static_cast<int32_t>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
  for (int32_t i = 0; i < TILE_M * TILE_N; ++i) {
    acc[i] = 0;
  }
  act += tileIdM * k;
  weight += tileIdN * k;
  output += tileIdM * n + tileIdN;
  if constexpr (UseBias) {
    bias += tileIdN;
  }
  for (int32_t idxK = tid * kStepK; idxK < k; idxK += kTileK) {
    for (int32_t i = 0; i < TILE_N; ++i) {
      auto tile_w_quantized =
          reinterpret_cast<VecType const*>(weight + i * k + idxK)[0];
#pragma unroll
      for (int32_t cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx) {
        reinterpret_cast<CvtResType*>(tile_w)[i * kCvtCount + cvtIdx] =
            Converter::convert(
                reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvtIdx]);
      }
    }
#pragma unroll
    for (int32_t i = 0; i < TILE_M; ++i) {
      auto tile_a_quantized =
          reinterpret_cast<VecType const*>(act + i * k + idxK)[0];
#pragma unroll
      for (int32_t cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx) {
        reinterpret_cast<CvtResType*>(tile_a)[cvtIdx] = Converter::convert(
            reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvtIdx]);
      }
#pragma unroll
      for (int32_t j = 0; j < TILE_N; ++j) {
#pragma unroll
        for (int32_t l = 0; l < kStepK; ++l) {
          acc[i * TILE_N + j] =
              fma(tile_a[l], tile_w[j * kStepK + l], acc[i * TILE_N + j]);
        }
      }
    }
  }

  typedef cub::WarpReduce<float> WarpReduce;

  static constexpr int32_t kWarpSize = 32;
  static constexpr int32_t kWarpNum = BLOCK_SIZE / kWarpSize;
  int32_t warpId = tid / kWarpSize, laneId = tid % kWarpSize;
  __shared__ float shmem[TILE_M * TILE_N * kWarpNum];
  __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];
#pragma unroll
  for (int32_t mi = 0; mi < TILE_M; ++mi) {
#pragma unroll
    for (int32_t ni = 0; ni < TILE_N; ++ni) {
      float val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * TILE_N + ni]);
      if (laneId == 0) {
        shmem[mi * TILE_N + ni + warpId * TILE_M * TILE_N] = val;
      }
    }
  }
  
  __syncthreads();
  for (int32_t ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE) {
    int32_t mid = ii / TILE_N, nid = ii % TILE_N;
    float val = 0;
#pragma unroll
    for (int32_t jj = 0; jj < kWarpNum; ++jj) {
      val += shmem[jj * TILE_M * TILE_N + ii];
    }

    if constexpr (UseBias) {
        output[mid * n + nid] = static_cast<OutputType>(val * alpha + (float)*(bias+nid)) ;
    } else {
        output[mid * n + nid] = static_cast<OutputType>(val * alpha);
    }
  }
}

template <typename InputType,
          typename OutputType,
          int32_t TILE_M,
          int32_t TILE_N,
          int32_t BLOCK_SIZE>
void cudaCoreGemmKernel(GemmParams const& params) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(params.m / TILE_M, params.n / TILE_N);
    // std::cout << "m" << params.m << " n" << params.n <<  " k " << params.k << std::endl;

    if (params.bias != nullptr) {
        cudaCoreGemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE, true>
        <<<grid, block, 0, params.stream>>>(
            reinterpret_cast<InputType const*>(params.act),
            reinterpret_cast<InputType const*>(params.weight),
            reinterpret_cast<OutputType const*>(params.bias),
            reinterpret_cast<OutputType*>(params.output),
            params.m,
            params.n,
            params.k,
            params.alpha);
    } else {
        cudaCoreGemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE, false>
        <<<grid, block, 0, params.stream>>>(
            reinterpret_cast<InputType const*>(params.act),
            reinterpret_cast<InputType const*>(params.weight),
            reinterpret_cast<OutputType const*>(params.bias),
            reinterpret_cast<OutputType*>(params.output),
            params.m,
            params.n,
            params.k,
            params.alpha);
    }
}

template <typename InputType,
          typename OutputType,
          int TILE_M,
          int TILE_N,
          int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(GemmParams const& params) {
  constexpr int cudaCoreGemmTemplateMaxM = 16;
  if (params.m == TILE_M) {
    cudaCoreGemmKernel<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>(
        params);
    return true;
  }
  if constexpr (TILE_M < cudaCoreGemmTemplateMaxM) {
    return cudaCoreGemmTemplateCaller<InputType,
                                      OutputType,
                                      TILE_M + 1,
                                      TILE_N,
                                      BLOCK_SIZE>(params);
  }
  return false;
}

template <typename InputType, typename OutputType>
bool cuda_core_gemm_launcher(GemmParams const& params) {
  return cudaCoreGemmTemplateCaller<InputType, OutputType, 1, 2, 256>(params);
}

template bool cuda_core_gemm_launcher<__nv_fp8_e4m3, __nv_bfloat16>(GemmParams const&);
template bool cuda_core_gemm_launcher<__nv_fp8_e4m3, half>(GemmParams const&);
template bool cuda_core_gemm_launcher<__nv_fp8_e5m2, __nv_bfloat16>(GemmParams const&);
template bool cuda_core_gemm_launcher<__nv_fp8_e5m2, half>(GemmParams const&);