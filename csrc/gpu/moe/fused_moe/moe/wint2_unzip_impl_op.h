
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

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cstdint>


struct WeightOnlyTraits {
  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kPackNum = 4;
  static constexpr int16_t kWeightMask = 0x3F;
  static constexpr int32_t kBBZip = 32;
};

template <typename T, int64_t TileRows, int64_t TileColumns>
struct Wint2UnzipFunctor {
  using ScaleComputeT = float;

  static constexpr int64_t kTileRows = TileRows;
  static constexpr int64_t kTileColumns = TileColumns;

  struct Arguments {
    const uint8_t* w_ptr;
    const T* w_scale_ptr;
    const float* w_code_scale_ptr;
    const float* w_code_zp_ptr;
    const T* w_super_scale_ptr;
    T* out_ptr;
    const int in_stride;
  };

  __device__ void operator()(const Arguments& args, const int tid, const int num_threads) {
    int16_t shift_bits[4] = {9, 6, 3, 0};

    for (int col = tid; col < kTileColumns; col += num_threads) {
      for (int row = 0; row < kTileRows; ++row) {
        int w_row = row / WeightOnlyTraits::kPackNum;
        int w_offset = w_row * args.in_stride + col;
        ScaleComputeT w = static_cast<ScaleComputeT>(args.w_ptr[w_offset]);
        ScaleComputeT w_code_scale = static_cast<ScaleComputeT>(args.w_code_scale_ptr[col]);
        ScaleComputeT w_code_zp = static_cast<ScaleComputeT>(args.w_code_zp_ptr[col]);
        
        int16_t w_zipped_value = static_cast<int16_t>(floor(w * w_code_scale + w_code_zp + 0.5));
        int16_t shift_bit = shift_bits[row % WeightOnlyTraits::kPackNum];
        int16_t w_shifted_value = (w_zipped_value >> shift_bit) & WeightOnlyTraits::kWeightMask;

        int w_scale_row = row / WeightOnlyTraits::kGroupSize;
        int w_scale_offset = w_scale_row * args.in_stride + col;
        T w_scale = static_cast<T>(args.w_scale_ptr[w_scale_offset]);
        
        if (args.w_super_scale_ptr) {
          T w_super_scale = static_cast<T>(args.w_super_scale_ptr[col]);
          w_scale = w_scale * w_super_scale;
        }

        args.out_ptr[row * kTileColumns + col] = static_cast<T>(w_scale) * (static_cast<T>(w_shifted_value) - static_cast<T>(WeightOnlyTraits::kBBZip));
      }
    }
    __syncthreads();
  }
};

template <typename T, int64_t TileRows, int64_t TileColumns>
__global__ void Wint2UnzipKernel(
    const uint8_t* w_ptr,
    const T* w_scale_ptr,
    const float* w_code_scale_ptr,
    const float* w_code_zp_ptr,
    const T* w_super_scale_ptr,
    T* output_tensor_ptr,
    const int64_t batch,
    const int64_t num_rows,
    const int64_t num_columns) {
  __shared__ T smem[TileRows * TileColumns];

  int64_t block_start_column = blockIdx.x * TileColumns;

  int64_t block_start_row = blockIdx.z * num_rows + blockIdx.y * TileRows;

  int64_t block_start_w_row = block_start_row / WeightOnlyTraits::kPackNum;
  int64_t block_w_offset = block_start_w_row * num_columns + block_start_column;
  const uint8_t* block_w_ptr = w_ptr + block_w_offset;

  int64_t block_start_w_scale_row = block_start_row / WeightOnlyTraits::kGroupSize;
  int64_t block_w_scale_offset = block_start_w_scale_row * num_columns + block_start_column;
  const T* block_w_scale_ptr = w_scale_ptr + block_w_scale_offset;

  const float* block_w_code_scale_ptr = w_code_scale_ptr + blockIdx.z * num_columns + block_start_column;
  const float* block_w_code_zp_ptr = w_code_zp_ptr + blockIdx.z * num_columns + block_start_column;
  const T* block_w_super_scale_ptr = w_super_scale_ptr ? w_super_scale_ptr + blockIdx.z * num_columns + block_start_column : nullptr;

  // unzip to shared memory
  typename Wint2UnzipFunctor<T, TileRows, TileColumns>::Arguments args{
      block_w_ptr, block_w_scale_ptr, block_w_code_scale_ptr, block_w_code_zp_ptr, block_w_super_scale_ptr, smem, num_columns};

  Wint2UnzipFunctor<T, TileRows, TileColumns> winx_unzipper;
  winx_unzipper(args, threadIdx.x, blockDim.x);

  // write back to global memory
  for (int row = 0; row < TileRows; ++row) {
    for (int col = 0; col < TileColumns; ++col) {
      int64_t global_row = block_start_row + row;
      int64_t global_col = block_start_column + col;
      output_tensor_ptr[global_row * num_columns + global_col] = smem[row * TileColumns + col];
    }
  }
}

template <typename T>
void Wint2UnzipKernelLauncher(
    const uint8_t* w_ptr,
    const T* w_scale_ptr,
    const float* w_code_scale_ptr,
    const float* w_code_zp_ptr,
    const T* w_super_scale_ptr,
    T* output_tensor_ptr,
    const int64_t batch,
    const int64_t num_rows,
    const int64_t num_columns) {
  constexpr int kTileRows = 64;
  constexpr int kTileColumns = 128;

  const int num_threads = 128;
  const int block_dim_x = (num_columns + kTileColumns - 1) / kTileColumns;
  const int block_dim_y = (num_rows + kTileRows - 1) / kTileRows;

  dim3 block_dim(num_threads, 1, 1); 
  dim3 grid_dim(block_dim_x, block_dim_y, batch);
  // printf("Launch config: grid_dim={%d, %d, %d}, block_dim={%d, 1, 1}\n", block_dim_x, block_dim_y, batch, num_threads);

  Wint2UnzipKernel<T, kTileRows, kTileColumns><<<grid_dim, block_dim>>>(
      w_ptr, w_scale_ptr, w_code_scale_ptr, w_code_zp_ptr, w_super_scale_ptr, output_tensor_ptr, batch, num_rows, num_columns);
}

