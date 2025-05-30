// /*
//  * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
//  * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

template<typename TYPE>
struct WintXPreProcessFunctor {
    struct Arguments {
      const uint16_t* in_ptr;
      const TYPE* supper_scale_ptr;
      const int* b_shift_bits;
      TYPE * out_ptr;
      const int n;
      const int k;
      const int in_stride;
      const int group_num;
      const int pack_num;
      const int weight_mask;
      const int local_scale_mask;
      const int bbzip;
    };

    // Note: need to support vectorized operation
    __device__ void operator()(const Arguments& args, const int tid, const int total_threads) {
      for(int col = tid; col < args.n; col += total_threads) {
        for(int row = 0; row < args.k; row++) {
          int local_scale_row = (row / args.group_num + 1) * args.group_num - 1;
          uint16_t local_scale = args.in_ptr[local_scale_row / args.pack_num * args.in_stride + col] &
                                 args.local_scale_mask;
          int16_t unzip_value = args.in_ptr[row / args.pack_num * args.in_stride + col] >>
                                args.b_shift_bits[row % args.pack_num] &
                                args.weight_mask - args.bbzip;
          args.out_ptr[row * args.n + col] = static_cast<TYPE>(unzip_value * local_scale) * args.supper_scale_ptr[col];
        }
      }
      __syncthreads();
    }
};


template<typename T>
__global__ void wintx_unzip_quantization_kernel(
    const uint16_t* in_ptr,
    const T* supper_scale_ptr,
    const int* b_shift_bits,
    T* out_ptr,
    const int64_t n,
    const int64_t k,
    const int num_n_per_block,
    const int num_blocks_per_expert,
    const int group_num,
    const int pack_num,
    const int weight_mask,
    const int local_scale_mask,
    const int bbzip,
    const int total_threads) {
    extern __shared__ uint16_t shared_b[];
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int idx_expert = block_id / num_blocks_per_expert;
    const int current_block_id = block_id % num_blocks_per_expert;

    const int in_offset_n = idx_expert * n * ((k / 64) * 10) +  current_block_id * num_n_per_block;
    const int out_offset_n = idx_expert * n * k + current_block_id * num_n_per_block;
    const int scale_offset_n = idx_expert * n;

    for(int i = 0; i < k; i += group_num) {
        const uint16_t* in_offset = in_ptr + in_offset_n + (i / 64 * 10) * n;
        const T* scale_offset = supper_scale_ptr + scale_offset_n;

        // wintx process deal with shareMemory
        typename WintXPreProcessFunctor<T>::Arguments args{
            in_offset, scale_offset, b_shift_bits, reinterpret_cast<T*>(shared_b),
            group_num, num_n_per_block, n, 
            pack_num, weight_mask, 
            local_scale_mask, bbzip};
        WintXPreProcessFunctor<T> winx_process;
        winx_process(args, tid, total_threads);

        T* out_offset = out_ptr + out_offset_n + i * n;
        // after winx process
        for(int row = 0; row < group_num; row++) {
            for(int col = 0; col < num_n_per_block; col++) {
                out_offset[row * n + col] = 
                shared_b[row * num_n_per_block + col];
            }
        }
    }
    return;
}

template<typename T>
void wintx_unzip_quantization_kernel_Launcher(
    const uint16_t* weight,
    const T* supper_scale,
    const int* b_shift_bits,
    T* out_weight,
    const int64_t n,
    const int64_t k,
    const int num_experts,
    const int group_num,
    const int pack_num,
    const int weight_mask,
    const int local_scale_mask,
    const int bbzip) {
    const int num_threads = 128;
    const int num_blocks_per_expert = (n +  num_threads - 1) / num_threads;
    const int num_blocks = num_blocks_per_expert * num_experts;
    size_t sharedMemSize = group_num * num_threads * sizeof(T);
    wintx_unzip_quantization_kernel<T><<<num_blocks, num_threads, sharedMemSize>>>(
        weight,
        supper_scale,
        b_shift_bits,
        out_weight,
        n,
        k,
        num_threads,
        num_blocks_per_expert,
        group_num,
        pack_num,
        weight_mask,
        local_scale_mask,
        bbzip,
        num_threads);
    return;
}

template void wintx_unzip_quantization_kernel_Launcher<half>(const uint16_t*,
                                                            const half*,
                                                            const int*,
                                                            half*,
                                                            const int64_t,
                                                            const int64_t,
                                                            const int,
                                                            const int, 
                                                            const int,
                                                            const int,
                                                            const int,
                                                            const int);


#ifdef PADDLE_CUDA_BF16
template void wintx_unzip_quantization_kernel_Launcher<__nv_bfloat16>(const uint16_t*,
                                                                        const __nv_bfloat16*,
                                                                        const int*,
                                                                        __nv_bfloat16*,
                                                                        const int64_t,
                                                                        const int64_t,
                                                                        const int,
                                                                        const int, 
                                                                        const int,
                                                                        const int,
                                                                        const int,
                                                                        const int);
#endif