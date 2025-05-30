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
      const uint16_t* zipped_weight_ptr; 
      const TYPE* super_scale_ptr; 
      const int* b_shift_bits; 
      TYPE * weight_share_ptr; 
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
          int local_scale_row = ((row / args.group_num + 1) * args.group_num - 1) / args.pack_num;
          int zipped_weight_row = row / args.pack_num;
          uint16_t local_scale = args.zipped_weight_ptr[local_scale_row * args.in_stride + col] &
                                 args.local_scale_mask;
          int16_t unzip_value = args.zipped_weight_ptr[row / args.pack_num * args.in_stride + col] >>
                                args.b_shift_bits[row % args.pack_num] &
                                args.weight_mask - args.bbzip;
          args.weight_share_ptr[row * args.n + col] = static_cast<TYPE>(unzip_value * local_scale) * args.super_scale_ptr[col];
        }
      }
      __syncthreads();
    }
};


template<typename T>
__global__ void wintx_unzip_quantization_kernel(
    const uint16_t* zipped_weight_ptr,
    const T* super_scale_ptr,
    const int* b_shift_bits,
    T* unzipped_weight_ptr,
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
    extern __shared__ uint16_t uint16_shared_b[];
    T* shared_b = reinterpret_cast<T*>(uint16_shared_b);
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int idx_expert = block_id / num_blocks_per_expert;
    const int block_id_per_expert = block_id % num_blocks_per_expert;

    const int in_offset_n = idx_expert * n * ((k / 64) * 10) +  block_id_per_expert * num_n_per_block;
    const int out_offset_n = idx_expert * n * k + block_id_per_expert * num_n_per_block;
    const int scale_offset_n = idx_expert * n + block_id_per_expert * num_n_per_block;

    for(int i = 0; i < k; i += group_num) {
        const uint16_t* in_offset = zipped_weight_ptr + in_offset_n + (i / 64 * 10) * n;
        const T* scale_offset = super_scale_ptr + scale_offset_n;

        // wintx process deal with shareMemory
        typename WintXPreProcessFunctor<T>::Arguments args{
            in_offset, 
            scale_offset, 
            b_shift_bits, 
            shared_b,
            num_n_per_block,
            group_num,
            n,
            group_num,
            pack_num,
            weight_mask,
            local_scale_mask,
            bbzip};
        WintXPreProcessFunctor<T> winx_process;
        winx_process(args, tid, total_threads);

        T* out_ptr = unzipped_weight_ptr + out_offset_n + i * n;

        for(int col = 0; col < num_n_per_block; col++) {
          for(int row = 0; row < group_num; row++) {
            out_ptr[row * n + col] = shared_b[row * num_n_per_block + col];
          }
        }
        __syncthreads();
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
    const int num_threads = n < 128 ? n : 128;
    // const int num_blocks_per_expert = (n +  num_threads - 1) / num_threads;
    const int num_blocks_per_expert = (n + num_threads - 1) / num_threads;
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