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

#include "helper.h"

// #define DEBUG_STEP

__global__ void free_and_dispatch_block(bool *stop_flags,
                                        int *seq_lens_this_time,
                                        int *seq_lens_decoder,
                                        int *block_tables,
                                        int *encoder_block_lens,
                                        bool *is_block_step,
                                        int *step_block_list, // [bsz]
                                        int *step_len,
                                        int *recover_block_list,
                                        int *recover_len,
                                        int *need_block_list,
                                        int *need_block_len,
                                        int *used_list_len,
                                        int *free_list,
                                        int *free_list_len,
                                        const int bsz,
                                        const int block_size,
                                        const int block_num_per_seq,
                                        const int max_decoder_block_num,
                                        const int speculate_step_token_num) {
    typedef cub::BlockReduce<cub::KeyValuePair<int, int>, 512> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const int tid = threadIdx.x;
    if (tid < bsz) {
        int *block_table_now = block_tables + tid * block_num_per_seq;
        if (stop_flags[tid] && !is_block_step[tid]) {
            // 回收block块
            const int encoder_block_len = encoder_block_lens[tid];
            const int decoder_used_len = used_list_len[tid];
            if (decoder_used_len > 0) {
                const int ori_free_list_len = atomicAdd(free_list_len, decoder_used_len);
#ifdef DEBUG_STEP
                printf("free block seq_id: %d, free block num: %d, encoder_block_len: %d, ori_free_list_len: %d\n", 
                        tid, decoder_used_len, encoder_block_len, ori_free_list_len);
#endif
                for (int i = 0; i < decoder_used_len; i++) {
                    free_list[ori_free_list_len + i] = block_table_now[encoder_block_len + i];
                    block_table_now[encoder_block_len + i] = -1;
                }
                encoder_block_lens[tid] = 0;
                used_list_len[tid] = 0;
            }
        } else if (seq_lens_decoder[tid] != 0 && block_table_now[(seq_lens_decoder[tid] + speculate_step_token_num) / block_size] == -1) {
            // 统计需要分配block的位置和总数
            const int ori_need_block_len = atomicAdd(need_block_len, 1);
            need_block_list[ori_need_block_len] = tid;
#ifdef DEBUG_STEP
            printf("seq_id: %d need block\n", tid);
#endif
        }
    }
    __syncthreads();
    if (tid == 0) {
        printf("need_block_len: %d, free_list_len: %d\n", need_block_len[0], free_list_len[0]);
    }

    while (need_block_len[0] > free_list_len[0]) {
        
#ifdef DEBUG_STEP
        if (tid == 0) {
            printf("need_block_len: %d, free_list_len: %d\n", need_block_len[0], free_list_len[0]);
        }
#endif
        // 调度block，根据used_list_len从大到小回收block，直到满足need_block_len
        const int used_block_num = tid < bsz && !is_block_step[tid]? used_list_len[tid] : 0;
        cub::KeyValuePair<int, int> kv_pair = {tid, used_block_num};
        kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, cub::ArgMax());
        if (tid == 0) {
            const int encoder_block_len = encoder_block_lens[kv_pair.key];
#ifdef DEBUG_STEP
            printf("max_id: %d, max_num: %d, encoder_block_len: %d\n", kv_pair.key, kv_pair.value, encoder_block_len);
#endif
            int *block_table_now = block_tables + kv_pair.key * block_num_per_seq;
            for (int i = 0; i < kv_pair.value; i++) {
                free_list[free_list_len[0] + i] = block_table_now[encoder_block_len + i];
                block_table_now[encoder_block_len + i] = -1;
            }
            step_block_list[step_len[0]] = kv_pair.key;
            step_len[0] += 1;
            free_list_len[0] += kv_pair.value;
            stop_flags[kv_pair.key] = true;
            is_block_step[kv_pair.key] = true;
            seq_lens_this_time[kv_pair.key] = 0;
            seq_lens_decoder[kv_pair.key] = 0;
        }
        __syncthreads();
    }
    // 为需要block的位置分配block，每个位置分配一个block
    if (tid < need_block_len[0]) {
        const int need_block_id = need_block_list[tid];
        if (!stop_flags[need_block_id]) {
            // 如果需要的位置正好是上一步中被释放的位置，不做处理
            used_list_len[need_block_id] += 1;
            const int ori_free_list_len = atomicSub(free_list_len, 1);
            int *block_table_now = block_tables + need_block_id * block_num_per_seq;
            block_table_now[(seq_lens_decoder[need_block_id] + speculate_step_token_num) / block_size] = free_list[ori_free_list_len - 1];
        }
        need_block_list[tid] = -1;
    }
    __syncthreads();
    // 计算可以复原的query id
    if (tid == 0) {
        int ori_free_list_len = free_list_len[0];
        int ori_step_len = step_len[0];
        if (ori_step_len > 0) {
            int ori_step_block_id = step_block_list[ori_step_len - 1];
            int tmp_used_len = used_list_len[ori_step_block_id];
            // 比之前调度时多分配一个block，防止马上恢复刚调度的query(比如回收的seq_id在need_block_list中）
            int used_len = tmp_used_len < max_decoder_block_num ? tmp_used_len + 1 : tmp_used_len;
            while (ori_step_len > 0 && ori_free_list_len >= used_len) {
#ifdef DEBUG_STEP
                printf("recover seq_id: %d, free_list_len: %d, used_list_len: %d\n", 
                        ori_step_block_id, ori_free_list_len, used_len);
#endif
                recover_block_list[recover_len[0]] = ori_step_block_id;
                is_block_step[ori_step_block_id] = false;
                used_list_len[ori_step_block_id] = used_len;
                ori_free_list_len -= used_len;
                step_block_list[ori_step_len - 1] = -1;
                step_len[0] -= 1;
                recover_len[0] += 1;
                ori_step_len = step_len[0];
                if (ori_step_len > 0) {
                    ori_step_block_id = step_block_list[ori_step_len - 1];
                    tmp_used_len = used_list_len[ori_step_block_id];
                    used_len = tmp_used_len < max_decoder_block_num ? tmp_used_len + 1 : tmp_used_len;
                }
            }
        }
        need_block_len[0] = 0;
    }
}

// 根据上一步计算出的可以复原的query_id进行状态恢复
__global__ void recover_block(int *recover_block_list, // [bsz]
                              int *recover_len,
                              bool *stop_flags,
                              int *seq_lens_this_time,
                              int *ori_seq_lens_encoder,
                              int *seq_lens_encoder,
                              int *seq_lens_decoder,
                              int *block_tables,
                              int *free_list,
                              int *free_list_len,
                              int64_t *input_ids,
                              int64_t *pre_ids,
                              int64_t *step_idx,
                              int *encoder_block_lens,
                              int *used_list_len,
                              const int64_t *next_tokens,
                              const int bsz,
                              const int block_num_per_seq,
                              const int length,
                              const int pre_id_length,
                              const int first_token_id) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int ori_free_list_len;
    if (bid < recover_len[0]) {
        const int recover_id = recover_block_list[bid];
        const int ori_seq_len_encoder = ori_seq_lens_encoder[recover_id];
        const int step_idx_now = step_idx[recover_id];
        const int seq_len = ori_seq_len_encoder + step_idx_now;
        const int encoder_block_len = encoder_block_lens[recover_id];
        const int decoder_used_len = used_list_len[recover_id];
        int *block_table_now = block_tables + recover_id * block_num_per_seq;
        int64_t *input_ids_now = input_ids + recover_id * length;
        int64_t *pre_ids_now = pre_ids + recover_id * pre_id_length;
        if (tid == 0) {
            seq_lens_this_time[recover_id] = seq_len;
            seq_lens_encoder[recover_id] = seq_len;
            stop_flags[recover_id] = false;
            input_ids_now[ori_seq_len_encoder + step_idx_now - 1] = next_tokens[recover_id]; // next tokens
            input_ids_now[0] = first_token_id; // set first prompt token
            const int ori_free_list_len_tid0 = atomicSub(free_list_len, decoder_used_len);
            ori_free_list_len = ori_free_list_len_tid0;
#ifdef DEBUG_STEP
            printf("seq_id: %d, ori_seq_len_encoder: %d, step_idx_now: %d, seq_len: %d, ori_free_list_len_tid0: %d, ori_free_list_len: %d\n", 
                    recover_id, ori_seq_len_encoder, step_idx_now, seq_len, ori_free_list_len_tid0, ori_free_list_len);
#endif
        }
        __syncthreads();
        // 恢复block table
        for (int i = tid; i < decoder_used_len; i += blockDim.x) {
            block_table_now[encoder_block_len + i] = free_list[ori_free_list_len - i - 1];
        }
        // 恢复input_ids
        for (int i = tid; i < step_idx_now - 1; i += blockDim.x) {
            input_ids_now[ori_seq_len_encoder + i] = pre_ids_now[i + 1];
        }
    }

    if (bid == 0 && tid == 0) {
        recover_len[0] = 0;
    }
}

void StepPaddle(const paddle::Tensor& stop_flags,
                const paddle::Tensor& seq_lens_this_time,
                const paddle::Tensor& ori_seq_lens_encoder,
                const paddle::Tensor& seq_lens_encoder,
                const paddle::Tensor& seq_lens_decoder,
                const paddle::Tensor& block_tables, // [bsz, block_num_per_seq]
                const paddle::Tensor& encoder_block_lens,
                const paddle::Tensor& is_block_step,
                const paddle::Tensor& step_block_list,
                const paddle::Tensor& step_lens,
                const paddle::Tensor& recover_block_list,
                const paddle::Tensor& recover_lens,
                const paddle::Tensor& need_block_list,
                const paddle::Tensor& need_block_len,
                const paddle::Tensor& used_list_len, 
                const paddle::Tensor& free_list,
                const paddle::Tensor& free_list_len,
                const paddle::Tensor& input_ids,
                const paddle::Tensor& pre_ids,
                const paddle::Tensor& step_idx,
                const paddle::Tensor& next_tokens,
                const int block_size,
                const int encoder_decoder_block_num,
                const int64_t first_token_id,
                const int speculate_step_token_num) {
    auto cu_stream = seq_lens_this_time.stream();
    const int bsz = seq_lens_this_time.shape()[0];
    const int block_num_per_seq = block_tables.shape()[1];
    const int length = input_ids.shape()[1];
    const int pre_id_length = pre_ids.shape()[1];
    constexpr int BlockSize = 512; // bsz < 256
    const int max_decoder_block_num = length / block_size;
#ifdef DEBUG_STEP
    printf("bsz: %d, block_num_per_seq: %d, length: %d, max_decoder_block_num: %d\n", bsz, block_num_per_seq, length, max_decoder_block_num);
#endif
    free_and_dispatch_block<<<1, BlockSize, 0, cu_stream>>>(
        const_cast<bool*>(stop_flags.data<bool>()),
        const_cast<int*>(seq_lens_this_time.data<int>()),
        const_cast<int*>(seq_lens_decoder.data<int>()),
        const_cast<int*>(block_tables.data<int>()),
        const_cast<int*>(encoder_block_lens.data<int>()),
        const_cast<bool*>(is_block_step.data<bool>()),
        const_cast<int*>(step_block_list.data<int>()),
        const_cast<int*>(step_lens.data<int>()),
        const_cast<int*>(recover_block_list.data<int>()),
        const_cast<int*>(recover_lens.data<int>()),
        const_cast<int*>(need_block_list.data<int>()),
        const_cast<int*>(need_block_len.data<int>()),
        const_cast<int*>(used_list_len.data<int>()),
        const_cast<int*>(free_list.data<int>()),
        const_cast<int*>(free_list_len.data<int>()),
        bsz,
        block_size,
        block_num_per_seq,
        max_decoder_block_num,
        speculate_step_token_num
    );
#ifdef DEBUG_STEP
#ifdef PADDLE_WITH_HIP
    hipDeviceSynchronize();
#else
    cudaDeviceSynchronize();
#endif
#endif
    auto cpu_recover_lens = recover_lens.copy_to(paddle::CPUPlace(), false);
    const int grid_size = cpu_recover_lens.data<int>()[0];
#ifdef DEBUG_STEP
    printf("grid_size2 %d\n", grid_size);
#endif
    if (grid_size > 0) {
        recover_block<<<grid_size, BlockSize, 0, cu_stream>>>(
            const_cast<int*>(recover_block_list.data<int>()),
            const_cast<int*>(recover_lens.data<int>()),
            const_cast<bool*>(stop_flags.data<bool>()),
            const_cast<int*>(seq_lens_this_time.data<int>()),
            const_cast<int*>(ori_seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_decoder.data<int>()),
            const_cast<int*>(block_tables.data<int>()),
            const_cast<int*>(free_list.data<int>()),
            const_cast<int*>(free_list_len.data<int>()),
            const_cast<int64_t*>(input_ids.data<int64_t>()),
            const_cast<int64_t*>(pre_ids.data<int64_t>()),
            const_cast<int64_t*>(step_idx.data<int64_t>()),
            const_cast<int*>(encoder_block_lens.data<int>()),
            const_cast<int*>(used_list_len.data<int>()),
            next_tokens.data<int64_t>(),
            bsz,
            block_num_per_seq,
            length,
            pre_id_length,
            first_token_id
        );
#ifdef DEBUG_STEP
#ifdef PADDLE_WITH_HIP
        hipDeviceSynchronize();
#else
        cudaDeviceSynchronize();
#endif
#endif
    }
}

PD_BUILD_OP(step_paddle)
    .Inputs({"stop_flags", 
             "seq_lens_this_time",
             "ori_seq_lens_encoder",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables",
             "encoder_block_lens",
             "is_block_step",
             "step_block_list",
             "step_lens",
             "recover_block_list",
             "recover_lens",
             "need_block_list",
             "need_block_len",
             "used_list_len",
             "free_list",
             "free_list_len",
             "input_ids",
             "pre_ids",
             "step_idx",
             "next_tokens"})
    .Attrs({"block_size: int",
            "encoder_decoder_block_num: int",
            "first_token_id: int64_t",
            "speculate_step_token_num: int"})
    .Outputs({"stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "block_tables_out",
              "encoder_block_lens_out",
              "is_block_step_out",
              "step_block_list_out",
              "step_lens_out",
              "recover_block_list_out",
              "recover_lens_out",
              "need_block_list_out",
              "need_block_len_out",
              "used_list_len_out",
              "free_list_out",
              "free_list_len_out",
              "input_ids_out"})
    .SetInplaceMap({{"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"block_tables", "block_tables_out"},
                    {"encoder_block_lens", "encoder_block_lens_out"},
                    {"is_block_step", "is_block_step_out"},
                    {"step_block_list", "step_block_list_out"},
                    {"step_lens", "step_lens_out"},
                    {"recover_block_list", "recover_block_list_out"},
                    {"recover_lens", "recover_lens_out"},
                    {"need_block_list", "need_block_list_out"},
                    {"need_block_len", "need_block_len_out"},
                    {"used_list_len", "used_list_len_out"},
                    {"free_list", "free_list_out"},
                    {"free_list_len", "free_list_len_out"},
                    {"input_ids", "input_ids_out"}})
    .SetKernelFn(PD_KERNEL(StepPaddle));