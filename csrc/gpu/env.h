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

inline uint32_t get_decoder_block_shape_q() {
    static const char* decoder_block_shape_q_env = std::getenv("FLAGS_dec_block_shape_q");
    static const uint32_t decoder_block_shape_q =
            decoder_block_shape_q_env == nullptr ? 16 : std::stoi(std::string(decoder_block_shape_q_env));
    return decoder_block_shape_q;
}

inline uint32_t get_encoder_block_shape_q() {
    static const char* encoder_block_shape_q_env = std::getenv("FLAGS_enc_block_shape_q");
    static const uint32_t encoder_block_shape_q =
            encoder_block_shape_q_env == nullptr ? 64 : std::stoi(std::string(encoder_block_shape_q_env));
    return encoder_block_shape_q;
}

inline uint32_t get_max_partition_size(int bsz) {
    static const char* max_partition_size_env = std::getenv("FLAGS_cascade_attention_max_partition_size");
    static const uint32_t max_partition_size =
            max_partition_size_env == nullptr ? 32768 : std::stoul(std::string(max_partition_size_env));
    return max_partition_size;
}

inline uint32_t get_cascade_attention_deal_each_time() {
    static const char* cascade_attention_deal_each_time_env = std::getenv("FLAGS_cascade_attention_deal_each_time");
    static const uint32_t cascade_attention_deal_each_time =
            cascade_attention_deal_each_time_env == nullptr ? 0 : std::stoul(std::string(cascade_attention_deal_each_time_env));
    return (cascade_attention_deal_each_time != 0 ? cascade_attention_deal_each_time : 32);
}

inline uint32_t get_cascade_attention_num_stages() {
    static const char* cascade_attention_num_stages_env = std::getenv("FLAGS_cascade_attention_num_stages");
    static const uint32_t cascade_attention_num_stages =
            cascade_attention_num_stages_env == nullptr ? 0 : std::stoul(std::string(cascade_attention_num_stages_env));
    return cascade_attention_num_stages != 0 ? cascade_attention_num_stages : 2;
}

inline uint32_t get_cascade_attention_num_threads() {
    static const char* cascade_attention_num_threads_env = std::getenv("FLAGS_cascade_attention_num_threads");
    static const uint32_t cascade_attention_num_threads =
            cascade_attention_num_threads_env == nullptr ? 0 : std::stoul(std::string(cascade_attention_num_threads_env));
    return cascade_attention_num_threads != 0 ? cascade_attention_num_threads : 128;
}

inline bool get_flags_mla_use_tensorcore() {
    static const char* mla_use_tensorcore_env = std::getenv("FLAGS_mla_use_tensorcore");
    static const uint32_t mla_use_tensorcore =
            mla_use_tensorcore_env == nullptr ? 1 : std::stoul(std::string(mla_use_tensorcore_env));
    return mla_use_tensorcore != 0 ? true : false;
}

inline bool get_mla_use_wg4() {
    static const char* mla_use_wg4_env = std::getenv("FLAGS_mla_use_wg4");
    static const uint32_t mla_use_wg4 =
            mla_use_wg4_env == nullptr ? 1 : std::stoul(std::string(mla_use_wg4_env));
    return mla_use_wg4 != 0 ? true : false;
}

inline bool enable_cuda_core_fp8_gemm() {
    static const char* enable_cuda_core_fp8_env = std::getenv("FLAGS_cuda_core_fp8_gemm");
    static const bool enable_cuda_core_fp8_gemm =
            enable_cuda_core_fp8_env != nullptr && std::string(enable_cuda_core_fp8_env) == "1";
    return enable_cuda_core_fp8_gemm;
}

inline int get_mla_dec_chunk_size(int bsz) {
    static const char* mla_dec_chunk_size_env = std::getenv("FLAGS_mla_dec_chunk_size");
    static const int mla_dec_chunk_size =
            mla_dec_chunk_size_env == nullptr ? -1 : std::stoi(std::string(mla_dec_chunk_size_env));
    return bsz > 1 ? mla_dec_chunk_size : 64;
}