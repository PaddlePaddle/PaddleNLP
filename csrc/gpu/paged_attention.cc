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

#include <stdio.h>
#include <string.h>
#include "paddle/extension.h"

void paged_attention_v1_opt_tc(
    const void* const out,    // [num_seqs, num_heads, head_size]
    const void* const query,  // [num_seqs, num_heads, head_size]
    const void* const key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    const void* const value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    const int* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* seq_lens,      // [num_seqs]
    int64_t block_size,
    int64_t max_seq_len,
    const float* alibi_slopes,
    const std::string& kv_cache_dtype, 
    const float* k_scale, 
    const float* v_scale, 
    const int64_t tp_rank, 
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, 
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step,
    const int q_num_seqs,
    const int q_num_heads,
    const int q_head_size,
    const int q_row_stride,
    const int max_num_blocks_per_seq,
    const int kv_block_stride,
    const int kv_head_stride,
    const std::string& query_dtype,
    hipStream_t stream);

void paged_attention_v2_opt_tc(
    const void* const out,         // [num_seqs, num_heads, head_size]
    const float* const exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    const float* const max_logits,  // [num_seqs, num_heads, max_num_partitions]
    const void* const tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    const void* const query,  // [num_seqs, num_heads, head_size]
    const void* const key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    const void* const value_cache,       // [num_blocks, num_heads, head_size, block_size]
    const int64_t num_kv_heads,  // [num_heads]
    const double scale,
    const int* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* seq_lens,      // [num_seqs]
    const int64_t block_size,
    const int64_t max_seq_len,
    const float* alibi_slopes,
    const std::string& kv_cache_dtype,
    const float* k_scale,
    const float* v_scale,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step,
    const int q_num_seqs,
    const int q_num_heads,
    const int q_head_size,
    const int q_row_stride,
    const int max_num_blocks_per_seq,
    const int k_cache_num_blocks,
    const int kv_block_stride,
    const int kv_head_stride,
    const std::string& query_dtype,
    hipStream_t stream);


void PagedAttentionV1Tc(
    const paddle::Tensor& out,
    const paddle::Tensor& query,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& seq_lens,
    const paddle::optional<paddle::Tensor>& alibi_slopes,
    const paddle::Tensor& k_scale,
    const paddle::Tensor& v_scale,
    int64_t num_kv_heads,
    double scale,
    int64_t block_size,
    int64_t max_seq_len,
    const std::string& kv_cache_dtype,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  auto query_dtype_pp = query.dtype();
  std::string query_dtype;
  switch(query_dtype_pp) {
    case paddle::DataType::FLOAT32:
      query_dtype = "float";
      break;
    case paddle::DataType::FLOAT16:
      query_dtype = "float16";
      break;
    case paddle::DataType::BFLOAT16:
      query_dtype = "bfloat16";
      break;
    default:
      PD_THROW("Only supported query dtype in ['float', 'float16', 'bfloat16'].");
          break;
  }

  paged_attention_v1_opt_tc(
    out.data(),
    query.data(),
    key_cache.data(),
    value_cache.data(),
    num_kv_heads,
    scale,
    block_tables.data<int32_t>(),
    seq_lens.data<int32_t>(),
    block_size,
    max_seq_len,
    alibi_slopes ? alibi_slopes.get().data<float>() : nullptr,
    kv_cache_dtype,
    k_scale.data<float>(),
    v_scale.data<float>(),
    tp_rank,
    blocksparse_local_blocks,
    blocksparse_vert_stride,
    blocksparse_block_size,
    blocksparse_head_sliding_step,
    query.shape()[0],
    query.shape()[1],
    query.shape()[2],
    query.strides()[0],
    block_tables.shape()[1],
    key_cache.strides()[0],
    key_cache.strides()[1],
    query_dtype,
    query.stream());
}

void PagedAttentionV2Tc(
    const paddle::Tensor& out,
    const paddle::Tensor& exp_sums,
    const paddle::Tensor& max_logits,
    const paddle::Tensor& tmp_out,
    const paddle::Tensor& query,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& seq_lens,
    const paddle::optional<paddle::Tensor>& alibi_slopes,
    const paddle::Tensor& k_scale,
    const paddle::Tensor& v_scale,
    int64_t num_kv_heads,
    double scale,
    int64_t block_size,
    int64_t max_seq_len,
    const std::string& kv_cache_dtype,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  auto query_dtype_pp = query.dtype();
  std::string query_dtype;
  switch(query_dtype_pp) {
    case paddle::DataType::FLOAT32:
      query_dtype = "float";
      break;
    case paddle::DataType::FLOAT16:
      query_dtype = "float16";
      break;
    case paddle::DataType::BFLOAT16:
      query_dtype = "bfloat16";
      break;
    default:
      PD_THROW("Only supported query dtype in ['float', 'float16', 'bfloat16'].");
          break;
  }

  paged_attention_v2_opt_tc(
    out.data(),
    exp_sums.data<float>(),
    max_logits.data<float>(),
    tmp_out.data(),
    query.data(),
    key_cache.data(),
    value_cache.data(),
    num_kv_heads,
    scale,
    block_tables.data<int32_t>(),
    seq_lens.data<int32_t>(),
    block_size,
    max_seq_len,
    alibi_slopes ? alibi_slopes.get().data<float>() : nullptr,
    kv_cache_dtype,
    k_scale.data<float>(),
    v_scale.data<float>(),
    tp_rank,
    blocksparse_local_blocks,
    blocksparse_vert_stride,
    blocksparse_block_size,
    blocksparse_head_sliding_step,
    query.shape()[0],
    query.shape()[1],
    query.shape()[2],
    query.strides()[0],
    block_tables.shape()[1],
    key_cache.shape()[0],
    key_cache.strides()[0],
    key_cache.strides()[1],
    query_dtype,
    query.stream());
}

PD_BUILD_OP(paged_attention_v1_opt_tc)
    .Inputs({"out",
             "query",
             "key_cache",
             "value_cache",
             "block_tables",
             "seq_lens",
             paddle::Optional("alibi_slopes"),
             "k_scale",
             "v_scale"})
    .Outputs({"output", "key_cache_out", "value_cache_out"})
    .SetInplaceMap({{"out", "output"},
                    {"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .Attrs({"num_kv_heads: int64_t",
            "scale: double",
            "block_size: int64_t",
            "max_seq_len: int64_t",
            "kv_cache_dtype: std::string",
            "tp_rank: int64_t",
            "blocksparse_local_blocks: int64_t",
            "blocksparse_vert_stride: int64_t",
            "blocksparse_block_size: int64_t",
            "blocksparse_head_sliding_step: int64_t"})
    .SetKernelFn(PD_KERNEL(PagedAttentionV1Tc));

PD_BUILD_OP(paged_attention_v2_opt_tc)
    .Inputs({"out",
             "exp_sums",
             "max_logits",
             "tmp_out",
             "query",
             "key_cache",
             "value_cache",
             "block_tables",
             "seq_lens",
             paddle::Optional("alibi_slopes"),
             "k_scale",
             "v_scale"})
    .Outputs({"output", "key_cache_out", "value_cache_out"})
    .SetInplaceMap({{"out", "output"},
                    {"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .Attrs({"num_kv_heads: int64_t",
            "scale: double",
            "block_size: int64_t",
            "max_seq_len: int64_t",
            "kv_cache_dtype: std::string",
            "tp_rank: int64_t",
            "blocksparse_local_blocks: int64_t",
            "blocksparse_vert_stride: int64_t",
            "blocksparse_block_size: int64_t",
            "blocksparse_head_sliding_step: int64_t"})
    .SetKernelFn(PD_KERNEL(PagedAttentionV2Tc));