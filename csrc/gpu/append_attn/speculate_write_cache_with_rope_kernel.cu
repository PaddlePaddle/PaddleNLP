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

#include "speculate_write_cache_with_rope_kernel.h"
#include "utils.cuh"

// rope + write
template <typename T, typename QKV_TYPE>
void append_speculate_cache_rope(const QKV_TYPE* qkv,
                                 T* key_cache,
                                 T* value_cache,
                                 T* qkv_out,
                                 const int* block_tables,
                                 const int* padding_offsets,
                                 const int* cum_offsets,
                                 const int* seq_lens,
                                 const int* seq_lens_encoder,
                                 const float* cos_emb,
                                 const float* sin_emb,
                                 const float* qkv_out_scales,
                                 const T* qkv_biases,
                                 const int max_seq_len,
                                 const int max_blocks_per_seq,
                                 const int num_heads,
                                 const int kv_num_heads,
                                 const int dim_head,
                                 const int block_size,
                                 const int bsz,
                                 const int token_num,
                                 const cudaStream_t& stream,
                                 const bool use_neox_style) {
  int output_inner_dim = num_heads + 2 * kv_num_heads;

  const uint32_t elem_nums =
      use_neox_style ? token_num * (num_heads + 2 * kv_num_heads) * dim_head / 2
                     : token_num * (num_heads + 2 * kv_num_heads) * dim_head;
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;

  const int threads_per_block = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (use_neox_style) {
    append_speculate_cache_neox_rope_kernel<T, PackSize>
        <<<grid_size, threads_per_block, 0, stream>>>(
            qkv,  // [token_num, num_heads + 2 * gqa_group_size, head_size]
            key_cache,
            value_cache,
            qkv_out,
            block_tables,
            padding_offsets,
            cum_offsets,
            seq_lens,
            cos_emb,
            sin_emb,
            qkv_out_scales,
            qkv_biases,  // [num_head + 2 * gqa_group_size, dim_head]
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            output_inner_dim,
            dim_head,
            block_size,
            elem_nums,
            kv_num_heads);
  } else {
    append_speculate_cache_rope_kernel<T, PackSize>
        <<<grid_size, threads_per_block, 0, stream>>>(
            qkv,  // [token_num, num_heads + 2 * gqa_group_size, head_size]
            key_cache,
            value_cache,
            qkv_out,
            block_tables,
            padding_offsets,
            cum_offsets,
            seq_lens,
            cos_emb,
            sin_emb,
            qkv_out_scales,
            qkv_biases,  // [num_head + 2 * gqa_group_size, dim_head]
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            output_inner_dim,
            dim_head,
            block_size,
            elem_nums,
            kv_num_heads);
  }
}

template <typename T, typename QKV_TYPE>
void append_speculate_cache_int8_rope(const QKV_TYPE* qkv,
                                      uint8_t* key_cache,
                                      uint8_t* value_cache,
                                      T* qkv_out,
                                      const int* block_tables,
                                      const int* padding_offsets,
                                      const int* cum_offsets,
                                      const int* seq_lens,
                                      const int* seq_lens_encoder,
                                      const float* cos_emb,
                                      const float* sin_emb,
                                      const float* qkv_out_scales,
                                      const T* qkv_biases,
                                      const T* cache_k_scale,
                                      const T* cache_v_scale,
                                      const int max_seq_len,
                                      const int max_blocks_per_seq,
                                      const int num_heads,
                                      const int kv_num_heads,
                                      const int dim_head,
                                      const int block_size,
                                      const int bsz,
                                      const int token_num,
                                      const cudaStream_t& stream,
                                      const bool use_neox_style) {
  constexpr int num_warps = 4;
  const int all_warps =
      ((num_heads + 2 * kv_num_heads) + num_warps - 1) / num_warps * num_warps;
  dim3 grids(token_num, all_warps / num_warps);

  append_clear_cache_int8_block<4>
      <<<grids, num_warps * 32, 0, stream>>>(key_cache,
                                             value_cache,
                                             seq_lens,
                                             block_tables,
                                             padding_offsets,
                                             cum_offsets,
                                             seq_lens_encoder,
                                             max_seq_len,
                                             max_blocks_per_seq,
                                             num_heads,
                                             block_size,
                                             kv_num_heads);
  if (use_neox_style) {
    append_speculate_cache_int8_neox_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(qkv,
                                               key_cache,
                                               value_cache,
                                               qkv_out,
                                               block_tables,
                                               padding_offsets,
                                               cum_offsets,
                                               seq_lens,
                                               seq_lens_encoder,
                                               cos_emb,
                                               sin_emb,
                                               qkv_out_scales,
                                               qkv_biases,
                                               cache_k_scale,
                                               cache_v_scale,
                                               max_seq_len,
                                               max_blocks_per_seq,
                                               num_heads,
                                               block_size,
                                               127.0f,
                                               -127.0f,
                                               kv_num_heads);
  } else {
    append_speculate_cache_int8_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(qkv,
                                               key_cache,
                                               value_cache,
                                               qkv_out,
                                               block_tables,
                                               padding_offsets,
                                               cum_offsets,
                                               seq_lens,
                                               seq_lens_encoder,
                                               cos_emb,
                                               sin_emb,
                                               qkv_out_scales,
                                               qkv_biases,
                                               cache_k_scale,
                                               cache_v_scale,
                                               max_seq_len,
                                               max_blocks_per_seq,
                                               num_heads,
                                               block_size,
                                               127.0f,
                                               -127.0f,
                                               kv_num_heads);
  }
}

template <typename T, typename QKV_TYPE>
void append_speculate_cache_int4_rope(const QKV_TYPE* qkv,
                                      uint8_t* key_cache,
                                      uint8_t* value_cache,
                                      T* qkv_out,
                                      const int* block_tables,
                                      const int* padding_offsets,
                                      const int* cum_offsets,
                                      const int* seq_lens,
                                      const int* seq_lens_encoder,
                                      const float* cos_emb,
                                      const float* sin_emb,
                                      const float* qkv_out_scales,
                                      const T* qkv_biases,
                                      const T* cache_k_scale,
                                      const T* cache_v_scale,
                                      const T* cache_k_zp,
                                      const T* cache_v_zp,
                                      const int max_seq_len,
                                      const int max_blocks_per_seq,
                                      const int num_heads,
                                      const int kv_num_heads,
                                      const int dim_head,
                                      const int block_size,
                                      const int bsz,
                                      const int token_num,
                                      const cudaStream_t& stream,
                                      const bool use_neox_style) {
  constexpr int num_warps = 4;
  const int all_warps =
      ((num_heads + 2 * kv_num_heads) + num_warps - 1) / num_warps * num_warps;
  dim3 grids(token_num, all_warps / num_warps);

  append_clear_cache_int4_block<4>
      <<<grids, num_warps * 32, 0, stream>>>(key_cache,
                                             value_cache,
                                             seq_lens,
                                             block_tables,
                                             padding_offsets,
                                             cum_offsets,
                                             seq_lens_encoder,
                                             max_seq_len,
                                             max_blocks_per_seq,
                                             num_heads,
                                             block_size,
                                             kv_num_heads);
  if (use_neox_style) {
    append_speculate_cache_int4_neox_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(qkv,
                                               key_cache,
                                               value_cache,
                                               qkv_out,
                                               block_tables,
                                               padding_offsets,
                                               cum_offsets,
                                               seq_lens,
                                               seq_lens_encoder,
                                               cos_emb,
                                               sin_emb,
                                               qkv_out_scales,
                                               qkv_biases,
                                               cache_k_scale,
                                               cache_v_scale,
                                               cache_k_zp,
                                               cache_v_zp,
                                               max_seq_len,
                                               max_blocks_per_seq,
                                               num_heads,
                                               block_size,
                                               7.0f,
                                               -8.0f,
                                               kv_num_heads);
  } else {
    append_speculate_cache_int4_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(qkv,
                                               key_cache,
                                               value_cache,
                                               qkv_out,
                                               block_tables,
                                               padding_offsets,
                                               cum_offsets,
                                               seq_lens,
                                               seq_lens_encoder,
                                               cos_emb,
                                               sin_emb,
                                               qkv_out_scales,
                                               qkv_biases,
                                               cache_k_scale,
                                               cache_v_scale,
                                               cache_k_zp,
                                               cache_v_zp,
                                               max_seq_len,
                                               max_blocks_per_seq,
                                               num_heads,
                                               block_size,
                                               7.0f,
                                               -8.0f,
                                               kv_num_heads);
  }
}
template <typename T, typename QKV_TYPE>
void SpeculateWriteCacheWithRoPEKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out) {
  typedef cascade_attn_type_traits<T> traits_;
  typedef cascade_attn_type_traits<QKV_TYPE> qkt_nv_type_;
  typedef typename traits_::type DataType_;
  typedef typename qkt_nv_type_::type QKV_Data_TYPE;
  const QKV_TYPE* qkv_ptr = qkv.data<QKV_TYPE>();

  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto bsz = meta_data.batch_size;
  auto token_nums = meta_data.token_nums;
  auto block_size = meta_data.block_size;
  auto dim_head = meta_data.head_dims;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;


  const float* cos_emb =
      rotary_embs ? rotary_embs.get().data<float>() : nullptr;
  const float* sin_emb;
  if (rotary_embs) {
    sin_emb =
        use_neox_rotary_style
            ? rotary_embs.get().data<float>() + max_seq_len * dim_head
            : rotary_embs.get().data<float>() + max_seq_len * dim_head / 2;
  }
  if (cache_quant_type_str == "none") {
    append_speculate_cache_rope(
        reinterpret_cast<const QKV_TYPE*>(qkv_ptr),
        reinterpret_cast<DataType_*>(key_cache_out->data<T>()),
        reinterpret_cast<DataType_*>(value_cache_out->data<T>()),
        reinterpret_cast<DataType_*>(qkv_out->data<T>()),
        block_tables.data<int>(),
        padding_offsets.data<int>(),
        cum_offsets.data<int>(),
        seq_lens.data<int>(),
        seq_lens_encoder.data<int>(),
        cos_emb,
        sin_emb,
        qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
        qkv_biases ? reinterpret_cast<DataType_*>(
                         const_cast<T*>(qkv_biases.get().data<T>()))
                   : nullptr,
        max_seq_len,
        max_blocks_per_seq,
        num_heads,
        kv_num_heads,
        dim_head,
        block_size,
        bsz,
        token_nums,
        stream,
        use_neox_rotary_style);
  } else if (cache_quant_type_str == "cache_int8") {
    append_speculate_cache_int8_rope(
        reinterpret_cast<const QKV_TYPE*>(qkv_ptr),
        key_cache_out->data<uint8_t>(),
        value_cache_out->data<uint8_t>(),
        reinterpret_cast<DataType_*>(qkv_out->data<T>()),
        block_tables.data<int>(),
        padding_offsets.data<int>(),
        cum_offsets.data<int>(),
        seq_lens.data<int>(),
        seq_lens_encoder.data<int>(),
        cos_emb,
        sin_emb,
        qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
        qkv_biases ? reinterpret_cast<DataType_*>(
                         const_cast<T*>(qkv_biases.get().data<T>()))
                   : nullptr,
        cache_k_scale ? reinterpret_cast<DataType_*>(
                            const_cast<T*>(cache_k_scale.get().data<T>()))
                      : nullptr,
        cache_v_scale ? reinterpret_cast<DataType_*>(
                            const_cast<T*>(cache_v_scale.get().data<T>()))
                      : nullptr,
        max_seq_len,
        max_blocks_per_seq,
        num_heads,
        kv_num_heads,
        dim_head,
        block_size,
        bsz,
        token_nums,
        stream,
        use_neox_rotary_style);
  } else if (cache_quant_type_str == "cache_int4_zp") {
    append_speculate_cache_int4_rope(
        reinterpret_cast<const QKV_TYPE*>(qkv_ptr),
        key_cache_out->data<uint8_t>(),
        value_cache_out->data<uint8_t>(),
        reinterpret_cast<DataType_*>(const_cast<T*>(qkv_out->data<T>())),
        block_tables.data<int>(),
        padding_offsets.data<int>(),
        cum_offsets.data<int>(),
        seq_lens.data<int>(),
        seq_lens_encoder.data<int>(),
        cos_emb,
        sin_emb,
        qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
        qkv_biases ? reinterpret_cast<DataType_*>(
                         const_cast<T*>(qkv_biases.get().data<T>()))
                   : nullptr,
        cache_k_scale ? reinterpret_cast<DataType_*>(
                            const_cast<T*>(cache_k_scale.get().data<T>()))
                      : nullptr,
        cache_v_scale ? reinterpret_cast<DataType_*>(
                            const_cast<T*>(cache_v_scale.get().data<T>()))
                      : nullptr,
        cache_k_zp ? reinterpret_cast<DataType_*>(
                         const_cast<T*>(cache_k_zp.get().data<T>()))
                   : nullptr,
        cache_v_zp ? reinterpret_cast<DataType_*>(
                         const_cast<T*>(cache_v_zp.get().data<T>()))
                   : nullptr,
        max_seq_len,
        max_blocks_per_seq,
        num_heads,
        kv_num_heads,
        dim_head,
        block_size,
        bsz,
        token_nums,
        stream,
        use_neox_rotary_style);
  } else {
    PD_THROW(
        "cache_quant_type_str should be one of [none, cache_int8, "
        "cache_int4_zp]");
  }
}

template void SpeculateWriteCacheWithRoPEKernel<paddle::bfloat16, int>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // gqa_group_size, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);

template void
SpeculateWriteCacheWithRoPEKernel<paddle::bfloat16, paddle::bfloat16>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // gqa_group_size, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);

template void SpeculateWriteCacheWithRoPEKernel<paddle::float16, int>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // gqa_group_size, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);


template void
SpeculateWriteCacheWithRoPEKernel<paddle::float16, paddle::float16>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // gqa_group_size, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);