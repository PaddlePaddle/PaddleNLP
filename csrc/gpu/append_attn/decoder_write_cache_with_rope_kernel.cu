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

#include "decoder_write_cache_with_rope_kernel.h"

template <typename T, typename QKV_TYPE>
void append_decode_cache_rope(const QKV_TYPE* qkv,
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
                              const int head_size,
                              const int block_size,
                              const int bsz,
                              const cudaStream_t& stream,
                              const bool use_neox_style) {
  const uint32_t elem_nums = use_neox_style ? bsz * (num_heads + 2 * kv_num_heads) * head_size / 2 : bsz * (num_heads + 2 * kv_num_heads) * head_size;

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  if (use_neox_style) {
    if (qkv_out_scales) {
      append_decode_cache_T_neox_rope_kernel<T, PackSize>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<const int*>(qkv),
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
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            kv_num_heads);
    } else {
      append_decode_cache_T_neox_rope_kernel<T, PackSize>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<const T*>(qkv),
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
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            kv_num_heads);
    }
  } else {
    if (qkv_out_scales) {
      append_decode_cache_T_rope_kernel<T, PackSize>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<const int*>(qkv),
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
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            kv_num_heads);
    } else {
      append_decode_cache_T_rope_kernel<T, PackSize>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<const T*>(qkv),
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
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            kv_num_heads);
    }
  }

}

template <typename T, typename QKV_TYPE>
void append_decode_cache_int8_rope(const QKV_TYPE* qkv,
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
                                  const int head_size,
                                  const int block_size,
                                  const int bsz,
                                  const cudaStream_t& stream,
                                  const bool use_neox_style) {
  constexpr int num_warps = 4;
  const int all_warps = ((num_heads + 2 * kv_num_heads) + num_warps - 1) /
                        num_warps * num_warps;
  dim3 grids(bsz, all_warps / num_warps);
  if (use_neox_style) {
    if (qkv_out_scales) {
      append_decode_cache_int8_neox_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(
          reinterpret_cast<const int*>(qkv),
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
      append_decode_cache_int8_neox_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(
          reinterpret_cast<const T*>(qkv),
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
  } else {
    if (qkv_out_scales) {
      append_decode_cache_int8_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(
          reinterpret_cast<const int*>(qkv),
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
      append_decode_cache_int8_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(
          reinterpret_cast<const T*>(qkv),
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
}

template <typename T, typename QKV_TYPE>
void append_decode_cache_int4_rope(const QKV_TYPE* qkv,
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
                                  const int head_size,
                                  const int block_size,
                                  const int bsz,
                                  const cudaStream_t& stream,
                                  const bool use_neox_style) {
  constexpr int num_warps = 4;
  const int all_warps = ((num_heads + 2 * kv_num_heads) + num_warps - 1) /
                        num_warps * num_warps;
  dim3 grids(bsz, all_warps / num_warps);
  if (use_neox_style) {
    if (qkv_out_scales) {
      append_decode_cache_int4_neox_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(
          reinterpret_cast<const int*>(qkv),
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
      append_decode_cache_int4_neox_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(
          reinterpret_cast<const T*>(qkv),
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
  } else {
    if (qkv_out_scales) {
      append_decode_cache_int4_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(
          reinterpret_cast<const int*>(qkv),
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
      append_decode_cache_int4_rope_kernel<T, 4>
        <<<grids, num_warps * 32, 0, stream>>>(
          reinterpret_cast<const T*>(qkv),
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
}
template <typename T, typename QKV_TYPE>
void DecoderWriteCacheWithRoPEKernel(
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
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out) {
  typedef cascade_attn_type_traits<T> traits_;
  typedef cascade_attn_type_traits<QKV_TYPE> qkt_nv_type_;
  typedef typename traits_::type DataType_;
  typedef typename qkt_nv_type_::type QKV_Data_TYPE;
  const QKV_TYPE* qkv_ptr = qkv.data<QKV_TYPE>();
  auto qkv_dims = qkv.dims();
  const int max_blocks_per_seq = block_tables.dims()[1];
  const int bsz = cum_offsets.dims()[0];

  // VLOG(1) << "gqa_group_size: " << gqa_group_size;
  const int32_t block_size = key_cache_out->dims()[2];

  const float* cos_emb = rotary_embs ? rotary_embs.get().data<float>() : nullptr;
  const float* sin_emb;
  if (rotary_embs) {
    sin_emb = use_neox_rotary_style ? rotary_embs.get().data<float>() + max_seq_len * head_size : rotary_embs.get().data<float>() + max_seq_len * head_size / 2;
  }
  if (cache_quant_type_str == "none") {
    append_decode_cache_rope(
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
                      const_cast<T*>(qkv_biases.get().data<T>())) : nullptr,
      max_seq_len,
      max_blocks_per_seq,
      num_heads,
      kv_num_heads,
      head_size,
      block_size,
      bsz,
      stream,
      use_neox_rotary_style);
  } else if (cache_quant_type_str == "cache_int8") {
    append_decode_cache_int8_rope(
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
                      const_cast<T*>(qkv_biases.get().data<T>())) : nullptr,
      cache_k_scale ? reinterpret_cast<DataType_*>(
                      const_cast<T*>(cache_k_scale.get().data<T>())) : nullptr,
      cache_v_scale ? reinterpret_cast<DataType_*>(
                      const_cast<T*>(cache_v_scale.get().data<T>())) : nullptr,
      max_seq_len,
      max_blocks_per_seq,
      num_heads,
      kv_num_heads,
      head_size,
      block_size,
      bsz,
      stream,
      use_neox_rotary_style);
  } else if (cache_quant_type_str == "cache_int4_zp") {
    append_decode_cache_int4_rope(
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
                            const_cast<T*>(qkv_biases.get().data<T>())) : nullptr,
            cache_k_scale ? reinterpret_cast<DataType_*>(
                                const_cast<T*>(cache_k_scale.get().data<T>())) : nullptr,
            cache_v_scale ? reinterpret_cast<DataType_*>(
                                const_cast<T*>(cache_v_scale.get().data<T>())) : nullptr,
            cache_k_zp ? reinterpret_cast<DataType_*>(
                            const_cast<T*>(cache_k_zp.get().data<T>())) : nullptr,
            cache_v_zp ? reinterpret_cast<DataType_*>(
                            const_cast<T*>(cache_v_zp.get().data<T>())) : nullptr,
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            kv_num_heads,
            head_size,
            block_size,
            bsz,
            stream,
            use_neox_rotary_style);
  } else {
    PD_THROW("append attention just support C16/C8/C4_zp now!");
  }
}

template void DecoderWriteCacheWithRoPEKernel<paddle::bfloat16, int>(
    const paddle::Tensor& qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
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
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);

template void DecoderWriteCacheWithRoPEKernel<paddle::bfloat16, paddle::bfloat16>(
    const paddle::Tensor& qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
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
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);

template void DecoderWriteCacheWithRoPEKernel<paddle::float16, int>(
    const paddle::Tensor& qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
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
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);

template void DecoderWriteCacheWithRoPEKernel<paddle::float16, paddle::float16>(
    const paddle::Tensor& qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
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
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out);