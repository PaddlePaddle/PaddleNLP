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

#include "encoder_write_cache_with_rope_kernel.h"

// #define DEBUG_APPEND

template <typename T>
void CascadeAppendWriteCacheKVQKV(const paddle::Tensor& qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
                                  const paddle::Tensor& block_table,
                                  const paddle::Tensor& padding_offsets,
                                  const paddle::Tensor& seq_lens_encoder,
                                  const paddle::Tensor& seq_lens_decoder,
                                  const int max_seq_len,
                                  const int num_heads,
                                  const int head_dim,
                                  const int kv_num_heads,
                                  cudaStream_t& stream,
                                  paddle::Tensor *key_cache_out, 
                                  paddle::Tensor *value_cache_out) {
  auto qkv_dims = qkv.dims();
  const int max_blocks_per_seq = block_table.dims()[1];
  const int num_tokens = qkv_dims[0];

  const int32_t block_size = key_cache_out->dims()[2];
  const uint32_t elem_nums = num_tokens * 2 * kv_num_heads * head_dim; // just k and v
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  cache_kernel<T, PackSize><<<grid_size, blocksize, 0, stream>>>(
    reinterpret_cast<T*>(const_cast<T*>(qkv.data<T>())),
    reinterpret_cast<T*>(key_cache_out->data<T>()),
    reinterpret_cast<T*>(value_cache_out->data<T>()),
    block_table.data<int>(),
    padding_offsets.data<int>(),
    seq_lens_encoder.data<int>(),
    seq_lens_decoder.data<int>(),
    max_seq_len,
    max_blocks_per_seq,
    num_heads,
    head_dim,
    block_size,
    elem_nums,
    kv_num_heads);
    }
template <typename T, uint32_t HEAD_DIM, uint32_t BLOCK_SIZE>
void CascadeAppendWriteCacheKVC8QKV(const paddle::Tensor &cache_k, // [max_block_num, num_heads, block_size, head_dim]
                                    const paddle::Tensor &cache_v, // [max_block_num, num_heads, head_dim, block_size]
                                    const paddle::Tensor &qkv, // [token_num, num_heads, head_dim]
                                    const paddle::Tensor &cache_k_scale, // [num_kv_heads, head_dim]
                                    const paddle::Tensor &cache_v_scale, // [num_kv_heads, head_dim]
                                    const paddle::Tensor &seq_lens_this_time,
                                    const paddle::Tensor &seq_lens_decoder,
                                    const paddle::Tensor &padding_offsets,
                                    const paddle::Tensor &cum_offsets,
                                    const paddle::Tensor &block_table,
                                    const paddle::Tensor &batch_ids,
                                    const paddle::Tensor &tile_ids_per_batch,
                                    int num_blocks_x_cpu,
                                    int max_seq_len,
                                    int q_num_heads,
                                    int kv_num_heads,
                                    cudaStream_t& stream,
                                    paddle::Tensor *cache_k_out,
                                    paddle::Tensor *cache_v_out) {
  const auto &qkv_dims = qkv.dims();
  const auto &cache_k_dims = cache_k.dims();
  const auto &cum_offsets_dims = cum_offsets.dims();
  const uint32_t token_num = qkv_dims[0];
  const uint32_t num_heads = kv_num_heads;
  const uint32_t bsz = cum_offsets_dims[0];
  const int max_block_num_per_seq = block_table.dims()[1];
  // VLOG(1) << "bsz: " << bsz << ", token_num: " << token_num << ", kv_num_heads: " << num_heads;
  // dev_ctx.template Alloc<uint8_t>(cache_k_out);
  // dev_ctx.template Alloc<uint8_t>(cache_v_out); 

  const uint32_t pad_len = BLOCK_SIZE;

  constexpr uint32_t num_warps = 4; 
  constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / num_warps;
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_row_per_block = num_warps * num_frags_z * 16;
  // VLOG(1) << "num_warps: " << num_warps << ", num_frags_z: " << num_frags_z << ", num_frags_y: " << num_frags_y << ", num_row_per_block: " << num_row_per_block;

  // VLOG(1) << "batch_ids: " << batch_ids;
  // VLOG(1) << "tile_ids_per_batch: " << tile_ids_per_batch;

  // int num_blocks_x_cpu;
  // paddle::memory::Copy(paddle::platform::CPUPlace(),
  //                      &num_blocks_x_cpu,
  //                      dev_ctx.GetPlace(),
  //                      num_blocks_x.data<int>(),
  //                      sizeof(int),
  //                      stream);
  // VLOG(1) << "num_blocks_x_cpu: " << num_blocks_x_cpu;

  dim3 grids(num_blocks_x_cpu, 1, num_heads);
  dim3 blocks(32, num_warps);

  const uint32_t smem_size = (BLOCK_SIZE * HEAD_DIM) * sizeof(T) * 2;
  auto kernel_fn = append_write_cache_kv_c8_qkv<T, num_frags_y, num_frags_z, HEAD_DIM, BLOCK_SIZE, num_warps>;
  // if (smem_size >= 48 * 1024) {
  cudaFuncSetAttribute(
      kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  // }
  kernel_fn<<<grids, blocks, 0, stream>>>(
    cache_k_out->data<uint8_t>(),
    cache_v_out->data<uint8_t>(),
    qkv.data<T>(),
    cache_k_scale.data<T>(),
    cache_v_scale.data<T>(),
    batch_ids.data<int>(),
    tile_ids_per_batch.data<int>(),
    seq_lens_this_time.data<int>(),
    seq_lens_decoder.data<int>(),
    padding_offsets.data<int>(),
    cum_offsets.data<int>(),
    block_table.data<int>(),
    max_seq_len,
    max_block_num_per_seq,
    q_num_heads,
    kv_num_heads
  );
}


template <typename T, uint32_t HEAD_DIM, uint32_t BLOCK_SIZE>
void CascadeAppendWriteCacheKVC4QKV(const paddle::Tensor &cache_k, // [max_block_num, num_heads, block_size, head_dim]
                                    const paddle::Tensor &cache_v, // [max_block_num, num_heads, head_dim, block_size]
                                    const paddle::Tensor &qkv, // [token_num, num_heads, head_dim]
                                    const paddle::Tensor &cache_k_scale, // [num_kv_heads, head_dim]
                                    const paddle::Tensor &cache_v_scale, // [num_kv_heads, head_dim]
                                    const paddle::Tensor &cache_k_zp, // [num_kv_heads, head_dim]
                                    const paddle::Tensor &cache_v_zp, // [num_kv_heads, head_dim]
                                    const paddle::Tensor &seq_lens_this_time,
                                    const paddle::Tensor &seq_lens_decoder,
                                    const paddle::Tensor &padding_offsets,
                                    const paddle::Tensor &cum_offsets,
                                    const paddle::Tensor &block_table,
                                    const paddle::Tensor &batch_ids,
                                    const paddle::Tensor &tile_ids_per_batch,
                                    int num_blocks_x_cpu,
                                    int max_seq_len,
                                    int q_num_heads,
                                    int kv_num_heads,
                                    cudaStream_t& stream,
                                    paddle::Tensor *cache_k_out,
                                    paddle::Tensor *cache_v_out) {
  const auto &qkv_dims = qkv.dims();
  const auto &cache_k_dims = cache_k.dims();
  const auto &cum_offsets_dims = cum_offsets.dims();
  const uint32_t token_num = qkv_dims[0];
  const uint32_t num_heads = kv_num_heads;
  const uint32_t bsz = cum_offsets_dims[0];
  const int max_block_num_per_seq = block_table.dims()[1];
  // VLOG(1) << "bsz: " << bsz << ", token_num: " << token_num << ", kv_num_heads: " << num_heads;
  // dev_ctx.template Alloc<uint8_t>(cache_k_out);
  // dev_ctx.template Alloc<uint8_t>(cache_v_out); 

  const uint32_t pad_len = BLOCK_SIZE;

  constexpr uint32_t num_warps = 4; 
  constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / num_warps;
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_row_per_block = num_warps * num_frags_z * 16;
  // VLOG(1) << "num_warps: " << num_warps << ", num_frags_z: " << num_frags_z << ", num_frags_y: " << num_frags_y << ", num_row_per_block: " << num_row_per_block;

  dim3 grids(num_blocks_x_cpu, 1, num_heads);
  dim3 blocks(32, num_warps);

  const uint32_t smem_size = (BLOCK_SIZE * HEAD_DIM) * sizeof(T) * 2 + HEAD_DIM * 4 * sizeof(T);
  // VLOG(1) << "smem_size: " << smem_size / 1024 << "KB";
  auto kernel_fn = append_write_cache_kv_c4_qkv<T, num_frags_y, num_frags_z, HEAD_DIM, BLOCK_SIZE, num_warps>;
  cudaFuncSetAttribute(
      kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kernel_fn<<<grids, blocks, 0, stream>>>(
    cache_k_out->data<uint8_t>(),
    cache_v_out->data<uint8_t>(),
    qkv.data<T>(),
    cache_k_scale.data<T>(),
    cache_v_scale.data<T>(),
    cache_k_zp.data<T>(),
    cache_v_zp.data<T>(),
    batch_ids.data<int>(),
    tile_ids_per_batch.data<int>(),
    seq_lens_this_time.data<int>(),
    seq_lens_decoder.data<int>(),
    padding_offsets.data<int>(),
    cum_offsets.data<int>(),
    block_table.data<int>(),
    max_seq_len,
    max_block_num_per_seq,
    q_num_heads,
    kv_num_heads
  );
}

template <typename T, typename QKV_TYPE>
void rotary_qk_variable(T *qkv_out, // [token_num, 3, num_head, dim_head]
                        const QKV_TYPE *qkv_input,  // qkv
                        const float *qkv_out_scales, // [3, num_head, dim_head]
                        const T *qkv_bias,
                        const float *rotary_emb, // [2, 1, 1, seq_len, dim_head / 2]
                        const int *padding_offsets,
                        const int *seq_lens,
                        const int *seq_lens_decoder,
                        const int token_num,
                        const int head_num,
                        const int seq_len,
                        const int input_output_len,
                        const int dim_head,
                        const cudaStream_t& stream,
                        bool use_neox_style = false) {
  int elem_nums = token_num * 3 * head_num * dim_head; // for all q k v
  if (use_neox_style) {
    elem_nums = token_num * 3 * head_num * dim_head / 2;
  }

  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    VariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
        qkv_input,
        cos_emb,
        sin_emb,
        padding_offsets,
        seq_lens,
        seq_lens_decoder,
        qkv_out_scales,
        qkv_bias,
        qkv_out,
        elem_nums,
        head_num,
        seq_len,
        dim_head);
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    NeoxVariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
        qkv_input,
        cos_emb,
        sin_emb,
        padding_offsets,
        seq_lens,
        seq_lens_decoder,
        qkv_out_scales,
        qkv_bias,
        qkv_out,
        elem_nums,
        head_num,
        seq_len,
        dim_head);
  }
}

template <typename T, typename QKV_TYPE>
void gqa_rotary_qk_variable(T *qkv_out, // [token_num, 3, num_head, dim_head]
                            const QKV_TYPE *qkv_input,  // qkv
                            const float *qkv_out_scales, // [3, num_head, dim_head]
                            const T *qkv_bias,
                            const float *rotary_emb, // [2, 1, 1, seq_len, dim_head / 2]
                            const int *padding_offsets,
                            const int *seq_lens,
                            const int *seq_lens_decoder,
                            const int token_num,
                            const int num_heads,
                            const int kv_num_heads,
                            const int seq_len,
                            const int input_output_len,
                            const int dim_head,
                            const cudaStream_t& stream,
                            bool use_neox_style = false) {
  int elem_nums = token_num * (num_heads + 2 * kv_num_heads) * dim_head; // for all q k v
  if (use_neox_style) {
    elem_nums /= 2;
  }
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (!use_neox_style) {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
    GQAVariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
        qkv_input,
        cos_emb,
        sin_emb,
        padding_offsets,
        seq_lens,
        seq_lens_decoder,
        qkv_out_scales,
        qkv_bias,
        qkv_out,
        elem_nums,
        num_heads,
        seq_len,
        dim_head,
        kv_num_heads);
  } else {
    const float *cos_emb = rotary_emb;
    const float *sin_emb = rotary_emb + input_output_len * dim_head;
    GQANeoxVariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, stream>>>(
        qkv_input,
        cos_emb,
        sin_emb,
        padding_offsets,
        seq_lens,
        seq_lens_decoder,
        qkv_out_scales,
        qkv_bias,
        qkv_out,
        elem_nums,
        num_heads,
        kv_num_heads,
        seq_len,
        dim_head);
  }
}

template <typename T, typename QKV_TYPE>
void EncoderWriteCacheWithRopeKernel(const paddle::Tensor& qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
                                    const paddle::Tensor& rotary_emb,
                                    const paddle::Tensor& seq_lens_this_time,
                                    const paddle::Tensor& seq_lens_encoder,
                                    const paddle::Tensor& seq_lens_decoder,
                                    const paddle::Tensor& padding_offsets,
                                    const paddle::Tensor& cum_offsets,
                                    const paddle::Tensor& block_table,
                                    const paddle::Tensor& batch_ids,
                                    const paddle::Tensor& tile_ids,
                                    const paddle::optional<paddle::Tensor>& qkv_out_scales,
                                    const paddle::optional<paddle::Tensor>& qkv_biases,
                                    const paddle::optional<paddle::Tensor>& cache_k_scale,
                                    const paddle::optional<paddle::Tensor>& cache_v_scale,
                                    const paddle::optional<paddle::Tensor>& cache_k_zp,
                                    const paddle::optional<paddle::Tensor>& cache_v_zp,
                                    const std::string& cache_quant_type_str,
                                    const int num_blocks,
                                    const int max_seq_len,
                                    const int num_heads,
                                    const int kv_num_heads,
                                    const int head_dim,
                                    const bool use_neox_style,
                                    cudaStream_t& stream,
                                    paddle::Tensor *qkv_out, 
                                    paddle::Tensor *key_cache_out, 
                                    paddle::Tensor *value_cache_out) {
  auto qkv_dims = qkv.dims();
  const uint32_t token_num = qkv_dims[0];
#ifdef DEBUG_APPEND
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
  std::cout << "gqa_rotary_qk_variable start" << std::endl;
#endif
  if (num_heads == kv_num_heads) {
    // std::cout << "rotary_qk_variable" << std::endl;
    rotary_qk_variable(
      qkv_out->data<T>(),
      // qkv_out_int32.data<int>(),
      qkv.data<QKV_TYPE>(),
      qkv_out_scales? qkv_out_scales.get().data<float>() : nullptr,
      qkv_biases? qkv_biases.get().data<T>(): nullptr,
      rotary_emb.data<float>(),
      padding_offsets.data<int>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      token_num,
      num_heads,
      max_seq_len,
      rotary_emb.dims()[2],
      head_dim,
      stream,
      use_neox_style
    );
  } else {
    // std::cout << "gqa_rotary_qk_variable" << std::endl;
    gqa_rotary_qk_variable(
      qkv_out->data<T>(),
      // qkv_out_int32.data<int>(),
      qkv.data<QKV_TYPE>(),
      qkv_out_scales? qkv_out_scales.get().data<float>() : nullptr,
      qkv_biases? qkv_biases.get().data<T>(): nullptr,
      rotary_emb.data<float>(),
      padding_offsets.data<int>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      token_num,
      num_heads,
      kv_num_heads,
      max_seq_len,
      rotary_emb.dims()[2],
      head_dim,
      stream,
      use_neox_style
    );
  }
#ifdef DEBUG_APPEND
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
  std::cout << "gqa_rotary_qk_variable end" << std::endl;
#endif
  const auto &cache_k_dims = key_cache_out->dims();
  const uint32_t block_size = cache_k_dims[2];
  if (cache_quant_type_str == "none") {
    CascadeAppendWriteCacheKVQKV<T>(*qkv_out, block_table, padding_offsets, seq_lens_encoder, seq_lens_decoder,
      max_seq_len,  num_heads, head_dim, kv_num_heads, stream, key_cache_out, value_cache_out);
  } else if (cache_quant_type_str == "cache_int8") {
    // std::cout << "CascadeAppendWriteCacheKVC8QKV" << std::endl;
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM,
      {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, 
        {CascadeAppendWriteCacheKVC8QKV<T, HEAD_DIM, BLOCK_SIZE>(
          *key_cache_out, *value_cache_out, *qkv_out, cache_k_scale.get(), cache_v_scale.get(), seq_lens_this_time,
          seq_lens_decoder, padding_offsets, cum_offsets, block_table, batch_ids, tile_ids, num_blocks, max_seq_len, num_heads, kv_num_heads, stream, key_cache_out, value_cache_out);})})
    // std::cout << "CascadeAppendWriteCacheKVC8QKV end" << std::endl;
  } else if (cache_quant_type_str == "cache_int4") {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, 
      {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, 
        {CascadeAppendWriteCacheKVC4QKV<T, HEAD_DIM, BLOCK_SIZE>(
          *key_cache_out, *value_cache_out, *qkv_out, cache_k_scale.get(), cache_v_scale.get(), cache_k_zp.get(), cache_v_zp.get(), seq_lens_this_time,
          seq_lens_decoder, padding_offsets, cum_offsets, block_table, batch_ids, tile_ids, num_blocks, max_seq_len, num_heads, kv_num_heads, stream, key_cache_out, value_cache_out);})})
  } else {
    PD_THROW(
        "NOT supported cache_quant_type. "
        "Only none, cache_int8 and cache_int4 are supported. ");
  }
#ifdef DEBUG_APPEND
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
  std::cout << "CascadeAppendWriteCacheKVQKV end" << std::endl;
#endif
}

template void EncoderWriteCacheWithRopeKernel<paddle::bfloat16, int>(const paddle::Tensor& qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
                                    const paddle::Tensor& rotary_emb,
                                    const paddle::Tensor& seq_lens_this_time,
                                    const paddle::Tensor& seq_lens_encoder,
                                    const paddle::Tensor& seq_lens_decoder,
                                    const paddle::Tensor& padding_offsets,
                                    const paddle::Tensor& cum_offsets,
                                    const paddle::Tensor& block_table,
                                    const paddle::Tensor& batch_ids,
                                    const paddle::Tensor& tile_ids,
                                    const paddle::optional<paddle::Tensor>& qkv_out_scales,
                                    const paddle::optional<paddle::Tensor>& qkv_biases,
                                    const paddle::optional<paddle::Tensor>& cache_k_scale,
                                    const paddle::optional<paddle::Tensor>& cache_v_scale,
                                    const paddle::optional<paddle::Tensor>& cache_k_zp,
                                    const paddle::optional<paddle::Tensor>& cache_v_zp,
                                    const std::string& cache_quant_type_str,
                                    const int num_blocks,
                                    const int max_seq_len,
                                    const int num_heads,
                                    const int kv_num_heads,
                                    const int head_dim,
                                    const bool use_neox_style,
                                    cudaStream_t& stream,
                                    paddle::Tensor *qkv_out, 
                                    paddle::Tensor *key_cache_out, 
                                    paddle::Tensor *value_cache_out);

template void EncoderWriteCacheWithRopeKernel<paddle::bfloat16, paddle::bfloat16>(const paddle::Tensor& qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
                                    const paddle::Tensor& rotary_emb,
                                    const paddle::Tensor& seq_lens_this_time,
                                    const paddle::Tensor& seq_lens_encoder,
                                    const paddle::Tensor& seq_lens_decoder,
                                    const paddle::Tensor& padding_offsets,
                                    const paddle::Tensor& cum_offsets,
                                    const paddle::Tensor& block_table,
                                    const paddle::Tensor& batch_ids,
                                    const paddle::Tensor& tile_ids,
                                    const paddle::optional<paddle::Tensor>& qkv_out_scales,
                                    const paddle::optional<paddle::Tensor>& qkv_biases,
                                    const paddle::optional<paddle::Tensor>& cache_k_scale,
                                    const paddle::optional<paddle::Tensor>& cache_v_scale,
                                    const paddle::optional<paddle::Tensor>& cache_k_zp,
                                    const paddle::optional<paddle::Tensor>& cache_v_zp,
                                    const std::string& cache_quant_type_str,
                                    const int num_blocks,
                                    const int max_seq_len,
                                    const int num_heads,
                                    const int kv_num_heads,
                                    const int head_dim,
                                    const bool use_neox_style,
                                    cudaStream_t& stream,
                                    paddle::Tensor *qkv_out, 
                                    paddle::Tensor *key_cache_out, 
                                    paddle::Tensor *value_cache_out);