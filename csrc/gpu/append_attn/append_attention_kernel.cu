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

#include "append_attention_kernel.h"


// #define DEBUG_ATTN
template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT,
          bool ENABLE_PREFILL = true>
void MultiQueryAppendAttention(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>&
        shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const int num_heads,
    const int kv_num_heads,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;
  // using NV_TYPE = T;
  // using OUT_NV_TYPE = OutT;

  const auto& q_dims = qkv.dims();
  const auto& k_dims = cache_k.dims();
  const auto& cum_offsets_dims = cum_offsets.dims();
  const uint32_t token_num = q_dims[0];
  const uint32_t bsz = cum_offsets_dims[0];
  const uint32_t max_block_num_per_seq = block_table.dims()[1];

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q);  // 1 or 2
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  auto* allocator = paddle::GetAllocator(qkv.place());

  const float scale = 1.f / sqrt(HEAD_DIM);

  if constexpr (NUM_WARP_Q == 4) {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16;  // !!!
    // constexpr uint32_t num_frags_z = 8; // 128 per iter, 4 is better?
    constexpr uint32_t smem_size =
        (num_warps * num_frags_x + NUM_WARP_KV * num_frags_z * 2) * 16 *
        HEAD_DIM * sizeof(T);
    auto split_kv_kernel = multi_query_append_attention_kernel<NV_TYPE,
                                                               true,
                                                               GROUP_SIZE,
                                                               CAUSAL,
                                                               num_warps,
                                                               NUM_WARP_Q,
                                                               NUM_WARP_KV,
                                                               HEAD_DIM,
                                                               BLOCK_SIZE,
                                                               num_frags_x,
                                                               num_frags_z,
                                                               num_frags_y,
                                                               OUT_NV_TYPE,
                                                               ENABLE_PREFILL>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    // int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_kernel start NUM_WARP_Q=" << NUM_WARP_Q << "sm_count:" << sm_count << std::endl;
#endif
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
    
    // assert(act_blocks_per_sm > 1);
    // const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    // const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    // const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    // const float ratio = static_cast<float>(num_blocks_need) /
    //                     static_cast<float>(num_blocks_per_wave);

    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_kernel<NV_TYPE,
                                              false,
                                              GROUP_SIZE,
                                              CAUSAL,
                                              num_warps,
                                              NUM_WARP_Q,
                                              NUM_WARP_KV,
                                              HEAD_DIM,
                                              BLOCK_SIZE,
                                              num_frags_x,
                                              num_frags_z,
                                              num_frags_y,
                                              OUT_NV_TYPE,
                                              ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_kernel start num_chunk = <1 NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_kernel end num_chunk <= 1 NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
    } else {
      phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
      if (ENABLE_PREFILL) {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(token_num * num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(token_num * num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(token_num * num_chunks * num_heads));
      } else {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads));
      }
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_kernel start num_chunk > 1  NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
      split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_kernel end num_chunk > 1  NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                speculate_max_draft_token_num);
      }
    }
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_kernel merge end num_chunk > 1  NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
  } else {
    
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV;  // !!!
    constexpr uint32_t smem_size =
        (num_frags_x + NUM_WARP_KV * num_frags_z * 2) * 16 * HEAD_DIM *
        sizeof(T);
    auto split_kv_kernel =
        multi_query_append_attention_warp1_4_kernel<NV_TYPE,
                                                    true,
                                                    GROUP_SIZE,
                                                    CAUSAL,
                                                    num_warps,
                                                    NUM_WARP_Q,
                                                    NUM_WARP_KV,
                                                    HEAD_DIM,
                                                    BLOCK_SIZE,
                                                    num_frags_x,
                                                    num_frags_z,
                                                    num_frags_y,
                                                    OUT_NV_TYPE,
                                                    ENABLE_PREFILL>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    // int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_warp1_4_kernel start NUM_WARP_Q=" << NUM_WARP_Q << "sm_count:" << sm_count << std::endl;
#endif

    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
    // assert(act_blocks_per_sm > 1);
    // const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    // const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    // const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    // const float ratio = static_cast<float>(num_blocks_need) /
    //                     static_cast<float>(num_blocks_per_wave);

    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);

    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    // dim3 grids(num_blocks_x_cpu, num_chunks, 1);
    dim3 blocks(32, num_warps);

    if (num_chunks <= 1) {
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_warp1_4_kernel start num_chunk <= 1  NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
      auto nosplit_kv_kernel =
          multi_query_append_attention_warp1_4_kernel<NV_TYPE,
                                                      false,
                                                      GROUP_SIZE,
                                                      CAUSAL,
                                                      num_warps,
                                                      NUM_WARP_Q,
                                                      NUM_WARP_KV,
                                                      HEAD_DIM,
                                                      BLOCK_SIZE,
                                                      num_frags_x,
                                                      num_frags_z,
                                                      num_frags_y,
                                                      OUT_NV_TYPE,
                                                      ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }

      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_warp1_4_kernel end num_chunk <= 1  NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
    } else {
      phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
      if (is_decoder) {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(bsz * num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(bsz * num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(bsz * num_chunks * num_heads));
      } else {
        if (ENABLE_PREFILL) {
          tmp_workspace =
              allocator->Allocate(phi::SizeOf(qkv.dtype()) *
                                  static_cast<size_t>(token_num * num_chunks *
                                                      num_heads * HEAD_DIM));
          tmp_m = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(token_num * num_chunks * num_heads));
          tmp_d = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(token_num * num_chunks * num_heads));
        } else {
          tmp_workspace = allocator->Allocate(
              phi::SizeOf(qkv.dtype()) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads * HEAD_DIM));
          tmp_m = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads));
          tmp_d = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads));
        }
      }
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_warp1_4_kernel start num_chunk > 1  NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
      split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_warp1_4_kernel end num_chunk > 1  NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                speculate_max_draft_token_num);
      }
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_warp1_4_kernel merge end num_chunk > 1  NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
    }
  }
}


template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
void MultiQueryAppendC8Attention(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::Tensor& cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::Tensor& cache_v_scale,
    const paddle::optional<paddle::Tensor>&
        shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const int num_heads,
    const int kv_num_heads,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t& stream,
    paddle::Tensor* out) {
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "MultiQueryAppendC8Attention start NUM_WARP_Q=" << NUM_WARP_Q << std::endl;
#endif
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;
  // using NV_TYPE = T;
  // using OUT_NV_TYPE = OutT;
  const auto& q_dims = qkv.dims();
  const auto& k_dims = cache_k.dims();
  const auto& cum_offsets_dims = cum_offsets.dims();
  const uint32_t token_num = q_dims[0];
  const uint32_t bsz = cum_offsets_dims[0];
  const uint32_t max_block_num_per_seq = block_table.dims()[1];

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q);  // 1 or 2
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  auto* allocator = paddle::GetAllocator(qkv.place());

  const float scale = 1.f / sqrt(HEAD_DIM);

  if constexpr (NUM_WARP_Q == 4) {
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_c8_kernel start NUM_WARP_Q=4" << std::endl;
#endif
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16;  // !!!
    constexpr uint32_t smem_size =
        num_warps * num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        num_frags_z * 16 * HEAD_DIM * sizeof(uint8_t) * 2;
    auto split_kv_kernel =
        multi_query_append_attention_c8_kernel<NV_TYPE,
                                               uint8_t,
                                               true,
                                               GROUP_SIZE,
                                               CAUSAL,
                                               num_warps,
                                               NUM_WARP_Q,
                                               NUM_WARP_KV,
                                               HEAD_DIM,
                                               BLOCK_SIZE,
                                               num_frags_x,
                                               num_frags_z,
                                               num_frags_y,
                                               OUT_NV_TYPE,
                                               ENABLE_PREFILL>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    // int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_c8_kernel launched "
        << " smem_size: " << smem_size << std::endl;
#endif
    // assert(act_blocks_per_sm > 1);
    // const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    // const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    // const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    // const float ratio = static_cast<float>(num_blocks_need) /
    //                     static_cast<float>(num_blocks_per_wave);

    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_c8_kernel num_chunks<= 1" << std::endl;
#endif
      auto nosplit_kv_kernel =
          multi_query_append_attention_c8_kernel<NV_TYPE,
                                                 uint8_t,
                                                 false,
                                                 GROUP_SIZE,
                                                 CAUSAL,
                                                 num_warps,
                                                 NUM_WARP_Q,
                                                 NUM_WARP_KV,
                                                 HEAD_DIM,
                                                 BLOCK_SIZE,
                                                 num_frags_x,
                                                 num_frags_z,
                                                 num_frags_y,
                                                 OUT_NV_TYPE,
                                                 ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }


      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_c8_kernel num_chunks <=1 end" << std::endl;
#endif
    } else {
      phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
      if (ENABLE_PREFILL) {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(token_num * num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(token_num * num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(token_num * num_chunks * num_heads));
      } else {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads));
      }
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_c8_kernel num_chunks > 1" << std::endl;
#endif
      split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_c8_kernel num_chunks > 1 end" << std::endl;
#endif
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                speculate_max_draft_token_num);
      }
    }
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_c8_kernel NUM_WARP_Q=4 end" << std::endl;
#endif
  } else {
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_c8_warp1_4_kernel start NUM_WARP_Q=1" << std::endl;
#endif
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV * 2;  // !!!
    constexpr uint32_t smem_size =
        num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM * sizeof(uint8_t) * 2;
    auto split_kv_kernel =
        multi_query_append_attention_c8_warp1_4_kernel<NV_TYPE,
                                                       uint8_t,
                                                       true,
                                                       GROUP_SIZE,
                                                       CAUSAL,
                                                       num_warps,
                                                       NUM_WARP_Q,
                                                       NUM_WARP_KV,
                                                       HEAD_DIM,
                                                       BLOCK_SIZE,
                                                       num_frags_x,
                                                       num_frags_z,
                                                       num_frags_y,
                                                       OUT_NV_TYPE>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    // int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "multi_query_append_attention_c8_warp1_4_kernel launched "
        << " smem_size: " << smem_size << std::endl;
#endif
    // assert(act_blocks_per_sm > 1);
    // const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    // const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    // const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    // const float ratio = static_cast<float>(num_blocks_need) /
    //                     static_cast<float>(num_blocks_per_wave);

    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }

    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    // dim3 grids(num_blocks_x_cpu, num_chunks, 1);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_c8_warp1_4_kernel<NV_TYPE,
                                                         uint8_t,
                                                         false,
                                                         GROUP_SIZE,
                                                         CAUSAL,
                                                         num_warps,
                                                         NUM_WARP_Q,
                                                         NUM_WARP_KV,
                                                         HEAD_DIM,
                                                         BLOCK_SIZE,
                                                         num_frags_x,
                                                         num_frags_z,
                                                         num_frags_y,
                                                         OUT_NV_TYPE,
                                                         ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }

      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
#ifdef DEBUG_ATTN
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      std::cout << "multi_query_append_attention_c8_warp1_4_kernel launched act_blocks_per_sm:" << std::endl;
#endif
    } else {
      phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
      if (is_decoder) {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(bsz * num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(bsz * num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(bsz * num_chunks * num_heads));
      } else {
        if (ENABLE_PREFILL) {
          tmp_workspace =
              allocator->Allocate(phi::SizeOf(qkv.dtype()) *
                                  static_cast<size_t>(token_num * num_chunks *
                                                      num_heads * HEAD_DIM));
          tmp_m = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(token_num * num_chunks * num_heads));
          tmp_d = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(token_num * num_chunks * num_heads));
        } else {
          tmp_workspace = allocator->Allocate(
              phi::SizeOf(qkv.dtype()) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads * HEAD_DIM));
          tmp_m = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads));
          tmp_d = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads));
        }
      }
      split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE, vec_size, blocky, HEAD_DIM>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                speculate_max_draft_token_num);
      }
    }
  }
}

template <typename T,
          uint32_t GROUP_SIZE,
          uint32_t HEAD_DIM,
          uint32_t BLOCK_SIZE,
          bool CAUSAL,
          uint32_t BLOCK_SHAPE_Q,
          uint32_t NUM_WARP_Q,
          typename OutT = T,
          bool ENABLE_PREFILL = true>
void MultiQueryAppendC4Attention(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::Tensor& cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::Tensor& cache_v_scale,
    const paddle::optional<paddle::Tensor>&
        cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>&
        shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const int num_blocks_x_cpu,
    const int max_seq_len,
    const int max_dec_len,
    const int num_heads,
    const int kv_num_heads,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool is_decoder,
    cudaStream_t& stream,
    paddle::Tensor* out) {
  using NV_TYPE = typename cascade_attn_type_traits<T>::type;
  using OUT_NV_TYPE = typename cascade_attn_type_traits<OutT>::type;
  // using NV_TYPE = T;
  // using OUT_NV_TYPE = OutT;

  const auto& q_dims = qkv.dims();
  const auto& k_dims = cache_k.dims();
  const auto& cum_offsets_dims = cum_offsets.dims();
  const uint32_t token_num = q_dims[0];
  const uint32_t bsz = cum_offsets_dims[0];
  const uint32_t max_block_num_per_seq = block_table.dims()[1];
  constexpr uint32_t num_warps = 4;
  constexpr uint32_t NUM_WARP_KV = num_warps / NUM_WARP_Q;
  constexpr uint32_t num_frags_x = BLOCK_SHAPE_Q / (16 * NUM_WARP_Q);  // 1 or 2
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  constexpr uint32_t num_qrow_per_block = NUM_WARP_Q * num_frags_x * 16;

  auto* allocator = paddle::GetAllocator(qkv.place());

  const float scale = 1.f / sqrt(HEAD_DIM);

  if constexpr (NUM_WARP_Q == 4) {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16;  // !!!
    // constexpr uint32_t num_frags_z = 8; // 128 per iter, 4 is better?
    constexpr uint32_t smem_size =
        num_warps * num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        num_frags_z * 16 * HEAD_DIM / 2 * sizeof(uint8_t) * 2 +
        HEAD_DIM * 4 * sizeof(T);
    auto split_kv_kernel =
        multi_query_append_attention_c4_kernel<NV_TYPE,
                                               uint8_t,
                                               true,
                                               GROUP_SIZE,
                                               CAUSAL,
                                               num_warps,
                                               NUM_WARP_Q,
                                               NUM_WARP_KV,
                                               HEAD_DIM,
                                               BLOCK_SIZE,
                                               num_frags_x,
                                               num_frags_z,
                                               num_frags_y,
                                               OUT_NV_TYPE,
                                               ENABLE_PREFILL>;
    // if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(split_kv_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    // }
    const int dev_id = 0;
    int sm_count;
    int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
    assert(act_blocks_per_sm > 1);
    const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    const float ratio = static_cast<float>(num_blocks_need) /
                        static_cast<float>(num_blocks_per_wave);

    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);

    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_c4_kernel<NV_TYPE,
                                                 uint8_t,
                                                 false,
                                                 GROUP_SIZE,
                                                 CAUSAL,
                                                 num_warps,
                                                 NUM_WARP_Q,
                                                 NUM_WARP_KV,
                                                 HEAD_DIM,
                                                 BLOCK_SIZE,
                                                 num_frags_x,
                                                 num_frags_z,
                                                 num_frags_y,
                                                 OUT_NV_TYPE,
                                                 ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }
      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_k_zp.get().data<T>()))
                     : nullptr,
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_v_zp.get().data<T>()))
                     : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
    } else {
      phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
      if (ENABLE_PREFILL) {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(token_num * num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(token_num * num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(token_num * num_chunks * num_heads));
      } else {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                num_chunks * num_heads));
      }
      split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_k_zp.get().data<T>()))
                     : nullptr,
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_v_zp.get().data<T>()))
                     : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                speculate_max_draft_token_num);
      }
    }
  } else {
    constexpr uint32_t num_frags_z = BLOCK_SIZE / 16 / NUM_WARP_KV * 4;  // !!!
    constexpr uint32_t smem_size =
        num_frags_x * 16 * HEAD_DIM * sizeof(T) +
        NUM_WARP_KV * num_frags_z * 16 * HEAD_DIM / 2 * sizeof(uint8_t) * 2 +
        HEAD_DIM * 4 * sizeof(T);
    auto split_kv_kernel =
        multi_query_append_attention_c4_warp1_4_kernel<NV_TYPE,
                                                       uint8_t,
                                                       true,
                                                       GROUP_SIZE,
                                                       CAUSAL,
                                                       num_warps,
                                                       NUM_WARP_Q,
                                                       NUM_WARP_KV,
                                                       HEAD_DIM,
                                                       BLOCK_SIZE,
                                                       num_frags_x,
                                                       num_frags_z,
                                                       num_frags_y,
                                                       OUT_NV_TYPE,
                                                       ENABLE_PREFILL>;
    if (smem_size >= 48 * 1024) {
      cudaFuncSetAttribute(split_kv_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size);
    }
    const int dev_id = 0;
    int sm_count;
    int act_blocks_per_sm;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &act_blocks_per_sm, split_kv_kernel, num_warps * 32, smem_size);
    assert(act_blocks_per_sm > 1);
    const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
    const int num_blocks_need = num_blocks_x_cpu * kv_num_heads;
    const int max_num_chunks = div_up(num_blocks_per_wave, num_blocks_need);
    const float ratio = static_cast<float>(num_blocks_need) /
                        static_cast<float>(num_blocks_per_wave);


    uint32_t chunk_size = static_cast<uint32_t>(max_partition_size);
    if (!is_decoder) {
      chunk_size = static_cast<uint32_t>(encoder_max_partition_size);
    }
    const int num_chunks = div_up(max_dec_len, chunk_size);
    dim3 grids(num_blocks_x_cpu, num_chunks, kv_num_heads);
    // dim3 grids(num_blocks_x_cpu, num_chunks, 1);
    dim3 blocks(32, num_warps);
    if (num_chunks <= 1) {
      auto nosplit_kv_kernel =
          multi_query_append_attention_c4_warp1_4_kernel<NV_TYPE,
                                                         uint8_t,
                                                         false,
                                                         GROUP_SIZE,
                                                         CAUSAL,
                                                         num_warps,
                                                         NUM_WARP_Q,
                                                         NUM_WARP_KV,
                                                         HEAD_DIM,
                                                         BLOCK_SIZE,
                                                         num_frags_x,
                                                         num_frags_z,
                                                         num_frags_y,
                                                         OUT_NV_TYPE,
                                                         ENABLE_PREFILL>;
      if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(nosplit_kv_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
      }
      nosplit_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_k_zp.get().data<T>()))
                     : nullptr,
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_v_zp.get().data<T>()))
                     : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          nullptr,
          nullptr,
          nullptr,
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
    } else {
      phi::Allocator::AllocationPtr tmp_workspace, tmp_m, tmp_d;
      if (is_decoder) {
        tmp_workspace = allocator->Allocate(
            phi::SizeOf(qkv.dtype()) *
            static_cast<size_t>(bsz * num_chunks * num_heads * HEAD_DIM));
        tmp_m = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(bsz * num_chunks * num_heads));
        tmp_d = allocator->Allocate(
            phi::SizeOf(paddle::DataType::FLOAT32) *
            static_cast<size_t>(bsz * num_chunks * num_heads));
      } else {
        if (ENABLE_PREFILL) {
          tmp_workspace =
              allocator->Allocate(phi::SizeOf(qkv.dtype()) *
                                  static_cast<size_t>(token_num * num_chunks *
                                                      num_heads * HEAD_DIM));
          tmp_m = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(token_num * num_chunks * num_heads));
          tmp_d = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(token_num * num_chunks * num_heads));
        } else {
          tmp_workspace = allocator->Allocate(
              phi::SizeOf(qkv.dtype()) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads * HEAD_DIM));
          tmp_m = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads));
          tmp_d = allocator->Allocate(
              phi::SizeOf(paddle::DataType::FLOAT32) *
              static_cast<size_t>(speculate_max_draft_token_num * bsz *
                                  num_chunks * num_heads));
        }
      }
      split_kv_kernel<<<grids, blocks, smem_size, stream>>>(
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(qkv.data<T>())),
          const_cast<uint8_t*>(cache_k.data<uint8_t>()),
          const_cast<uint8_t*>(cache_v.data<uint8_t>()),
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_k_scale.data<T>())),
          cache_k_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_k_zp.get().data<T>()))
                     : nullptr,
          reinterpret_cast<NV_TYPE*>(const_cast<T*>(cache_v_scale.data<T>())),
          cache_v_zp ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(cache_v_zp.get().data<T>()))
                     : nullptr,
          shift_bias ? reinterpret_cast<NV_TYPE*>(
                           const_cast<T*>(shift_bias.get().data<T>()))
                     : nullptr,
          smooth_weight ? reinterpret_cast<NV_TYPE*>(
                              const_cast<T*>(smooth_weight.get().data<T>()))
                        : nullptr,
          seq_lens_q.data<int>(),
          seq_lens_kv.data<int>(),
          batch_ids.data<int>(),
          tile_ids_per_batch.data<int>(),
          cum_offsets.data<int>(),
          block_table.data<int>(),
          max_seq_len,
          max_dec_len,
          max_block_num_per_seq,
          scale,
          in_scale,
          chunk_size,
          reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
          static_cast<float*>(tmp_m->ptr()),
          static_cast<float*>(tmp_d->ptr()),
          reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
          speculate_max_draft_token_num);
      // merge
      constexpr int vec_size = num_elems_per_128b<NV_TYPE>();
      if (is_decoder) {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(bsz, num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_decoder_kernel<NV_TYPE,
                                          vec_size,
                                          blocky,
                                          HEAD_DIM,
                                          OUT_NV_TYPE,
                                          ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                cum_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM);
      } else {
        constexpr int blockx = HEAD_DIM / vec_size;
        constexpr int blocky = (128 + blockx - 1) / blockx;
        dim3 grids_merge(min(sm_count * 4, token_num),
                         num_heads);  // 128k is too large
        dim3 blocks_merge(blockx, blocky);
        merge_multi_chunks_v2_kernel<NV_TYPE,
                                     vec_size,
                                     blocky,
                                     HEAD_DIM,
                                     OUT_NV_TYPE,
                                     ENABLE_PREFILL>
            <<<grids_merge, blocks_merge, 0, stream>>>(
                reinterpret_cast<NV_TYPE*>(tmp_workspace->ptr()),
                static_cast<float*>(tmp_m->ptr()),
                static_cast<float*>(tmp_d->ptr()),
                seq_lens_q.data<int>(),
                seq_lens_kv.data<int>(),
                seq_lens_encoder.data<int>(),
                padding_offsets.data<int>(),
                shift_bias ? reinterpret_cast<NV_TYPE*>(
                                 const_cast<T*>(shift_bias.get().data<T>()))
                           : nullptr,
                smooth_weight ? reinterpret_cast<NV_TYPE*>(const_cast<T*>(
                                    smooth_weight.get().data<T>()))
                              : nullptr,
                reinterpret_cast<OUT_NV_TYPE*>(out->data<OutT>()),
                in_scale,
                max_seq_len,
                num_chunks,
                num_heads,
                chunk_size,
                HEAD_DIM,
                token_num,
                speculate_max_draft_token_num);
      }
    }
  }
}

template <typename T, typename OutT>
void CascadeAppendAttentionKernel(
    const paddle::Tensor& qkv,  // [token_num, num_heads, head_dim]
    const paddle::Tensor&
        cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const paddle::Tensor&
        cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>&
        cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const std::string& cache_quant_type_str,
    const int num_blocks,
    const int block_shape_q,
    const int max_seq_len,
    const int max_dec_len,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out) {
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "CascadeAppendAttentionKernel start max_dec_len=" << max_dec_len << 
       "cache_quant_type_str:" << cache_quant_type_str << std::endl;
#endif
//   if (max_dec_len <= 0) {
//     return;
//   }

  const auto& q_dims = qkv.dims();
  const auto& k_dims = cache_k.dims();
  const auto& cum_offsets_dims = cum_offsets.dims();
  const uint32_t token_num = q_dims[0];
  const uint32_t block_size = k_dims[2];
  const uint32_t bsz = cum_offsets_dims[0];
  const uint32_t group_size = num_heads / kv_num_heads;

  if (cache_quant_type_str == "none") {
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "MultiQueryAppendAttentionC16 start" << std::endl;
#endif
    DISPATCH_CAUSAL(causal, CAUSAL,
        {DISPATCH_ENABLE_PREFILL(enable_prefill, ENABLE_PREFILL,
            {DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE,
                {DISPATCH_HEAD_DIM(head_dim, HEAD_DIM,
                    {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE,
                        {DISPATCH_BLOCKSHAPE_Q(block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, 
                        {MultiQueryAppendAttention<T, GROUP_SIZE, HEAD_DIM, BLOCK_SIZE, CAUSAL, BLOCK_SHAPE_Q, NUM_WARP_Q, OutT, ENABLE_PREFILL>(
                            qkv,
                            cache_k,
                            cache_v,
                            attn_mask,
                            shift_bias,
                            smooth_weight,
                            seq_lens_q,
                            seq_lens_kv,
                            seq_lens_encoder,
                            padding_offsets,
                            cum_offsets,
                            block_table,
                            batch_ids,
                            tile_ids_per_batch,
                            num_blocks,
                            max_seq_len,
                            max_dec_len,
                            num_heads,
                            kv_num_heads,
                            in_scale,
                            max_partition_size,
                            encoder_max_partition_size,
                            speculate_max_draft_token_num,
                            is_decoder,
                            stream,
                            out);
                            })})})})})})
  } else if (cache_quant_type_str == "cache_int8") {
#ifdef DEBUG_ATTN
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    std::cout << "MultiQueryAppendAttentionC8 start" << std::endl;
#endif
    DISPATCH_CAUSAL(causal, CAUSAL,
        {DISPATCH_ENABLE_PREFILL(enable_prefill, ENABLE_PREFILL,
            {DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE,
                {DISPATCH_HEAD_DIM(head_dim, HEAD_DIM,
                    {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE,
                        {DISPATCH_BLOCKSHAPE_Q(block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q, {
                            MultiQueryAppendC8Attention<T, GROUP_SIZE, HEAD_DIM, BLOCK_SIZE, CAUSAL, BLOCK_SHAPE_Q, NUM_WARP_Q, OutT, ENABLE_PREFILL>(
                                qkv,
                                cache_k,
                                cache_v,
                                attn_mask,
                                cache_k_scale.get(),
                                cache_v_scale.get(),
                                shift_bias,
                                smooth_weight,
                                seq_lens_q,
                                seq_lens_kv,
                                seq_lens_encoder,
                                padding_offsets,
                                cum_offsets,
                                block_table,
                                batch_ids,
                                tile_ids_per_batch,
                                num_blocks,
                                max_seq_len,
                                max_dec_len,
                                num_heads,
                                kv_num_heads,
                                in_scale,
                                max_partition_size,
                                encoder_max_partition_size,
                                speculate_max_draft_token_num,
                                is_decoder,
                                stream,
                                out);
                            })})})})})})
  } else if (cache_quant_type_str == "cache_int4") {
    DISPATCH_CAUSAL(causal, CAUSAL,
        {DISPATCH_ENABLE_PREFILL(enable_prefill, ENABLE_PREFILL,
            {DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE,
                {DISPATCH_HEAD_DIM(head_dim, HEAD_DIM,
                    {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE,
                        {DISPATCH_BLOCKSHAPE_Q(block_shape_q, BLOCK_SHAPE_Q, NUM_WARP_Q,
                            {MultiQueryAppendC4Attention<T, GROUP_SIZE, HEAD_DIM, BLOCK_SIZE, CAUSAL, BLOCK_SHAPE_Q, NUM_WARP_Q, OutT, ENABLE_PREFILL>(
                                qkv,
                                cache_k,
                                cache_v,
                                attn_mask,
                                cache_k_scale.get(),
                                cache_v_scale.get(),
                                cache_k_zp,
                                cache_v_zp,
                                shift_bias,
                                smooth_weight,
                                seq_lens_q,
                                seq_lens_kv,
                                seq_lens_encoder,
                                padding_offsets,
                                cum_offsets,
                                block_table,
                                batch_ids,
                                tile_ids_per_batch,
                                num_blocks,
                                max_seq_len,
                                max_dec_len,
                                num_heads,
                                kv_num_heads,
                                in_scale,
                                max_partition_size,
                                encoder_max_partition_size,
                                speculate_max_draft_token_num,
                                is_decoder,
                                stream,
                                out);
                            })})})})})})
  } else {
    PD_THROW("append attention just support C16/C8/C4_zp now!");
  }
}


template void CascadeAppendAttentionKernel<paddle::bfloat16, int8_t>(
    const paddle::Tensor& qkv,  // [token_num, num_heads, head_dim]
    const paddle::Tensor&
        cache_k,  // [max_block_num, num_heads, block_size, head_dim]
    const paddle::Tensor&
        cache_v,  // [max_block_num, num_heads, head_dim, block_size]
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>&
        cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_q,
    const paddle::Tensor& seq_lens_kv,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& block_table,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const std::string& cache_quant_type_str,
    const int num_blocks,
    const int block_shape_q,
    const int max_seq_len,
    const int max_dec_len,
    const int num_heads,
    const int kv_num_heads,
    const int head_dim,
    const float in_scale,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool is_decoder,
    const bool enable_prefill,
    cudaStream_t& stream,
    paddle::Tensor* out);
// template void CascadeAppendAttentionKernel<paddle::bfloat16, paddle::bfloat16>(
//     const paddle::Tensor& qkv,  // [token_num, num_heads, head_dim]
//     const paddle::Tensor&
//         cache_k,  // [max_block_num, num_heads, block_size, head_dim]
//     const paddle::Tensor&
//         cache_v,  // [max_block_num, num_heads, head_dim, block_size]
//     const paddle::optional<paddle::Tensor>& attn_mask,
//     const paddle::optional<paddle::Tensor>&
//         cache_k_scale,  // [num_kv_heads, head_dim]
//     const paddle::optional<paddle::Tensor>&
//         cache_v_scale,  // [num_kv_heads, head_dim]
//     const paddle::optional<paddle::Tensor>&
//         cache_k_zp,  // [num_kv_heads, head_dim]
//     const paddle::optional<paddle::Tensor>&
//         cache_v_zp,  // [num_kv_heads, head_dim]
//     const paddle::optional<paddle::Tensor>&
//         shift_bias,  // [num_kv_heads, head_dim]
//     const paddle::optional<paddle::Tensor>&
//         smooth_weight,  // [num_kv_heads, head_dim]
//     const paddle::Tensor& seq_lens_q,
//     const paddle::Tensor& seq_lens_kv,
//     const paddle::Tensor& seq_lens_encoder,
//     const paddle::Tensor& padding_offsets,
//     const paddle::Tensor& cum_offsets,
//     const paddle::Tensor& block_table,
//     const paddle::Tensor& batch_ids,
//     const paddle::Tensor& tile_ids_per_batch,
//     const std::string& cache_quant_type_str,
//     const int num_blocks,
//     const int block_shape_q,
//     const int max_seq_len,
//     const int max_dec_len,
//     const int num_heads,
//     const int kv_num_heads,
//     const int head_dim,
//     const float in_scale,
//     const int max_partition_size,
//     const int encoder_max_partition_size,
//     const int speculate_max_draft_token_num,
//     const bool causal,
//     const bool is_decoder,
//     const bool enable_prefill,
//     cudaStream_t& stream,
//     paddle::Tensor* out);