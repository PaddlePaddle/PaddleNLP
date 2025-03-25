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

/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */

#pragma once

#ifndef ATTENTION_HOPPER_PREFILL_SM90_CUH_
#define ATTENTION_HOPPER_PREFILL_SM90_CUH_

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <type_traits>
#include <vector>

#include "attention_updater.cuh"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "epilogue.cuh"
#include "helper.h"
#include "kernel_traits.cuh"
#include "mainloop_mma.cuh"
#include "mainloop_load.cuh"
#include "utils.cuh"

#ifdef DEBUG_MLA
#undef DEBUG_MLA
#endif
// #define DEBUG_MLA

namespace mla_attn {

using namespace cute;

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct Params {
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;

    alignas(16) DTypeQ *Q; // [token_num, head_num, dim_head]
    alignas(16) DTypeKV *KV; // [max_block_num, block_size, dim_head]
    alignas(16) DTypeO *O; // [token_num, head_num, dim_head]
    alignas(16) DTypeO *O_tmp; // [num_chunks, bsz, head_num, dim_head]
    alignas(16) float *m; // [num_chunks, bsz * draft_total_token_num * head_num]
    alignas(16) float *d; // [num_chunks, bsz * draft_total_token_num * head_num]

    alignas(16) IdType *block_tables;
    alignas(16) IdType *seq_lens_this_time;
    alignas(16) IdType *seq_lens_encoder;
    alignas(16) IdType *seq_lens_decoder;
    alignas(16) IdType *cumsum_q_seqlens;
    alignas(16) IdType *padding_offsets;

    alignas(16) IdType *batch_ids;
    alignas(16) IdType *q_tile_ids_per_batch = nullptr;
    alignas(16) IdType *tile_ids_per_batch = nullptr;
    alignas(16) IdType *num_blocks_x;


    uint32_t q_stride_bsz;
    uint32_t q_stride_head_num;

    uint32_t kv_stride_block_num;
    uint32_t kv_stride_block_size;

    uint32_t o_stride_bsz;
    uint32_t o_stride_head_num;

    int bsz;
    int token_num;
    int max_seq_len;
    int max_block_num;
    int max_block_num_per_seq;
    int q_num_head;
    int qk_head_dim;
    int vo_head_dim;
    int block_size;
    int draft_total_token_num;
    int chunk_size;
    int chunk_num;
    int num_blocks_x_int;

    float sm_scale;
};

#define DISPATCH_GROUP_SIZE(group_size, GROUP_SIZE, ...)     \
  if (group_size == 8) {                                     \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 16) {                             \
    constexpr size_t GROUP_SIZE = 16;                        \
    __VA_ARGS__                                              \
  } else if (group_size == 64) {                             \
    constexpr size_t GROUP_SIZE = 64;                        \
    __VA_ARGS__                                              \
  } else if (group_size == 128) {                            \
    constexpr size_t GROUP_SIZE = 128;                       \
    __VA_ARGS__                                              \
  } else {                                                   \
    PD_THROW("not support the group_size: ", group_size);    \
    return cudaErrorNotSupported;                            \
  }
template <typename CollectiveMainloop, typename CollectiveEpilogue, typename Ktraits, bool CAUSAL, bool USE_REG_EALLOC = false>
__global__ void __launch_bounds__(Ktraits::NUM_WARPS * cutlass::NumThreadsPerWarp, 1)
MLAWithKVCacheKernel(CUTE_GRID_CONSTANT
                     typename CollectiveMainloop::Params const mainloop_params,
                     CUTE_GRID_CONSTANT
                     typename CollectiveEpilogue::Params const epilogue_params) {

  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;

  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_COPY_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  static constexpr int BLOCK_SHAPE_Q = Ktraits::BLOCK_SHAPE_Q;
  static constexpr int BLOCK_SHAPE_KV = Ktraits::BLOCK_SHAPE_KV;
  const int num_blocks_x = mainloop_params.num_blocks_x[0];

  static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  using MainloopPipelineQ = typename CollectiveMainloop::MainloopPipelineQ;
  using PipelineParamsQ = typename MainloopPipelineQ::Params;
  using PipelineStateQ = typename MainloopPipelineQ::PipelineState;

  extern __shared__ char shared_memory[];
  auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
  }

  // Obtain warp index
  int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  if constexpr (use_tma_load_kv) {
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NUM_MMA_THREADS;
  } else {
    pipeline_params.producer_arv_count = NUM_COPY_THREADS;
    pipeline_params.consumer_arv_count = NUM_MMA_THREADS;
  }

  PipelineParamsQ pipeline_params_q;
  pipeline_params_q.role = warp_group_idx == 0 ? MainloopPipelineQ::ThreadCategory::Producer
                                               : MainloopPipelineQ::ThreadCategory::Consumer;
  pipeline_params_q.producer_arv_count = NUM_COPY_THREADS;
  pipeline_params_q.consumer_arv_count = cutlass::NumThreadsPerWarpGroup; // just one wg qk
  

  MainloopPipelineQ pipeline_q(shared_storage.pipeline_q, pipeline_params_q);
  MainloopPipeline pipeline_kv = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesKV;
      return MainloopPipeline(shared_storage.pipeline_kv, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_kv, pipeline_params);
    }
  }();
  __syncthreads();

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;
  
  if (warp_group_idx == 0) {
    // producer
    if (USE_REG_EALLOC) {
      cutlass::arch::warpgroup_reg_dealloc<88>();
    }
    const uint32_t warp_idx_in_warpgroup = __shfl_sync(0xffffffff, warp_idx % 4, 0);
    
    PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
    PipelineState smem_pipe_write_kv = cutlass::make_producer_start_state<MainloopPipeline>();
    const int block_id = blockIdx.x;
    const int bid = mainloop_params.batch_ids[block_id];
    const int tile_id = mainloop_params.tile_ids_per_batch ? mainloop_params.tile_ids_per_batch[block_id] : 0;
    const int q_tile_id = mainloop_params.q_tile_ids_per_batch ? mainloop_params.q_tile_ids_per_batch[block_id] : 0;
    const int seq_len_now = mainloop_params.seq_lens_this_time[bid];
    const int seq_len_encoder_now = mainloop_params.seq_lens_encoder[bid];
    const int seq_len_decoder_now = mainloop_params.seq_lens_decoder[bid] + seq_len_now;
    const int start_token_idx = mainloop_params.cumsum_q_seqlens[bid];

    // load Q
    collective_mainloop.load_q(
        mainloop_params,
        pipeline_q,
        smem_pipe_write_q,
        shared_storage,
        threadIdx.x,
        bid,
        q_tile_id);

    if constexpr (!use_tma_load_kv) {
      // load kv
      collective_mainloop.load_kv(
          mainloop_params,
          pipeline_kv,
          smem_pipe_write_kv,
          shared_storage,
          bid,
          seq_len_decoder_now,
          tile_id
      );
    } else {
      if (warp_idx_in_warpgroup == 0) {
        // load kv tma
        collective_mainloop.load_kv_tma(
            mainloop_params,
            pipeline_kv,
            smem_pipe_write_kv,
            shared_storage,
            bid,
            seq_len_decoder_now,
            tile_id
        );
      }
    }
  } else {
    // consumer
    if (USE_REG_EALLOC) {
      cutlass::arch::warpgroup_reg_alloc<208>();
    }
    PipelineStateQ smem_pipe_read_q;
    PipelineState smem_pipe_read_kv;

    typename Ktraits::TiledMmaPVSS tiled_mma_pv;
    Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));

    auto attention_updater = OnlineSoftmax<2 * size<1>(tOrO), /*WITH_SCALE=*/true>(mainloop_params.sm_scale);
    const int block_id = blockIdx.x;
    clear(tOrO);
    clear(attention_updater.scores_scale);
    const int bid = mainloop_params.batch_ids[block_id];
    const int tile_id = mainloop_params.tile_ids_per_batch ? mainloop_params.tile_ids_per_batch[block_id] : 0;
    const int q_tile_id = mainloop_params.q_tile_ids_per_batch ? mainloop_params.q_tile_ids_per_batch[block_id] : 0;
    const int seq_len_now = mainloop_params.seq_lens_this_time[bid];
    const int seq_len_encoder_now = mainloop_params.seq_lens_encoder[bid];
    const int seq_len_decoder_now = mainloop_params.seq_lens_decoder[bid] + seq_len_now;
    const int start_token_idx = mainloop_params.cumsum_q_seqlens[bid];

    if constexpr (BLOCK_SHAPE_KV == 64) {
      mma_f16<Ktraits, CAUSAL>(
        mainloop_params, 
        pipeline_q, 
        smem_pipe_read_q,
        pipeline_kv, 
        smem_pipe_read_kv,
        tOrO, 
        attention_updater,
        threadIdx.x - NUM_COPY_THREADS,
        bid,
        seq_len_decoder_now,
        seq_len_now,
        tile_id,
        q_tile_id,
        shared_storage);
    } else if (BLOCK_SHAPE_KV == 32) {
      mma_f16_two_stages<Ktraits, CAUSAL>(
        mainloop_params, 
        pipeline_q, 
        smem_pipe_read_q,
        pipeline_kv, 
        smem_pipe_read_kv,
        tOrO, 
        attention_updater,
        threadIdx.x - NUM_COPY_THREADS,
        bid,
        seq_len_decoder_now,
        seq_len_now,
        tile_id,
        q_tile_id,
        shared_storage);
    }

    collective_epilogue.store(
        epilogue_params, 
        tOrO, 
        attention_updater.get_lse(),
        shared_storage,
        tiled_mma_pv, 
        threadIdx.x - NUM_COPY_THREADS,
        bid,
        mainloop_params.bsz,
        seq_len_now,
        start_token_idx,
        tile_id,
        q_tile_id,
        seq_len_decoder_now,
        mainloop_params.chunk_size,
        mainloop_params.draft_total_token_num,
        mainloop_params.o_stride_head_num);
  }
}


template <typename KernelTraits, bool CAUSAL, typename Params, bool USE_REG_EALLOC=false>
cudaError_t BatchMLAWithPagedKVCacheKernelTraitsDispatched(Params& params,
                                                           cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;
  using NV_TYPE = typename KernelTraits::NV_TYPE;

  using CollectiveMainloop =
      CollectiveMainloop<KernelTraits, CAUSAL>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;

  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments({
      make_layout(make_shape(KernelTraits::BLOCK_SHAPE_Q, params.qk_head_dim), make_stride(params.qk_head_dim, _1{})), // layout q
      make_layout(make_shape(params.block_size, params.qk_head_dim, params.max_block_num), make_stride(params.qk_head_dim, _1{}, params.block_size * params.qk_head_dim)),
      make_layout(make_shape(params.chunk_num, params.bsz * params.draft_total_token_num * params.q_num_head), make_stride(params.bsz * params.draft_total_token_num * params.q_num_head, _1{})),
      params.Q,
      params.KV,
      params.m,
      params.d,
      params.block_tables,
      params.seq_lens_this_time,
      params.seq_lens_encoder,
      params.seq_lens_decoder,
      params.cumsum_q_seqlens,
      params.batch_ids,
      params.tile_ids_per_batch,
      params.q_tile_ids_per_batch,
      params.num_blocks_x,
      params.sm_scale,
      params.bsz,
      params.max_block_num,
      params.max_block_num_per_seq,
      params.q_stride_bsz,
      params.q_stride_head_num,
      params.kv_stride_block_num,
      params.kv_stride_block_size,
      params.o_stride_bsz,
      params.o_stride_head_num,
      params.chunk_size,
      params.chunk_num,
      params.draft_total_token_num
  });
  typename CollectiveEpilogue::Params epilogue_params = CollectiveEpilogue::to_underlying_arguments_ntma({
      params.O,
      make_layout(make_shape(KernelTraits::BLOCK_SHAPE_Q, params.vo_head_dim), make_stride(params.vo_head_dim, _1{})), // layout O
      params.O_tmp,
      make_layout(make_shape(KernelTraits::BLOCK_SHAPE_Q, params.vo_head_dim), make_stride(params.vo_head_dim, _1{})) // layout O_tmp
  });

  // Get the ptr to kernel function.
  auto kernel =
      MLAWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits, CAUSAL, USE_REG_EALLOC>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
  int act_blocks_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &act_blocks_per_sm, kernel, KernelTraits::NUM_WARPS * 32, smem_size);

  dim3 grid_dims = {params.num_blocks_x_int, 1, 1}; // todo: split kv
  static constexpr int ctaSize = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(ctaSize, 1, 1);
  kernel<<<grid_dims, block_dims, smem_size, stream>>>(
    mainloop_params, epilogue_params
  );
  if (params.chunk_num > 1) {
    constexpr int vec_size = 16 / sizeof(DTypeO);
    constexpr int merge_block_size = 256;
    constexpr int blockx = KernelTraits::HEAD_DIM_VO / vec_size;
    constexpr int blocky = (merge_block_size + blockx - 1) / blockx;
    dim3 grids_merge(min(multiprocessor_count, params.token_num), params.q_num_head); // 128k is too large
    dim3 blocks_merge(blockx, blocky);
    merge_multi_chunks_kernel<NV_TYPE, vec_size, blocky, KernelTraits::HEAD_DIM_VO><<<grids_merge, blocks_merge, 0, stream>>>(
      reinterpret_cast<NV_TYPE*>(params.O_tmp),
      params.m,
      params.d,
      params.seq_lens_this_time,
      params.seq_lens_decoder,
      params.seq_lens_encoder,
      params.padding_offsets,
      reinterpret_cast<NV_TYPE*>(params.O),
      params.max_seq_len,
      params.chunk_num,
      params.q_num_head,
      params.chunk_size,
      params.vo_head_dim,
      params.token_num,
      params.bsz,
      params.draft_total_token_num
    );
  }
  return cudaSuccess;
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, typename NV_TYPE, typename Params, bool USE_REG_EALLOC=false>
cudaError_t BatchMLAWithPagedKVCacheDispatched(Params& params, cudaStream_t stream) {
  constexpr bool CAUSAL = true;
  if constexpr (HEAD_DIM_QK == 576) {
    if (params.block_size == 32) {
      DISPATCH_GROUP_SIZE(params.q_num_head, GROUP_SIZE,
        BatchMLAWithPagedKVCacheKernelTraitsDispatched<
            AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, 
                                  HEAD_DIM_QK, 
                                  HEAD_DIM_VO, 
                                  GROUP_SIZE,
                                  /*BLOCK_SHAPE_Q_=*/64,
                                  /*BLOCK_SHAPE_KV_=*/32,
                                  /*NUM_STAGES_=*/4, 
                                  typename Params::DTypeQ,
                                  typename Params::DTypeKV, 
                                  typename Params::DTypeO,
                                  typename Params::IdType,
                                  NV_TYPE>,
            CAUSAL,
            Params,
            USE_REG_EALLOC>(params, stream);)
    } else if (params.block_size == 64) {
      DISPATCH_GROUP_SIZE(params.q_num_head, GROUP_SIZE,
        BatchMLAWithPagedKVCacheKernelTraitsDispatched<
            AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, 
                                  HEAD_DIM_QK, 
                                  HEAD_DIM_VO, 
                                  GROUP_SIZE,
                                  /*BLOCK_SHAPE_Q_=*/64,
                                  /*BLOCK_SHAPE_KV_=*/64,
                                  /*NUM_STAGES_=*/2, 
                                  typename Params::DTypeQ,
                                  typename Params::DTypeKV, 
                                  typename Params::DTypeO,
                                  typename Params::IdType,
                                  NV_TYPE>,
            CAUSAL,
            Params,
            USE_REG_EALLOC>(params, stream);)
    }
  } else {
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
};

}  // namespace mla_attn

#endif  // ATTENTION_HOPPER_PREFILL_SM90_CUH_
