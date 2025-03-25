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

#ifndef ATTENTION_HOPPER_PREFILL_WG4_SM90_CUH_
#define ATTENTION_HOPPER_PREFILL_WG4_SM90_CUH_

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
#include "kernel_traits_wg4.cuh"
#include "mla_hopper.cuh"
#include "mainloop_mma.cuh"
#include "mainloop_load.cuh"
#include "utils.cuh"

#ifdef DEBUG_MLA
#undef DEBUG_MLA
#endif
// #define DEBUG_MLA

namespace mla_attn {

using namespace cute;

template <typename CollectiveMainloop, typename CollectiveEpilogue, typename Ktraits, bool CAUSAL, bool USE_REG_EALLOC=false, bool USE_QK_TWO_STAGE=false>
__global__ void __launch_bounds__(Ktraits::NUM_WARPS * cutlass::NumThreadsPerWarp, 1)
MLAWithKVCacheWG4Kernel(CUTE_GRID_CONSTANT
                        typename CollectiveMainloop::Params const mainloop_params,
                        CUTE_GRID_CONSTANT
                        typename CollectiveEpilogue::Params const epilogue_params) {
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;

  static constexpr int NUM_MMA_THREADS_QK = Ktraits::NUM_MMA_THREADS_QK;
  static constexpr int NUM_MMA_THREADS_PV = Ktraits::NUM_MMA_THREADS_PV;
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

  using MainloopPipelineQK = typename Ktraits::MainloopPipelineQK;
  using PipelineParamsQK = typename MainloopPipelineQK::Params;
  using PipelineStateQK = typename MainloopPipelineQK::PipelineState;

  extern __shared__ char shared_memory[];
  auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

  PipelineParams pipeline_params;
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  if constexpr (use_tma_load_kv) {
    int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NUM_MMA_THREADS_QK + NUM_MMA_THREADS_PV;
  } else {
    pipeline_params.producer_arv_count = NUM_COPY_THREADS;
    pipeline_params.consumer_arv_count = NUM_MMA_THREADS_QK + NUM_MMA_THREADS_PV;
  }

  PipelineParamsQ pipeline_params_q;
  if (warp_group_idx == 0) {
    pipeline_params_q.role = MainloopPipelineQ::ThreadCategory::Producer;
  } else if (warp_group_idx == 1) {
    pipeline_params_q.role = MainloopPipelineQ::ThreadCategory::Consumer;
  } else {
    pipeline_params_q.role = MainloopPipelineQ::ThreadCategory::NonParticipant;
  }
  pipeline_params_q.producer_arv_count = NUM_COPY_THREADS;
  pipeline_params_q.consumer_arv_count = cutlass::NumThreadsPerWarpGroup; // just one wg qk
  

  MainloopPipelineQ pipeline_q(shared_storage.pipeline_q, pipeline_params_q);

  PipelineParamsQK pipeline_params_qk;
  if (warp_group_idx == 0) {
    pipeline_params_qk.role = MainloopPipelineQK::ThreadCategory::NonParticipant;
  } else if (warp_group_idx == 1) {
    pipeline_params_qk.role = MainloopPipelineQK::ThreadCategory::Producer;
  } else {
    pipeline_params_qk.role = MainloopPipelineQK::ThreadCategory::Consumer;
  }
  pipeline_params_qk.producer_arv_count = NUM_MMA_THREADS_QK;
  pipeline_params_qk.consumer_arv_count = NUM_MMA_THREADS_PV;
  MainloopPipelineQK pipeline_qk(shared_storage.pipeline_qk, pipeline_params_qk);

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
    if constexpr (USE_REG_EALLOC) {
        cutlass::arch::warpgroup_reg_dealloc<64>();
    }
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t warp_idx_in_warpgroup = __shfl_sync(0xffffffff, warp_idx % 4, 0);
    PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();
    PipelineState smem_pipe_write_kv = cutlass::make_producer_start_state<MainloopPipeline>();
    const int block_id = blockIdx.x;
    const int bid = mainloop_params.batch_ids[block_id];
    const int tile_id = mainloop_params.tile_ids_per_batch ? mainloop_params.tile_ids_per_batch[block_id] : 0;
    const int q_tile_id = mainloop_params.q_tile_ids_per_batch ? mainloop_params.q_tile_ids_per_batch[block_id] : 0;
    const int seq_len_now = mainloop_params.seq_lens_this_time[bid];
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
  } else if (warp_group_idx == 1) {
    // mma qk
    if constexpr (USE_REG_EALLOC) {
        cutlass::arch::warpgroup_reg_dealloc<64>(); 
    }

    auto attention_updater = OnlineSoftmax<2 * BLOCK_SHAPE_Q / 64, /*WITH_SCALE=*/true>(mainloop_params.sm_scale);
    PipelineStateQ smem_pipe_read_q;
    PipelineState smem_pipe_read_kv;
    PipelineStateQK smem_pipe_write_qk = cutlass::make_producer_start_state<MainloopPipelineQK>();
    const int block_id = blockIdx.x;
    clear(attention_updater.scores_scale);
    const int bid = mainloop_params.batch_ids[block_id];
    const int tile_id = mainloop_params.tile_ids_per_batch ? mainloop_params.tile_ids_per_batch[block_id] : 0;
    const int q_tile_id = mainloop_params.q_tile_ids_per_batch ? mainloop_params.q_tile_ids_per_batch[block_id] : 0;
    const int seq_len_now = mainloop_params.seq_lens_this_time[bid];
    const int seq_len_decoder_now = mainloop_params.seq_lens_decoder[bid] + seq_len_now;
    if constexpr (USE_QK_TWO_STAGE) {
      mma_qk_two_stages<Ktraits, CAUSAL>(
          mainloop_params, 
          pipeline_q, 
          smem_pipe_read_q,
          pipeline_qk,
          smem_pipe_write_qk,
          pipeline_kv, 
          smem_pipe_read_kv,
          attention_updater, 
          threadIdx.x - NUM_COPY_THREADS,
          bid,
          seq_len_decoder_now,
          seq_len_now,
          tile_id,
          q_tile_id,
          shared_storage);
    } else {
      mma_qk_one_stages<Ktraits, CAUSAL>(
          mainloop_params, 
          pipeline_q, 
          smem_pipe_read_q,
          pipeline_qk,
          smem_pipe_write_qk,
          pipeline_kv, 
          smem_pipe_read_kv,
          attention_updater, 
          threadIdx.x - NUM_COPY_THREADS,
          bid,
          seq_len_decoder_now,
          seq_len_now,
          tile_id,
          q_tile_id,
          shared_storage);
    }
  } else {
    // mm pv and store 160 reg
    if constexpr (USE_REG_EALLOC) {
        cutlass::arch::warpgroup_reg_alloc<184>();
    }
    typename Ktraits::TiledMmaPVSS tiled_mma_pv;
    Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));
    auto attention_updater = OnlineSoftmax<2 * BLOCK_SHAPE_Q / 64, /*WITH_SCALE=*/true>(mainloop_params.sm_scale);
    PipelineState smem_pipe_read_kv;
    PipelineStateQK smem_pipe_read_qk;
    const int block_id = blockIdx.x;
    const int bid = mainloop_params.batch_ids[block_id];
    const int tile_id = mainloop_params.tile_ids_per_batch[block_id];
    const int q_tile_id = mainloop_params.q_tile_ids_per_batch ? mainloop_params.q_tile_ids_per_batch[block_id] : 0;
    const int seq_len_now = mainloop_params.seq_lens_this_time[bid];
    const int seq_len_decoder_now = mainloop_params.seq_lens_decoder[bid] + seq_len_now;
    const int start_token_idx = mainloop_params.cumsum_q_seqlens[bid];
    clear(tOrO);
    clear(attention_updater.scores_scale);
    mma_pv_one_stages<Ktraits, CAUSAL>(
        mainloop_params, 
        pipeline_qk, 
        smem_pipe_read_qk,
        pipeline_kv, 
        smem_pipe_read_kv,
        attention_updater,
        tOrO,
        threadIdx.x - NUM_COPY_THREADS - NUM_MMA_THREADS_QK,
        bid,
        seq_len_decoder_now,
        seq_len_now,
        tile_id,
        shared_storage);
    collective_epilogue.store(
      epilogue_params, 
      tOrO, 
      attention_updater.get_lse(),
      shared_storage,
      tiled_mma_pv, 
      threadIdx.x - NUM_COPY_THREADS - NUM_MMA_THREADS_QK,
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

template <typename KernelTraits, bool CAUSAL, typename Params, bool USE_REG_EALLOC=false, bool USE_QK_TWO_STAGE=false>
cudaError_t BatchMLAWithPagedKVCacheWG4KernelTraitsDispatched(Params& params,
                                                              cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;
  using NV_TYPE = typename KernelTraits::NV_TYPE;

  using CollectiveMainloop = CollectiveMainloop<KernelTraits, CAUSAL>;
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
      MLAWithKVCacheWG4Kernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits, CAUSAL, USE_REG_EALLOC, USE_QK_TWO_STAGE>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device);
  int act_blocks_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &act_blocks_per_sm, kernel, KernelTraits::NUM_WARPS * 32, smem_size);
  dim3 grid_dims = {params.num_blocks_x_int, 1, 1};
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

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, typename NV_TYPE, typename Params, bool USE_REG_EALLOC=false, bool USE_QK_TWO_STAGE=false>
cudaError_t BatchMLAWithPagedKVCacheWG4Dispatched(Params& params, cudaStream_t stream) {
  constexpr bool CAUSAL = true;
  if constexpr (HEAD_DIM_QK == 576) {
    if (params.block_size == 32) {
      DISPATCH_GROUP_SIZE(params.q_num_head, GROUP_SIZE,
        BatchMLAWithPagedKVCacheWG4KernelTraitsDispatched<
            AttentionKernelWG4Traits</*USE_TMA_LOAD_KV=*/false, 
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
            USE_REG_EALLOC,
            USE_QK_TWO_STAGE>(params, stream);)
    }
  } else {
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
};

}  // namespace mla_attn

#endif  // ATTENTION_HOPPER_PREFILL_WG4_SM90_CUH_
