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

#ifndef ATTENTION_HOPPER_MAINLOOP_LOAD_CUH_
#define ATTENTION_HOPPER_MAINLOOP_LOAD_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "named_barrier.cuh"
#include "utils.cuh"

#ifdef DEBUG_MLA
#undef DEBUG_MLA
#endif
// #define DEBUG_MLA

namespace mla_attn {

using namespace cute;

template <typename Ktraits, bool CAUSAL>
struct CollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeMD = float;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int BLOCK_SHAPE_Q = get<0>(TileShape_QKD{});
  static constexpr int BLOCK_SHAPE_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;

  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(DTypeQ); // 8
  static_assert(HEAD_DIM_QK % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad"); // 576 512
  static constexpr int kGmemThreadsPerRow = 64 / kGmemElemsPerLoad; // 8
  using AlignmentTypeQ = cute::uint_byte_t<static_cast<int>(sizeof(DTypeQ)) * kGmemElemsPerLoad>;
  using GmemCopyAtomQ = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<AlignmentTypeQ>, DTypeQ>;
  static constexpr int kNThreadsLoad = Ktraits::NUM_PRODUCER_THREADS;
  static_assert(kNThreadsLoad % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<
            Shape<Int<kNThreadsLoad / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, // 32, 8
            Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopy = decltype(make_tiled_copy(
          GmemCopyAtomQ{},
          GmemLayoutAtom{},
          Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

  using GmemLayoutAtomQ = Layout<
            Shape<Int<Ktraits::NUM_PRODUCER_THREADS / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, // 32, 8
            Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyQ = decltype(make_tiled_copy(
          GmemCopyAtomQ{},
          GmemLayoutAtomQ{},
          Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutAtomQ = typename Ktraits::SmemLayoutAtomQ;

  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  using ShapeQT = cute::Shape<int32_t, int32_t>;
  using StrideQT = cute::Shape<int32_t, _1>;
  using LayoutQT = cute::Layout<ShapeQT, StrideQT>;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1, int32_t>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeMDT = cute::Shape<int32_t, int32_t>;
  using StrideMDT = cute::Shape<int32_t, _1>;
  using LayoutMDT = cute::Layout<ShapeMDT, StrideMDT>;
  
  using TMA_KV = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)), 
          repeat_like(StrideT{}, int32_t(0)), StrideT{}
      ),
      take<0, 2>(SmemLayoutK{}),
      select<1, 2>(TileShape_QKD{}),
      _1{})); // no mcast for KV

  static constexpr bool USE_TMA_LOAD_KV = Ktraits::USE_TMA_LOAD_KV;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  using MainloopPipelineQ = typename Ktraits::MainloopPipelineQ;
  using PipelineParamsQ = typename MainloopPipelineQ::Params;
  using PipelineStateQ = typename MainloopPipelineQ::PipelineState;

  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);
  static constexpr uint32_t TmaTransactionBytesKV =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);

  // Host side kernel arguments
  struct Arguments {
    LayoutQT layout_Q;
    LayoutT layout_KV;
    LayoutMDT layout_MD;
    DTypeQ const* Q_ptr;
    DTypeKV const* KV_ptr;
    DTypeMD const* m_ptr;
    DTypeMD const* d_ptr;
    IdType const* kv_block_tables;
    IdType const* seq_lens_this_time;
    IdType const* seq_lens_encoder;
    IdType const* seq_lens_decoder;
    IdType const* cumsum_q_seqlens;
    IdType const* batch_ids;
    IdType const* tile_ids_per_batch;
    IdType const* q_tile_ids_per_batch;
    IdType const* num_blocks_x;
    float sm_scale;
    int bsz;
    int max_block_num;
    int max_block_num_per_seq;
    int q_stride_bsz;
    int q_stride_head_num;
    int kv_stride_block_num;
    int kv_stride_block_size;
    int o_stride_bsz;
    int o_stride_head_num;
    int chunk_size;
    int chunk_num;
    int draft_total_token_num;
  };

  // Device side kernel params
  struct Params {
    LayoutQT layout_Q;
    LayoutT layout_KV;
    LayoutMDT layout_MD;
    DTypeQ *Q_ptr;
    DTypeKV* KV_ptr;
    DTypeMD* m_ptr;
    DTypeMD* d_ptr;
    IdType* kv_block_tables;
    IdType* seq_lens_this_time;
    IdType* seq_lens_encoder;
    IdType* seq_lens_decoder;
    IdType* cumsum_q_seqlens;
    IdType* batch_ids;
    IdType* tile_ids_per_batch;
    IdType* q_tile_ids_per_batch;
    IdType* num_blocks_x;
    float sm_scale;
    int bsz;
    int max_block_num;
    int max_block_num_per_seq;
    int q_stride_bsz;
    int q_stride_head_num;
    int kv_stride_block_num;
    int kv_stride_block_size;
    int o_stride_bsz;
    int o_stride_head_num;
    int chunk_size;
    int chunk_num;
    int draft_total_token_num;
    TMA_KV tma_load_KV;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    TMA_KV tma_load_KV;
    if constexpr (USE_TMA_LOAD_KV) {
      Tensor mKV = make_tensor(make_gmem_ptr(args.KV_ptr), args.layout_KV);
      tma_load_KV = 
          make_tma_copy(GmemTiledCopyKV{}, mKV, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_QKD{}), _1{});
    }
    return {args.layout_Q,
            args.layout_KV,
            args.layout_MD,
            const_cast<DTypeQ*>(args.Q_ptr),
            const_cast<DTypeKV*>(args.KV_ptr),
            const_cast<DTypeMD*>(args.m_ptr),
            const_cast<DTypeMD*>(args.d_ptr),
            const_cast<IdType*>(args.kv_block_tables),
            const_cast<IdType*>(args.seq_lens_this_time),
            const_cast<IdType*>(args.seq_lens_encoder),
            const_cast<IdType*>(args.seq_lens_decoder),
            const_cast<IdType*>(args.cumsum_q_seqlens),
            const_cast<IdType*>(args.batch_ids),
            const_cast<IdType*>(args.tile_ids_per_batch),
            const_cast<IdType*>(args.q_tile_ids_per_batch),
            const_cast<IdType*>(args.num_blocks_x),
            args.sm_scale,
            args.bsz,
            args.max_block_num,
            args.max_block_num_per_seq,
            args.q_stride_bsz,
            args.q_stride_head_num,
            args.kv_stride_block_num,
            args.kv_stride_block_size,
            args.o_stride_bsz,
            args.o_stride_head_num,
            args.chunk_size,
            args.chunk_num,
            args.draft_total_token_num,
            tma_load_KV
            };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    if constexpr (USE_TMA_LOAD_KV) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_KV.get_tma_descriptor());
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load_q(Params const& mainloop_params, 
                             MainloopPipelineQ pipeline_q,
                             PipelineStateQ& smem_pipe_write_q,
                             SharedStorage& shared_storage,
                             const int thread_idx,
                             const int bid,
                             const int q_tile_idx) {
    const int q_group_offset = q_tile_idx * BLOCK_SHAPE_Q;
    int start_q_head_idx = mainloop_params.cumsum_q_seqlens[bid] * Ktraits::GROUP_SIZE + q_group_offset;
    int offset_Q = mainloop_params.q_stride_head_num * start_q_head_idx;
    Tensor mQ = make_tensor(make_gmem_ptr(mainloop_params.Q_ptr + offset_Q), mainloop_params.layout_Q);
    Tensor gQ =
        local_tile(mQ, select<0, 2>(TileShape_QKD{}), make_coord(_, _0{}))(_, _, _0{});
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor cQ = cute::make_identity_tensor(gQ.shape());

    GmemTiledCopyQ gmem_tiled_copy_q;
    auto gmem_thr_copy_q = gmem_tiled_copy_q.get_slice(thread_idx);
    Tensor tQgQ = gmem_thr_copy_q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_q.partition_D(sQ);
    Tensor tQcQ = gmem_thr_copy_q.partition_D(cQ);
    Tensor tQcQGroup = flatten_1(tQcQ);

    int valid_q_size = mainloop_params.seq_lens_this_time[bid];
    auto q_predicate_fn = [&](auto coords) {
      auto s_coords = tQcQGroup(_0{}, coords);
      return elem_less((get<0>(s_coords) + q_group_offset) / Ktraits::GROUP_SIZE, valid_q_size);
    };
    Tensor tQgQiGroup = flatten_1(tQgQ);
    Tensor tQsQiGroup = flatten_1(tQsQ);

    pipeline_q.producer_acquire(smem_pipe_write_q);
    copy_if(gmem_tiled_copy_q, q_predicate_fn, tQgQiGroup, tQsQiGroup);
    pipeline_q.producer_commit(smem_pipe_write_q, cutlass::arch::cpasync_barrier_arrive);
    ++smem_pipe_write_q;
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load_kv(Params const& mainloop_params, 
                              MainloopPipeline pipeline_kv,
                              PipelineState& smem_pipe_write_kv,
                              SharedStorage& shared_storage,
                              const int bid,
                              const int kv_len,
                              const int tile_idx) {
    int thread_idx = threadIdx.x;
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (thread_idx / 32) % 4, 0);
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});
    Tensor mKV = make_tensor(make_gmem_ptr(mainloop_params.KV_ptr), mainloop_params.layout_KV);
    Tensor gKV = local_tile(mKV, make_shape(get<1>(TileShape_QKD{}), get<2>(TileShape_QKD{})), make_coord(_, _))(_, _, _0{}, _0{}, _);
    GmemTiledCopy gmem_tiled_copy_kv;
    auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_slice(thread_idx);

    static constexpr int BLOCK_SHAPE_KV = get<1>(TileShape_QKD{});
    const int start_len = tile_idx * mainloop_params.chunk_size;
    const int start_tile_idx = start_len / BLOCK_SHAPE_KV;
    const int end_tile_idx = cute::ceil_div(min(start_len + mainloop_params.chunk_size, kv_len), BLOCK_SHAPE_KV) - 1;

    auto kv_block_tables = make_tensor(make_gmem_ptr(mainloop_params.kv_block_tables), make_layout(make_shape(mainloop_params.bsz, mainloop_params.max_block_num_per_seq), make_stride(mainloop_params.max_block_num_per_seq, 1)));

    Tensor tKgK = gmem_thr_copy_kv.partition_S(gKV);
    Tensor tKsK = gmem_thr_copy_kv.partition_S(sK);

    for (int kv_tile_idx = end_tile_idx; kv_tile_idx >= start_tile_idx; --kv_tile_idx) {
      const int block_idx = kv_block_tables(bid, kv_tile_idx);
      pipeline_kv.producer_acquire(smem_pipe_write_kv);
      Tensor tKgKiGroup = flatten_1(tKgK(_, _, _, block_idx));
      Tensor tKsKiGroup =
          flatten_1(tKsK(_, _, _, smem_pipe_write_kv.index()));
      copy(gmem_tiled_copy_kv, tKgKiGroup, tKsKiGroup);
      pipeline_kv.producer_commit(smem_pipe_write_kv, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write_kv;
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load_kv_tma(Params const& mainloop_params, 
                                  MainloopPipeline pipeline_kv,
                                  PipelineState& smem_pipe_write_kv,
                                  SharedStorage& shared_storage,
                                  const int bid,
                                  const int kv_len,
                                  const int tile_idx) {
    int thread_idx = threadIdx.x;
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});

    Tensor mKV = mainloop_params.tma_load_KV.get_tma_tensor(mainloop_params.layout_KV.shape());

    // Prepare the TMA loads
    Tensor gKV = local_tile(mKV, make_shape(get<1>(TileShape_QKD{}), get<2>(TileShape_QKD{})), make_coord(_, _))(_, _, _0{}, _0{}, _);
    auto [tKgK, tKsK] = 
        tma_partition(mainloop_params.tma_load_KV, _0{}, Layout<_1>{},
                      group_modes<0, 2>(sK), group_modes<0, 2>(gKV));

    static constexpr int BLOCK_SHAPE_KV = get<1>(TileShape_QKD{});
    const int start_len = tile_idx * mainloop_params.chunk_size;
    const int start_tile_idx = start_len / BLOCK_SHAPE_KV;
    const int end_tile_idx = cute::ceil_div(min(start_len + mainloop_params.chunk_size, kv_len), BLOCK_SHAPE_KV) - 1;

    auto kv_block_tables = make_tensor(make_gmem_ptr(mainloop_params.kv_block_tables), make_layout(make_shape(mainloop_params.bsz, mainloop_params.max_block_num_per_seq), make_stride(mainloop_params.max_block_num_per_seq, 1)));

    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
#pragma unroll 2
      for (int kv_tile_idx = end_tile_idx; kv_tile_idx >= start_tile_idx; --kv_tile_idx) {
        const int block_idx = kv_block_tables(bid, kv_tile_idx);
        pipeline_kv.producer_acquire(smem_pipe_write_kv);
        copy(mainloop_params.tma_load_KV.with(*pipeline_kv.producer_get_barrier(smem_pipe_write_kv), /*mcast_mask=*/0),
             tKgK(_, block_idx), tKsK(_, smem_pipe_write_kv.index()));
        ++smem_pipe_write_kv;
      }
    }
  }
};

}  // namespace mla_attn

#endif  // ATTENTION_HOPPER_SPARSE_MAINLOOP_CUH_
