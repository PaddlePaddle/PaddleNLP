/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/pipeline.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class ClusterShape,
  int PipelineAsyncMmaStages,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90TmaGmma<Stages, ClusterShape, PipelineAsyncMmaStages>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmma<Stages, ClusterShape, PipelineAsyncMmaStages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<
                             DispatchPolicy::Stages,
                             typename DispatchPolicy::ClusterShape>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      Step<_2,_1,_3>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      Step<_2,_1,_3>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(std::is_base_of<cute::GMMA::gmma_descriptor_iterator, typename TiledMma::FrgTypeA>::value &&
                std::is_base_of<cute::GMMA::gmma_descriptor_iterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(std::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || std::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(std::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || std::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = std::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = std::is_same_v<float, ElementB>;
  using InternalElementA = std::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = std::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  struct SharedStorage
  {
    cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
    cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };

  struct Params {
    InternalElementA const* ptr_A;
    StrideA dA;
    InternalElementB const* ptr_B;
    StrideB dB;
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(ptr_A, repeat_like(StrideA{}, int32_t(0)), dA),
        SmemLayoutA{}(_,_,0),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(ptr_B, repeat_like(StrideB{}, int32_t(0)), dB),
        SmemLayoutB{}(_,_,0),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any
    TMA_A tma_load_a;
    TMA_B tma_load_b;
  };

  //
  // Methods
  //

  template <class Args>
  static constexpr Params
  to_underlying_arguments(Args const& args, void* workspace) {
    (void) workspace;
    // Optionally append _1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(args.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    auto reinterpreted_ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto reinterpreted_ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    Tensor tensor_a = make_tensor(reinterpreted_ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(reinterpreted_ptr_B, make_layout(make_shape(N,K,L), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    return {
      reinterpreted_ptr_A,
      args.dA,
      reinterpreted_ptr_B,
      args.dB,
      tma_load_a,
      tma_load_b
    };
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params)
  {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class TensorA, class TMA_LOAD_A,
    class TensorB, class TMA_LOAD_B,
    class FrgTensorC,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  operator() (
      TensorA const& gA, TMA_LOAD_A& tma_load_a,
      TensorB const& gB, TMA_LOAD_B& tma_load_b,
      FrgTensorC& accum,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      char* shared_memory,
      Params const& mainloop_params)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2.");
    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(std::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(std::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and B
    //
    dim3 cluster_local_block_id = cute::block_id_in_cluster();
    auto block_tma_a = tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = tma_load_b.get_slice(cluster_local_block_id.x);

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);                                   // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                   // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);                                   // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                   // (TMA,TMA_N,TMA_K,PIPE)

    //
    // Prepare TMA membars and PREFETCH
    //

    // Number of pipelined k-tiles in smem
    constexpr int K_PIPE_MAX = DispatchPolicy::Stages;

    // NOTE: Another parameter: Partition the pipeline between active MMAs and active TMAs
    // Tunable via the dispatch policy to tollerate latencies evenly across the math and compute stages
    // K_PIPE_MMAS: The max number of active MMA pipes at beginning of every loop
    // K_PIPE_TMAS: The max number of active TMA pipes at beginning of every loop (geq 1)
    constexpr int K_PIPE_MMAS = DispatchPolicy::PipelineAsyncMmaStages;
    constexpr int K_PIPE_TMAS = K_PIPE_MAX - K_PIPE_MMAS;
    static_assert(0 <= K_PIPE_MMAS && K_PIPE_MMAS <  K_PIPE_MAX);
    static_assert(0 <  K_PIPE_TMAS && K_PIPE_TMAS <= K_PIPE_MAX);

    static_assert(K_PIPE_MMAS < K_PIPE_MAX - 1);

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    constexpr uint32_t TmaTransactionBytes = static_cast<uint32_t>(
        (size<0>(sA) * size<1>(sA) * sizeof(InternalElementA)) +
        (size<0>(sB) * size<1>(sB) * sizeof(InternalElementB)));


    // Obtain warp index
    int warp_idx = canonical_warp_idx();
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;

    PipelineParams params;
    params.transaction_bytes = TmaTransactionBytes;
    params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    params.is_leader = warp_group_thread_idx == 0;
    params.num_consumers = NumThreadsPerWarpGroup;

    MainloopPipeline pipeline(
      storage.pipeline_storage,
      params);

    // State variables used for iterating the circular buffer
    // smem_pipe_read / release is used by the consumer of SMEM data - i.e MMA
    // smem_pipe_write is used by the producer of SMEM data - i.e TMA
    PipelineState smem_pipe_read;
    PipelineState smem_pipe_release;
    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer blocks in the Cluster
    if constexpr (size(ClusterShape{}) > 1) {
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    }
    else {
      __syncthreads();
    }

    // Set predicate for the lowest lane_id in the warp
    int lane_predicate = cute::elect_one_sync();

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;
    // Keep a copy to know when to stop issuing loads
    int k_tile_count_tma = k_tile_count;

    // Issue TmaLoads (Prologue fetches)
    if (warp_idx == 0 && lane_predicate == 1) {
      // Maps the tile -> block, value
      if constexpr (std::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (std::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      // Issue the prologue loads
      int prologue_tma_count = min(K_PIPE_MAX, k_tile_count);
      CUTLASS_PRAGMA_UNROLL
      for (int stage = 0; stage < prologue_tma_count; ++stage) {
        pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename MainloopPipeline::ValueType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(stage);

        copy(tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,stage));
        copy(tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,stage));
        ++k_tile_iter;
        ++smem_pipe_write;
      }
      k_tile_count_tma -= prologue_tma_count;
    }

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thread_mma.partition_A(sA);                                  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                  // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                            // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                            // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                     // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                     // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                      // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                      // PIPE
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tAsA));                      // PIPE
    CUTE_STATIC_ASSERT_V(size<3>(tCsB) == size<3>(tBsB));                      // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    __syncthreads();

    warpgroup_fence_operand(accum);
    // Prologue MMAs
    CUTLASS_PRAGMA_UNROLL
    for (int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count); 
        prologue_mma_count > 0; --prologue_mma_count)
    {
      // WAIT on smem_pipe_read until it's data is available
      pipeline.consumer_wait(smem_pipe_read);
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_,_,_,smem_pipe_read.index()), tCrB(_,_,_,smem_pipe_read.index()), accum);  // (V,M,K) x (V,N,K) => (V,M,N)
      warpgroup_commit_batch();
      ++smem_pipe_read;
      --k_tile_count;
    }
    warpgroup_fence_operand(accum);

    //
    // PIPELINED MAIN LOOP
    //

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read until data is available
      pipeline.consumer_wait(smem_pipe_read);

      //
      // Compute on k_tile
      //

      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_,_,_,smem_pipe_read.index()), tCrB(_,_,_,smem_pipe_read.index()), accum);  // (V,M,K) x (V,N,K) => (V,M,N)
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      pipeline.consumer_release(smem_pipe_release);  // UNLOCK wr stage, done _computing_ on it

      //
      // Copy gmem to smem for *k_tile_iter
      //

      // Do Acquire & Load only if needed - helps with both performance and also corner case illegal barrier-ops
      if (warp_idx == 0 && lane_predicate == 1 && (k_tile_count_tma > 0) ) {
        pipeline.producer_acquire(smem_pipe_write);  // LOCK wr stage, for _writing_

        using BarrierType = typename MainloopPipeline::ValueType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write.index());

        copy(tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write.index()));
        copy(tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write.index()));
        ++smem_pipe_write;
        ++k_tile_iter;
        --k_tile_count_tma;
      }

      // Advance consumer pipeline
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    // Wait on all GMMAs
    warpgroup_wait<0>();
    warpgroup_fence_operand(accum);

    // Workaround for ensuring Smem destruction doesn't happen accidentally
    if constexpr (size(typename DispatchPolicy::ClusterShape{}) > 1) {
      cute::cluster_arrive();
      cute::cluster_wait();
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
