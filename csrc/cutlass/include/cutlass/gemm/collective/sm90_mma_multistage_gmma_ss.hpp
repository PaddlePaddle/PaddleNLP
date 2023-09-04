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
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"

#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"

#include <cuda/std/type_traits>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class ClusterShape,
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
    MainloopSm90CpAsyncGmmaUnpredicated<Stages, ClusterShape>,
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
  using DispatchPolicy = MainloopSm90CpAsyncGmmaUnpredicated<Stages, ClusterShape>;
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

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(std::is_base_of<cute::GMMA::gmma_descriptor_iterator, typename TiledMma::FrgTypeA>::value &&
                std::is_base_of<cute::GMMA::gmma_descriptor_iterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  struct Params {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class Args>
  static constexpr Params
  to_underlying_arguments(Args const& args, void* workspace) {
    (void) workspace;
    return {args.ptr_A, args.dA, args.ptr_B, args.dB};
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      TensorA gA,
      TensorB gB,
      FrgTensorC& accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf,
      Params const& mainloop_params)
  {
    using namespace cute;

    (void) residue_mnk;

    static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2.");
    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(std::is_same<TransformA, cute::identity>::value,
      "SM90 warpgroup MMA must specify transforms through MMA_Atom.");
    static_assert(std::is_same<TransformB, cute::identity>::value,
      "SM90 warpgroup MMA must specify transforms through MMA_Atom.");
    static_assert(std::is_same<SmemCopyAtomA, void>::value,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(std::is_same<SmemCopyAtomA, void>::value,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_a;
    GmemTiledCopyB gmem_tiled_copy_b;
    auto gmem_thr_copy_a = gmem_tiled_copy_a.get_slice(thread_idx);
    auto gmem_thr_copy_b = gmem_tiled_copy_b.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_a.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_a.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_b.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_b.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    // Tile MMA atom and compute thread partitions across A, B and C
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA);                                     // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                     // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                               // (MMA,MMA_M,MMA_N,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                     // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                     // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                      // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                      // PIPE
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tAsA));                      // PIPE
    CUTE_STATIC_ASSERT_V(size<3>(tCsB) == size<3>(tBsB));                      // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    //
    // Prologue
    //

    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      copy(gmem_tiled_copy_a, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));
      copy(gmem_tiled_copy_b, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));
      cp_async_fence();
      ++k_tile_iter;
      --k_tile_count;
    }

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    //
    // Pipelined Main Loop
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      // Copy gmem to smem before computing gemm on each k-pipe
      // pipe index in smem where the next gmem tile will be read into
      copy(gmem_tiled_copy_a, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));
      copy(gmem_tiled_copy_b, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));
      cp_async_fence();
      if (k_tile_count > 0) { ++k_tile_iter; }

      //
      // Compute on k_tile
      //
      warpgroup_fence_operand(accum);
      warpgroup_arrive();

      cp_async_wait<DispatchPolicy::Stages-2>();
      cute::gemm(tiled_mma, tCrA(_,_,_,smem_pipe_read), tCrB(_,_,_,smem_pipe_read), accum);
      warpgroup_commit_batch();

      //
      // Advance the pipe
      //
      ++smem_pipe_read;
      smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? smem_pipe_read = 0 : smem_pipe_read;

      ++smem_pipe_write;
      smem_pipe_write = (smem_pipe_write == DispatchPolicy::Stages) ? smem_pipe_write = 0 : smem_pipe_write;

      // Wait for the pipeline MMAs to drain
      warpgroup_wait<0>();
      warpgroup_fence_operand(accum);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class ClusterShape,
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
    MainloopSm90CpAsyncGmma<Stages, ClusterShape>,
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
  using DispatchPolicy = MainloopSm90CpAsyncGmma<Stages, ClusterShape>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{})));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(std::is_base_of<cute::GMMA::gmma_descriptor_iterator, typename TiledMma::FrgTypeA>::value &&
                std::is_base_of<cute::GMMA::gmma_descriptor_iterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  struct Params {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class Args>
  static constexpr Params
  to_underlying_arguments(Args const& args, void* workspace) {
    (void) workspace;
    return {args.ptr_A, args.dA, args.ptr_B, args.dB};
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  template <
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2.");
    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(std::is_same<TransformA, cute::identity>::value,
      "SM90 warpgroup MMA must specify transforms through MMA_Atom.");
    static_assert(std::is_same<TransformB, cute::identity>::value,
      "SM90 warpgroup MMA must specify transforms through MMA_Atom.");
    static_assert(std::is_same<SmemCopyAtomA, void>::value,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(std::is_same<SmemCopyAtomA, void>::value,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    gA.data() = &gA(0, get<2>(residue_mnk), 0);
    gB.data() = &gB(0, get<2>(residue_mnk), 0);

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_a;
    GmemTiledCopyB gmem_tiled_copy_b;
    auto gmem_thr_copy_a = gmem_tiled_copy_a.get_slice(thread_idx);
    auto gmem_thr_copy_b = gmem_tiled_copy_b.get_slice(thread_idx);

    Tensor tAgA = gmem_thr_copy_a.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_a.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_b.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_b.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    //
    // PREDICATES
    //

    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tAcA = gmem_thr_copy_a.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = gmem_thr_copy_b.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m,0) = get<0>(tAcA(0,m,0)) < get<0>(residue_mnk);  // blk_m coord < residue_m
    }
    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = get<0>(tBcB(0,n,0)) < get<1>(residue_mnk);  // blk_n coord < residue_n
    }

    //
    // Prologue/PREFETCH
    //

    // Clear the smem tiles to account for predicated off loads
    clear(tAsA);
    clear(tBsB);

    // Start async loads for 0th k-tile, where we take care of the k residue
    {
      constexpr int k_pipe = 0;

      Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tAsA); ++k) {
        if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gA shifted)
          copy_if(gmem_tiled_copy_a, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,k_pipe));
        }
      }
      Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBsB); ++k) {
        if (get<1>(tBcB(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gB shifted)
          copy_if(gmem_tiled_copy_b, tBpB(_,k), tBgBk(_,_,k), tBsB(_,_,k,k_pipe));
        }
      }
      cp_async_fence();
      ++k_tile_iter;
      --k_tile_count;
    }

    // Start async loads for 1st k-tile onwards, no k-residue handling needed
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 1; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      if (k_tile_count <= 0) {
        clear(tApA);
        clear(tBpB);
      }
      copy_if(gmem_tiled_copy_a, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,k_pipe));  // CpAsync
      copy_if(gmem_tiled_copy_b, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,k_pipe));  // CpAsync
      cp_async_fence();
      ++k_tile_iter;
      --k_tile_count;
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA atom and compute thread partitions across A, B and C
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA);                                     // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                                     // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                               // (MMA,MMA_M,MMA_N,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                     // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(src_accum));                 // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                     // N
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(src_accum));                 // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                      // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                      // PIPE
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tAsA));                      // PIPE
    CUTE_STATIC_ASSERT_V(size<3>(tCsB) == size<3>(tBsB));                      // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Current pipe index in smem to read from
    int smem_pipe_read = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    //
    // Pipelined Main Loop
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > -(DispatchPolicy::Stages-1); --k_tile_count)
    {
      //
      // Copy gmem to smem for *k_tile_iter
      //
      if (k_tile_count <= 0) {
        clear(tApA);
        clear(tBpB);
      }
      copy_if(gmem_tiled_copy_a, tApA, tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,smem_pipe_write));  // CpAsync
      copy_if(gmem_tiled_copy_b, tBpB, tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,smem_pipe_write));  // CpAsync
      cp_async_fence();
      ++k_tile_iter;

      //
      // Compute on k_tile
      //
      warpgroup_fence_operand(accum);
      warpgroup_arrive();

      cp_async_wait<DispatchPolicy::Stages-2>();
      cute::gemm(tiled_mma, accum, tCrA(_,_,_,smem_pipe_read), tCrB(_,_,_,smem_pipe_read), src_accum);
      warpgroup_commit_batch();

      //
      // Advance the pipe
      //
      ++smem_pipe_read;
      smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? smem_pipe_read = 0 : smem_pipe_read;

      ++smem_pipe_write;
      smem_pipe_write = (smem_pipe_write == DispatchPolicy::Stages) ? smem_pipe_write = 0 : smem_pipe_write;

      // Wait for the pipeline MMAs to drain
      warpgroup_wait<0>();
      warpgroup_fence_operand(accum);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
