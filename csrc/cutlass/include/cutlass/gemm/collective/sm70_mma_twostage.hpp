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

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor_predicate.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////
 
namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
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
    MainloopSm70TwoStageUnpredicated,
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
  using DispatchPolicy = MainloopSm70TwoStageUnpredicated;
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
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));

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

  /// Perform a threadblock-scoped matrix multiply-accumulate
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

    (void)residue_mnk;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 2,
      "MainloopTwoStage must not have a smem shape with a pipeline mode.");
    static_assert(rank(SmemLayoutB{}) == 2,
      "MainloopTwoStage must not have a smem shape with a pipeline mode.");

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_a;
    GmemTiledCopyB gmem_tiled_copy_b;
    auto copy_a_thr = gmem_tiled_copy_a.get_slice(thread_idx);
    auto copy_b_thr = gmem_tiled_copy_b.get_slice(thread_idx);

    Tensor tAgA = copy_a_thr.partition_S(gA);                                  // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = copy_a_thr.partition_D(sA);                                  // (ACPY,ACPY_M,ACPY_K)
    Tensor tBgB = copy_b_thr.partition_S(gB);                                  // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = copy_b_thr.partition_D(sB);                                  // (BCPY,BCPY_N,BCPY_K)

    // Allocate the register tiles for double buffering -- same shape as partitioned data
    Tensor tArA = make_fragment_like(tAsA);                                    // (ACPY,ACPY_M,ACPY_K)
    Tensor tBrB = make_fragment_like(tBsB);                                    // (BCPY,BCPY_N,BCPY_K)

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA  = thr_mma.partition_fragment_A(sA);                           // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.partition_fragment_B(sB);                           // (MMA,MMA_M,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //

    auto thr_copy_A       = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCsA           = thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M

    auto thr_copy_B       = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCsB           = thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    //
    // Prologue
    //

    // Copy gmem to rmem for the first k_tile
    copy(gmem_tiled_copy_a, tAgA(_,_,_,*k_tile_iter), tArA);
    copy(gmem_tiled_copy_b, tBgB(_,_,_,*k_tile_iter), tBrB);
    if (--k_tile_count > 0) ++k_tile_iter;
    // Copy rmem to smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    // Clear accumulators
    __syncthreads();

    // Load A, B smem->rmem for k=0
    copy(tCsA(_,_,0), tCrA_copy_view(_,_,0));
    copy(tCsB(_,_,0), tCrB_copy_view(_,_,0));
    //
    // Mainloop
    //

    // Size of the k-tiles's outer product mode (k)
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -1)
    {
      // Pipeline the outer products with a static for loop
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) 
      {
        if (k_block == K_BLOCK_MAX - 1) 
        {
          __syncthreads();

          // Copy rmem to smem
          copy(tArA, tAsA);
          copy(tBrB, tBsB);
          __syncthreads();
        }

        // Load A, B smem->rmem for k+1
        int k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;     // static
        copy(tCsA(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(tCsB(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        if (k_block == 0) 
        {
          // Copy gmem to rmem
          copy(gmem_tiled_copy_a, tAgA(_,_,_,*k_tile_iter), tArA);
          copy(gmem_tiled_copy_b, tBgB(_,_,_,*k_tile_iter), tBrB);
          if (--k_tile_count > 0) ++k_tile_iter;
        }

        // transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});

        // Thread-level register gemm for k
        // disambiguate gemm (shared with the namespace name)
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
      });
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
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
    MainloopSm70TwoStage,
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
  using DispatchPolicy = MainloopSm70TwoStage;
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
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));

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

  /// Perform a threadblock-scoped matrix multiply-accumulate
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
    static_assert(rank(SmemLayoutA{}) == 2,
      "MainloopTwoStage must not have a smem shape with a pipeline mode.");
    static_assert(rank(SmemLayoutB{}) == 2,
      "MainloopTwoStage must not have a smem shape with a pipeline mode.");

    // Construct shared memory tiles
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

    // Allocate the register tiles for double buffering -- same shape as partitioned data
    Tensor tArA = make_fragment_like(tAsA);                                    // (ACPY,ACPY_M,ACPY_K)
    Tensor tBrB = make_fragment_like(tBsB);                                    // (BCPY,BCPY_N,BCPY_K)

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
    // PREFETCH
    //

    // Clear the rmem tiles to account for predicated off loads
    clear(tArA);
    clear(tBrB);

    // Start async loads for 0th k-tile, where we take care of the k residue
    {
      Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tArA); ++k) {
        if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gA shifted)
          copy_if(gmem_tiled_copy_a, tApA(_,k), tAgAk(_,_,k), tArA(_,_,k));
        }
      }
      Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBrB); ++k) {
        if (get<1>(tBcB(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gB shifted)
          copy_if(gmem_tiled_copy_b, tBpB(_,k), tBgBk(_,_,k), tBrB(_,_,k));
        }
      }
      ++k_tile_iter;
      --k_tile_count;
    }

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA  = thr_mma.make_fragment_A(thr_mma.partition_A(sA));           // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.make_fragment_B(thr_mma.partition_B(sB));           // (MMA,MMA_M,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //

    auto thr_copy_A       = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCsA           = thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M

    auto thr_copy_B       = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCsB           = thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    //
    // Prologue
    //

    // Copy rmem to smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    // Clear accumulators
    __syncthreads();

    // Load A, B smem->rmem for k=0
    copy(tCsA(_,_,0), tCrA_copy_view(_,_,0));
    copy(tCsB(_,_,0), tCrB_copy_view(_,_,0));
    //
    // Mainloop
    //

    // Size of the k-tiles's outer product mode (k)
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -1)
    {
      // Pipeline the outer products with a static for loop
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) 
      {
        if (k_block == K_BLOCK_MAX - 1) 
        {
          __syncthreads();

          // Copy rmem to smem
          copy(tArA, tAsA);
          copy(tBrB, tBsB);
          __syncthreads();
        }

        // Load A, B smem->rmem for k+1
        int k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;    // static
        copy(tCsA(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(tCsB(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        if (k_block == 0) 
        {
          if (k_tile_count <= 0) {
            clear(tApA);
            clear(tBpB);
          }
          copy_if(gmem_tiled_copy_a, tApA, tAgA(_,_,_,*k_tile_iter), tArA);
          copy_if(gmem_tiled_copy_b, tBpB, tBgB(_,_,_,*k_tile_iter), tBrB);
          ++k_tile_iter;
          --k_tile_count;
        }

        // transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});

        // Thread-level register gemm for k
        // disambiguate gemm (shared with the namespace name)
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
      });
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
