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
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/pipeline.hpp"
#include "cute/tensor.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class GridSwizzle_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  GridSwizzle_,
  std::enable_if_t<std::is_base_of_v<KernelTmaWarpSpecialized, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using GridSwizzle = GridSwizzle_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 90);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(std::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  static constexpr int SharedStorageSize = cute::max(
      sizeof(typename CollectiveMainloop::SharedStorage),
      sizeof(typename CollectiveEpilogue::SharedStorage));

  static constexpr uint32_t NumDmaWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{}) + (NumDmaWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    ElementA const* ptr_A = nullptr;
    StrideA dA{};
    ElementB const* ptr_B = nullptr;
    StrideB dB{};
    EpilogueParams epilogue_params{};
    KernelHardwareInfo hw_info;
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (detail::IF_SWAP_AB<CollectiveMainloop>::value) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args, workspace),
      CollectiveEpilogue::to_underlying_arguments(args, workspace)
    };
  }

  CUTLASS_HOST_DEVICE static
  bool
  can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
  }

  static
  int
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static constexpr
  dim3
  get_grid_shape(Params const& params) {
    auto cluster_shape = ClusterShape{};
    auto tile_shape = TileShape{};
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return detail::PersistentTileSchedulerSm90::get_tiled_blk_shape_mnl(
        problem_shape_MNKL, tile_shape, cluster_shape);
  }

  static constexpr
  dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    // Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
    #if ! defined(__CUDA_ARCH_FEAT_SM90_ALL)
      if constexpr(size<0>(typename TiledMma::AtomShape_MNK{}) == 64) {
        printf("ERROR : Arch conditional MMA instruction used without targetting sm90a compute capability. Aborting.\n");
        return;
      }
    #endif

    enum class WarpGroupRole {
      Producer = 0,
      Consumer = 1,
    };

    int thread_idx = int(threadIdx.x);
    int warp_idx   = canonical_warp_idx();
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role      = WarpGroupRole(canonical_warp_group_idx());
    int lane_predicate = cute::elect_one_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    using Pipeline = typename CollectiveMainloop::MainloopPipeline;

    using PipelineParams = typename CollectiveMainloop::PipelineParams;
    PipelineParams params_pipeline;
    params_pipeline.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    if (warp_group_role == WarpGroupRole::Producer) {
      params_pipeline.role = Pipeline::ThreadCategory::Producer;
    }
    else {
      params_pipeline.role = Pipeline::ThreadCategory::Consumer;
    }
    params_pipeline.is_leader = warp_group_thread_idx == 0;
    params_pipeline.num_consumers = NumThreadsPerWarpGroup;

    // Initialize pipeline and setup starting pipeline state for the collectives
    Pipeline pipeline = CollectiveMainloop::make_pipeline(smem_buf, params_pipeline);

    auto cluster_wait_fn = [&] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Separate out problem shape for convenience
    // Optionally append _1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = params.mainloop.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = params.mainloop.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    auto blk_coord = make_coord(_,_,_);                                                   // (m,n,k) -- defer the slice

    // Make tiled views
    Tensor gA_mkl = local_tile(mA_mkl, blk_shape, blk_coord, Step<_1, X,_1>{});                  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, blk_shape, blk_coord, Step< X,_1,_1>{});                  // (BLK_N,BLK_K,n,k,l)

    // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
    auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
    auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
    auto l_coord = idx2crd(int(blockIdx.z), shape<4>(gB_nkl));
    auto output_tile_coord = make_coord(m_coord, n_coord, _, l_coord);

    // Slice with m_coord and n_coord
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                       // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                       // (BLK_N,BLK_K,k)

    auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
    auto k_tile_count = size<2>(gA);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    // In a warp specialized kernel, CollectiveMainloop exposes data movement and compute operations separately
    CollectiveMainloop collective_mainloop;

    if (warp_group_role == WarpGroupRole::Producer) {
      // For the DMA (prologue) - we start with an opposite phase - since we skip all waits
      // i.e., we know that the buffer is indeed empty
      typename CollectiveMainloop::PipelineState smem_pipe_write = cutlass::make_producer_start_state<Pipeline>();
      collective_mainloop.dma(
        pipeline,
        smem_pipe_write,
        gA, params.mainloop.tma_load_a,
        gB, params.mainloop.tma_load_b,
        k_tile_iter, k_tile_count,
        thread_idx,
        smem_buf
      );
      // Update starting pipeline state for the next tile
      smem_pipe_write.advance(k_tile_count);
      // Make sure all Consumer Warp Groups have been waited upon
      collective_mainloop.dma_epilogue(pipeline, smem_pipe_write);
    }
    else if (warp_group_role == WarpGroupRole::Consumer) {
      typename CollectiveMainloop::PipelineState smem_pipe_read;
      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                 // (MMA,MMA_M,MMA_N)
      clear(accumulators);

      collective_mainloop.mma(
        pipeline,
        smem_pipe_read,
        accumulators,
        k_tile_count,
        thread_idx,
        smem_buf,
        params.mainloop
      );

      constexpr int BLK_M_RANK = rank<0>(blk_shape);
      bool m_oob = int(blockIdx.x) >= size<2>(gA_mkl);
      auto m_max_coord = unwrap(cute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
          return  m_oob ? 0 : get<i>(M) - get<0,i>(blk_shape) * get<i>(m_coord);
        }));

      constexpr int BLK_N_RANK = rank<1>(blk_shape);
      bool n_oob = int(blockIdx.y) >= size<2>(gB_nkl);
      auto n_max_coord = unwrap(cute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
          return  n_oob ? 0 : get<i>(N) - get<1,i>(blk_shape) * get<i>(n_coord);
        }));
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

      // Epilogue and write to gD
      CollectiveEpilogue epilogue{params.epilogue};
      epilogue(
        problem_shape_MNKL,
        blk_shape,
        output_tile_coord,
        accumulators,
        tiled_mma,
        residue_mnk,
        warp_group_thread_idx,
        smem_buf
      );
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
