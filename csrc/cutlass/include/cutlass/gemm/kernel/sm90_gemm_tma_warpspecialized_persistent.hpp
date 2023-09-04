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
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"

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
  std::enable_if_t<std::is_base_of_v<KernelTmaWarpSpecializedPersistent, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
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

  static constexpr uint32_t NumDmaWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 2;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{}) + (NumMmaWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  /// Register requirement for DMA and MATH WGs
  static constexpr uint32_t DmaRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = 232;

  /* Order Sequence barrier with two stages: one for Mainloop and one for Epilogue */
  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<
    StagesPerMathWarpGroup, NumMmaWarpGroups>;

  // Kernel level shared memory storage
  struct SharedStorage {
    using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
    using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
    using MathWarpGroupOrderBarrierStorage = typename MathWarpGroupOrderBarrier::SharedStorage;

    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
    alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order_barrier_storage;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

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
    KernelHardwareInfo hw_info;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (detail::IF_SWAP_AB<CollectiveMainloop>::value) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args, workspace),
      CollectiveEpilogue::to_underlying_arguments(args, workspace),
      {args.hw_info.device_id, sm_count}
    };
  }

  CUTLASS_HOST_DEVICE static
  bool
  can_implement(Arguments const& args) {
    bool implementable = args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);

    // Number of blocks per problem (without batch) must not exceed 2^31 for the persistent scheduler to calculate using FastDivmod
    auto problem_shape_MNKL = append<4>(args.problem_shape, Int<1>{});
    auto [problem_blocks_m, problem_blocks_n, problem_blocks_l] =
        detail::PersistentTileSchedulerSm90::get_tiled_blk_shape_mnl(problem_shape_MNKL, TileShape{}, ClusterShape{});
    uint64_t problem_blocks = problem_blocks_m * problem_blocks_n * problem_blocks_l;
    implementable = implementable && (problem_blocks < (uint64_t(1) << 31));

    return implementable;
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
    int sm_count = params.hw_info.sm_count;
    CUTLASS_TRACE_HOST("get_grid_shape(): Persistent schedule grid plan using SM count = " << sm_count);

    // Compute the total number of output tiles our problem has
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [problem_blocks_m, problem_blocks_n, problem_blocks_l] =
        detail::PersistentTileSchedulerSm90::get_tiled_blk_shape_mnl(problem_shape_MNKL, TileShape{}, ClusterShape{});
    int problem_blocks_total = problem_blocks_m * problem_blocks_n * problem_blocks_l;

    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    dim3 launch_grid(1, cute::size<1>(ClusterShape{}), 1);

    // The else path is generic, however, we can avoid some divs if we know Cluster size is 1
    if constexpr (size(ClusterShape{}) == 1) {
      launch_grid.x = std::min(sm_count, problem_blocks_total);
    }
    else {
      /*
      * Optimal grid size calculation is based on
      * GH100: 8 GPCs, 72 TPCs (9 TPCs/GPC), 2 SMs/TPC, 144 SMs per full GPU
      * Hence, maximum SMs per GPC = 18
      */
      constexpr int max_sm_per_gpc = 18;
      // Provided SM count could possibly be less than the assumed maximum SMs per GPC
      int min_num_gpc = sm_count < max_sm_per_gpc ? 1 : sm_count / max_sm_per_gpc;
      int max_blk_occupancy_per_gpc = max_sm_per_gpc - (max_sm_per_gpc % size(ClusterShape{}));
      int blk_per_device = min_num_gpc * max_blk_occupancy_per_gpc;

      launch_grid.x = std::min(
          blk_per_device       / size<1>(ClusterShape{}),
          problem_blocks_total / size<1>(ClusterShape{}));
    }

    return launch_grid;
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

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int warp_idx   = canonical_warp_idx();
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
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
    typename CollectiveMainloop::PipelineState collective_start_state_pipe;

    typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
    // DMA WG will not participate in these Ordered Barrier syncs
    params_math_wg_order_barrier.group_id = canonical_warp_group_idx() - static_cast<int>(WarpGroupRole::Consumer0);
    params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup; // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(shared_storage.math_wg_order_barrier_storage, params_math_wg_order_barrier);

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

    // Slice to get the tiles this thread block is responsible for
    Tensor gA_mkl = local_tile(mA_mkl, blk_shape, blk_coord, Step<_1, X,_1>{});                  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, blk_shape, blk_coord, Step< X,_1,_1>{});                  // (BLK_N,BLK_K,n,k,l)

    // Get iterations along k-dimension
    auto k_tile_count = size<3>(gA_mkl);

    detail::PersistentTileSchedulerSm90 scheduler(problem_shape_MNKL, blk_shape, ClusterShape{});

    if (warp_group_role == WarpGroupRole::Consumer1) {
      /* Advance 2nd Math WG to the next work tile for the startup */
      scheduler.advance_to_next_work();
      /* Advance 2nd Math WG pipeline state to the end of 1st Math WG */
      collective_start_state_pipe.advance(k_tile_count);
    }
    auto work_tile_info = scheduler.get_current_work();

    // Perform the collective scoped MMA
    CollectiveMainloop collective_mainloop;

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<DmaRegisterRequirement>();

      // For the DMA (prologue) - we start with an opposite phase - since we skip all waits
      // i.e., we know that the buffer is indeed empty
      typename CollectiveMainloop::PipelineState smem_pipe_write = cutlass::make_producer_start_state<Pipeline>();
      while (work_tile_info.is_valid_tile) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        // Slice with our work tile coordinates to construct mainloop tensor views
        Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                   // (BLK_M,BLK_K,k)
        Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                   // (BLK_N,BLK_K,k)

        auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));

        collective_mainloop.dma(
          pipeline,
          smem_pipe_write,
          gA, params.mainloop.tma_load_a,
          gB, params.mainloop.tma_load_b,
          k_tile_iter, k_tile_count,
          thread_idx,
          reinterpret_cast<char*>(&shared_storage.mainloop)
        );
        // Update starting pipeline state for the next tile
        smem_pipe_write.advance(k_tile_count);
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop

      // Make sure all Consumer Warp Groups have been waited upon
      collective_mainloop.dma_epilogue(pipeline, smem_pipe_write);
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      while (work_tile_info.is_valid_tile) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        // Slice with our work tile coordinates to construct mainloop tensor views
        Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                   // (BLK_M,BLK_K,k)
        Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                   // (BLK_N,BLK_K,k)

        auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));

        TiledMma tiled_mma;
        Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)
        clear(accumulators);

        /* Order two Math WG's MMA one after the other, helps hide Epilogue */
        math_wg_order_barrier.wait();

        collective_mainloop.mma(
          pipeline,
          collective_start_state_pipe,
          accumulators,
          k_tile_count,
          thread_idx,
          reinterpret_cast<char*>(&shared_storage.mainloop), 
          params.mainloop
        );

        /* Cue for next Math WG's MMA to start */
        math_wg_order_barrier.arrive();

        /* Order two Math WG's Epilogue one after the other */
        math_wg_order_barrier.wait();

        constexpr int BLK_M_RANK = rank<0>(blk_shape);
        bool m_oob = int(work_tile_info.M_idx) >= size<2>(gA_mkl);
        auto m_max_coord = unwrap(cute::transform(make_seq<BLK_M_RANK>{}, [&](auto i) {
            return  m_oob ? 0 : get<i>(M) - get<0,i>(blk_shape) * get<i>(m_coord);
          }));

        constexpr int BLK_N_RANK = rank<1>(blk_shape);
        bool n_oob = int(work_tile_info.N_idx) >= size<2>(gB_nkl);
        auto n_max_coord = unwrap(cute::transform(make_seq<BLK_N_RANK>{}, [&](auto i) {
            return  n_oob ? 0 : get<i>(N) - get<1,i>(blk_shape) * get<i>(n_coord);
          }));
        auto residue_mnk = make_tuple(m_max_coord, n_max_coord, Int<0>{});

        // Epilogue and write to gD
        CollectiveEpilogue epilogue{params.epilogue};
        epilogue(
          problem_shape_MNKL,
          blk_shape,
          blk_coord,
          accumulators,
          tiled_mma,
          residue_mnk,
          warp_group_thread_idx,
          reinterpret_cast<char*>(&shared_storage.epilogue)
        );

        /* Cue for next Math WG's Epilogue to start */
        math_wg_order_barrier.arrive();

        // Update starting pipeline state for the next tile
        collective_start_state_pipe.advance(k_tile_count * NumMmaWarpGroups);

        scheduler.advance_to_next_work(NumMmaWarpGroups);
        work_tile_info = scheduler.get_current_work();
      } // Scheduler work fetch loop
    } // Consumer Warp Groups End
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
