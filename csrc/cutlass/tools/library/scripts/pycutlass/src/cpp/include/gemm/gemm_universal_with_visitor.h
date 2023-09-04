/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/params_universal_base.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/layout/matrix.h"

#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmUniversalwithEpilogueVisitor {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueVisitor::ElementOutput;
  using LayoutC = typename EpilogueVisitor::OutputTileIterator::Layout;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(
    128 / sizeof_bits<ElementA>::value,
    128 / sizeof_bits<ElementB>::value
  );

  //
  // Structures
  //

  /// Argument structure
  struct Arguments : UniversalArgumentsBase {

    //
    // Data members
    //

    typename EpilogueVisitor::Arguments epilogue_visitor;

    void const * ptr_A;
    void const * ptr_B;
    void const * ptr_C;
    void * ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;

    typename LayoutA::Stride stride_a;
    typename LayoutB::Stride stride_b;
    typename LayoutC::Stride stride_c;
    typename LayoutC::Stride stride_d;

    typename LayoutA::Stride::LongIndex lda;
    typename LayoutB::Stride::LongIndex ldb;
    typename LayoutC::Stride::LongIndex ldc;
    typename LayoutC::Stride::LongIndex ldd;

    int const * ptr_gather_A_indices;
    int const * ptr_gather_B_indices;
    int const * ptr_scatter_D_indices;

    //
    // Methods
    //

    Arguments():
      ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr),
      ptr_gather_A_indices(nullptr),
      ptr_gather_B_indices(nullptr),
      ptr_scatter_D_indices(nullptr) {}

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueVisitor::Arguments epilogue_visitor,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      typename LayoutA::Stride stride_a,
      typename LayoutB::Stride stride_b,
      typename LayoutC::Stride stride_c,
      typename LayoutC::Stride stride_d,
      int const *ptr_gather_A_indices = nullptr,
      int const *ptr_gather_B_indices = nullptr,
      int const *ptr_scatter_D_indices = nullptr
    ):
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue_visitor(epilogue_visitor),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C),
      stride_a(stride_a), stride_b(stride_b), stride_c(stride_c), stride_d(stride_d),
      ptr_gather_A_indices(ptr_gather_A_indices), ptr_gather_B_indices(ptr_gather_B_indices),
      ptr_scatter_D_indices(ptr_scatter_D_indices) {
      lda = 0;
      ldb = 0;
      ldc = 0;
      ldd = 0;
      CUTLASS_TRACE_HOST("GemmUniversal::Arguments::Arguments() - problem_size: " << problem_size);
      }

    /// constructs an arguments structure
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueVisitor::Arguments epilogue_visitor,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      typename LayoutA::Stride::LongIndex lda,
      typename LayoutB::Stride::LongIndex ldb,
      typename LayoutC::Stride::LongIndex ldc,
      typename LayoutC::Stride::LongIndex ldd,
      int const *ptr_gather_A_indices = nullptr,
      int const *ptr_gather_B_indices = nullptr,
      int const *ptr_scatter_D_indices = nullptr
    ):
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue_visitor(epilogue_visitor),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D),
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C),
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd),
      ptr_gather_A_indices(ptr_gather_A_indices), ptr_gather_B_indices(ptr_gather_B_indices),
      ptr_scatter_D_indices(ptr_scatter_D_indices) {
      stride_a = make_Coord(lda);
      stride_b = make_Coord(ldb);
      stride_c = make_Coord(ldc);
      stride_d = make_Coord(ldd);
      CUTLASS_TRACE_HOST("GemmUniversal::Arguments::Arguments() - problem_size: " << problem_size);
      }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);

      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.stride_a, args.stride_b);
      std::swap(args.batch_stride_A, args.batch_stride_B);
      std::swap(args.ptr_gather_A_indices, args.ptr_gather_B_indices);

      return args;
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params : UniversalParamsBase<
    ThreadblockSwizzle,
    ThreadblockShape,
    ElementA,
    ElementB,
    ElementC> {

    using ParamsBase = UniversalParamsBase<
      ThreadblockSwizzle,
      ThreadblockShape,
      ElementA,
      ElementB,
      ElementC>;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename EpilogueVisitor::OutputTileIterator::Params params_C;
    typename EpilogueVisitor::OutputTileIterator::Params params_D;

    typename EpilogueVisitor::Params epilogue_visitor;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C;
    void * ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;

    int * ptr_gather_A_indices;
    int * ptr_gather_B_indices;
    int * ptr_scatter_D_indices;

    int *semaphore;

    //
    // Methods
    //

    /// Default constructor
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int device_sms,
      int sm_occupancy
    ):
      ParamsBase(args, device_sms, sm_occupancy),
      params_A(args.lda ? make_Coord_with_padding<LayoutA::kStrideRank>(args.lda) : args.stride_a),
      params_B(args.ldb ? make_Coord_with_padding<LayoutB::kStrideRank>(args.ldb) : args.stride_b),
      params_C(args.ldc ? make_Coord_with_padding<LayoutC::kStrideRank>(args.ldc) : args.stride_c),
      params_D(args.ldd ? make_Coord_with_padding<LayoutC::kStrideRank>(args.ldd) : args.stride_d),
      epilogue_visitor(args.epilogue_visitor),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      ptr_gather_A_indices(const_cast<int *>(args.ptr_gather_A_indices)),
      ptr_gather_B_indices(const_cast<int *>(args.ptr_gather_B_indices)),
      ptr_scatter_D_indices(const_cast<int *>(args.ptr_scatter_D_indices)) {

    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr) {

      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;

      ptr_gather_A_indices = const_cast<int *>(args.ptr_gather_A_indices);
      ptr_gather_B_indices = const_cast<int *>(args.ptr_gather_B_indices);
      ptr_scatter_D_indices = const_cast<int *>(args.ptr_scatter_D_indices);

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;

      epilogue_visitor = args.epilogue_visitor;

      semaphore = static_cast<int *>(workspace);
      CUTLASS_TRACE_HOST("GemmUniversal::Params::update()");
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
    typename EpilogueVisitor::SharedStorage visitor;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GemmUniversalwithEpilogueVisitor() { }

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    CUTLASS_TRACE_HOST("GemmUniversalwithEpilogueVisitor::can_implement()");

    static int const kAlignmentA = (platform::is_same<LayoutA,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<LayoutA,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = (platform::is_same<LayoutB,
                                                      layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<LayoutB,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (platform::is_same<LayoutC,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<LayoutC,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value
            || platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm ||
      params.mode == GemmUniversalMode::kGemmSplitKParallel) {

      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {

        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[threadblock_tile_offset.k()];
    }

    __syncthreads();

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{
      offset_k,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      params.ptr_gather_A_indices);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      params.ptr_gather_B_indices);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations,
      accumulators,
      iterator_A,
      iterator_B,
      accumulators);

    //
    // Epilogue
    //

    // EpilogueOutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

    //
    // Fetch pointers based on mode.
    //

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // Tile iterator loading from source tensor.

    EpilogueVisitor epilogue_visitor(
        params.epilogue_visitor,
        shared_storage.visitor,
        threadblock_offset,
        threadblock_tile_offset,
        thread_idx,
        params.problem_size.mn()
    );

    if (params.mode == GemmUniversalMode::kBatched || params.mode == GemmUniversalMode::kArray) {
      epilogue_visitor.set_batch_index(threadblock_tile_offset.k());
    }

    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      semaphore.wait(threadblock_tile_offset.k());
    }


    // Execute the epilogue operator to update the destination tensor.
    epilogue(epilogue_visitor, accumulators);

    //
    // Release the semaphore
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
