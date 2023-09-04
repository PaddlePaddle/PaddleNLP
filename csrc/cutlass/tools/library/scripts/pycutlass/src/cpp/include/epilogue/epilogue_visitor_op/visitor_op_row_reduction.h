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
 * this layernormware without specific prior written permission.
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
  
  \brief A file contains the epilogue visitor Op with reduction over rows in CTA
*/

#pragma once
#include "cutlass/cutlass.h"
#include "stdio.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Epilogue Visitor operator for the following computation:
///
///  ElementReductionAccumulator R[i] = \sum_i ElementReductionAccumulator(T[i][j])
///  device memory <- ElementReduction(R[i])
///
template <
    typename ThreadblockShape_,             /// Threadblock shape
    typename ElementAccumulator_,           ///< Data type of the Accumulator
    typename ElementReduction_,             ///< Data type of the output reduction in device memory
    typename ElementReductionAccumulator_ , ///< Data type to accumulate reduction in smem and register
    typename OutputTileIterator_,           ///< Tile Iterator type
    typename Visitor_                       ///< preceeding visitor op
>
class VisitorOpRowReduction {
public:
    using ElementAccumulator = ElementAccumulator_;
    using ElementReductionAccumulator = ElementReductionAccumulator_;
    using ElementReduction = ElementReduction_;
    using OutputTileIterator = OutputTileIterator_;
    using ThreadblockShape = ThreadblockShape_;
    using Visitor = Visitor_;

    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

    using ReductionOp = cutlass::plus<Array<ElementReductionAccumulator, kElementsPerAccess>>;
    using ReductionOpScalar = cutlass::plus<ElementReductionAccumulator>;
    using ElementOutput = typename OutputTileIterator::Element;

    /// Fragment type returned from Visitor
    using VisitAccessTypeVisitor = typename Visitor::VisitAccessType;
    using ElementVisitor = typename VisitAccessTypeVisitor::Element;

    using VisitAccessType = VisitAccessTypeVisitor;

    /// Fragment type of accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    /// Fragment type of redcution
    using ReductionAccumulatorAccessType = Array<ElementReductionAccumulator, kElementsPerAccess>;

    /// Thread map used by output tile iterators
    using ThreadMap = typename OutputTileIterator::ThreadMap;
    /// Used for the reduction
    struct ReductionDetail {

        /// Number of threads per warp
        static int const kWarpSize = 32;

        /// Number of distinct scalar column indices handled by each thread
        static int const kColumnsPerThread = ThreadMap::Iterations::kColumn * ThreadMap::kElementsPerAccess;

        /// Number of distinct scalar row indices handled by each thread
        static int const kRowsPerThread = ThreadMap::Iterations::kCount / ThreadMap::Iterations::kColumn;

        /// Number of threads per threadblock
        static int const kThreadCount = ThreadMap::kThreads;

        /// Number of distinct threads per row of output tile
        static int const kThreadsPerRow = ThreadblockShape::kN / kColumnsPerThread;

        /// Half number of threads per row used for cross-thread reduction
        static int const kHalfThreadsPerRow = (kThreadsPerRow >> 1);

        /// Number of distinct threads which must be reduced during the final reduction phase within the threadblock
        static int const kThreadRows = kThreadCount / kThreadsPerRow;
    };

    /// Shared storage
    struct SharedStorage {
        typename Visitor::SharedStorage storage_visitor;
        CUTLASS_HOST_DEVICE
        SharedStorage() { }
    };

    /// Host-constructable Argument structure
    struct Arguments {
        ElementReduction *reduction_ptr;            ///< Pointer to the reduction tensor in device memory
        int64_t batch_stride;
        typename Visitor::Arguments visitor_arg;    ///< Argument type of visitor

        /// Method
        CUTLASS_HOST_DEVICE
        Arguments(): reduction_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(
            ElementReduction *reduction_ptr,
            int64_t batch_stride,
            typename Visitor::Arguments visitor_arg
        ):
            reduction_ptr(reduction_ptr),
            batch_stride(batch_stride),
            visitor_arg(visitor_arg)
        { }
    };

    /// Param structure
    struct Params {
        ElementReduction *reduction_ptr;            ///< Pointer to the reduction tensor in device memory
        int64_t batch_stride;
        typename Visitor::Params visitor_param;     ///< Argument type of visitor

        /// Method
        CUTLASS_HOST_DEVICE
        Params(): reduction_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            reduction_ptr(args.reduction_ptr),
            batch_stride(args.batch_stride),
            visitor_param(args.visitor_arg)
        { }
    };

private:
    ElementReduction *reduction_output_ptr_;           ///< Pointer to the reduction tensor in device memory
    ElementReductionAccumulator reduction_accum;
    Visitor visitor_;                                  ///< visitor
    int thread_idx_;
    MatrixCoord threadblock_offset;
    MatrixCoord problem_size_;

    int thread_start_row_;                             /// used to identify
    int state_[3];                                     /// used to track row iterator
    int thread_offset_row_;                            
    int64_t batch_stride_;
public:
    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpRowReduction(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ):
        visitor_(params.visitor_param, shared_storage.storage_visitor,
            thread_idx, threadblock_offset, problem_size),
        reduction_output_ptr_(params.reduction_ptr),
        thread_idx_(thread_idx),
        threadblock_offset(threadblock_offset),
        problem_size_(problem_size),
        thread_start_row_(ThreadMap::initial_offset(thread_idx).row() + threadblock_offset.row()),
        batch_stride_(params.batch_stride)
    {
        state_[0] = state_[1] = state_[2] = 0;
    }

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {
        reduction_output_ptr_ += batch_idx * batch_stride_;
        visitor_.set_batch_index(batch_idx);
    }

    CUTLASS_DEVICE
    void begin_epilogue() {
        visitor_.begin_epilogue();
    }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {
        visitor_.begin_step(step_idx);
    }

    CUTLASS_DEVICE
    void begin_row(int row_idx) {
        visitor_.begin_row(row_idx);

        reduction_accum = ElementReductionAccumulator(0);
    }

    CUTLASS_DEVICE
    VisitAccessType visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frag_idx,
        AccumulatorAccessType const &accum
    ) {
        /// Get result from visitor
        VisitAccessTypeVisitor result = visitor_.visit(iter_idx, row_idx, column_idx, frag_idx, accum);

        thread_offset_row_ = thread_start_row_ + ThreadMap::iteration_offset(frag_idx).row();

        ReductionOpScalar reduction_op;

        ElementReductionAccumulator reduction_accum_ = reduction(result);

        // After performing the in-thread reduction, we then perform cross-thread / in-warp reduction
        CUTLASS_PRAGMA_UNROLL
        for (int i = ReductionDetail::kHalfThreadsPerRow; i > 0; i >>= 1) {
            reduction_accum_ = reduction_op(reduction_accum_, __shfl_xor_sync(0xFFFFFFFF, reduction_accum_, i));
        }
        reduction_accum = reduction_op(reduction_accum, reduction_accum_);

        return result;
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) {
        visitor_.end_row(row_idx);
        NumericConverter<ElementReduction, ElementReductionAccumulator> output_converter;

        bool is_write_thread = (thread_offset_row_ < problem_size_.row() && (thread_idx_ % ReductionDetail::kThreadsPerRow) == 0);
        int row_offset = thread_offset_row_ + threadblock_offset.column() / ThreadblockShape::kN * problem_size_.row();

        ElementReduction *curr_ptr_reduction = reduction_output_ptr_ + row_offset;

        arch::global_store<ElementReduction, sizeof(ElementReduction)>(
            output_converter(reduction_accum),
            (void *)curr_ptr_reduction,
            is_write_thread);
    }

    CUTLASS_DEVICE
    void end_step(int step_idx) {
        visitor_.end_step(step_idx);

        // run operator ++
        ++state_[0];

        thread_start_row_ += ThreadMap::Shape::kRow;
        if (state_[0] == ThreadMap::Count::kRow) {
            state_[0] = 0;
            ++state_[1];
            thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * 
                ThreadMap::Shape::kRow * ThreadMap::Count::kRow;
            
            if (state_[1] == ThreadMap::Count::kGroup) {
                state_[1] = 0;
                ++state_[2];
                thread_start_row_ += ThreadMap::Count::kGroup *
                    ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
                
                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                }
            }
        }

    }

    CUTLASS_DEVICE
    void end_epilogue() {
        visitor_.end_epilogue();
    }

private:

    CUTLASS_DEVICE
    ElementReductionAccumulator reduction(VisitAccessTypeVisitor const& result) {
        ElementReductionAccumulator sum_ = ElementReductionAccumulator(0);

        ReductionOpScalar reduction_op;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < VisitAccessTypeVisitor::kElements; ++i) {
            sum_ = reduction_op(sum_, result[i]);
        }

        return sum_;
    }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
