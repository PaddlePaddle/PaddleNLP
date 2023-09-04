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
  
  \brief A file contains the epilogue visitor Op with reduction over columns in CTA
*/

#pragma once
#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Epilogue Visitor operator for the following computation:
///
///  ElementReductionAccumulator R[j] = \sum_i ElementReductionAccumulator(T[i][j])
///  device memory <- ElementReduction(R[j])
///
template <
    typename ThreadblockShape_,             /// Threadblock shape
    typename ElementAccumulator_,           ///< Data type of the Accumulator
    typename ElementReduction_,             ///< Data type of the output reduction in device memory
    typename ElementReductionAccumulator_ , ///< Data type to accumulate reduction in smem and register
    typename OutputTileIterator_,           ///< Tile Iterator type
    typename Visitor_                       ///< preceeding visitor op
>
class VisitorOpColumnReduction {
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

        /// Number of distinct threads which must be reduced during the final reduction phase within the threadblock
        static int const kThreadRows = kThreadCount / kThreadsPerRow;

        /// Number of iterations (accesses) the threadblock takes to reduce a row
        static int const kThreadAccessesPerRow = const_max(1, (ThreadblockShape::kN + kThreadCount - 1) / kThreadCount);

        using StorageShape = MatrixShape<
            kThreadRows,
            ThreadblockShape::kN
        >;
    };

    using ReductionFragment = Array<ElementReductionAccumulator, ReductionDetail::kColumnsPerThread>;

    /// Shared storage
    struct SharedStorage {
        typename Visitor::SharedStorage storage_visitor;
        AlignedArray<ElementReductionAccumulator, ReductionDetail::StorageShape::kCount, 16> reduction;

        CUTLASS_HOST_DEVICE
        SharedStorage() {}
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
    ElementReductionAccumulator *reduction_smem_ptr_;  ///< Pointer to the partial reductions in shared memory
    ReductionFragment reduction_fragment;              ///< register fragments that hold the partial reduction
    Visitor visitor_;                                  ///< visitor
    int thread_idx_;
    MatrixCoord threadblock_offset;
    MatrixCoord problem_size_;
    int64_t batch_stride_;

public:

    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpColumnReduction(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ):
        visitor_(params.visitor_param, shared_storage.storage_visitor,
            thread_idx, threadblock_offset, problem_size),
        reduction_smem_ptr_(shared_storage.reduction.data()),
        reduction_output_ptr_(params.reduction_ptr),
        thread_idx_(thread_idx),
        threadblock_offset(threadblock_offset),
        problem_size_(problem_size),
        batch_stride_(params.batch_stride)
    { }

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {
        reduction_output_ptr_ += batch_idx * batch_stride_;
        visitor_.set_batch_index(batch_idx);
    }

    CUTLASS_DEVICE
    void begin_epilogue() {
        visitor_.begin_epilogue();
        
        // clear the reduction fragment
        reduction_fragment.clear();
    }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {
        visitor_.begin_step(step_idx);
    }

    CUTLASS_DEVICE
    void begin_row(int row_idx) {
        visitor_.begin_row(row_idx);
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

        NumericArrayConverter<ElementReductionAccumulator, ElementVisitor, kElementsPerAccess> reduction_converter;
        ReductionOp reduction_op;
        ReductionAccumulatorAccessType* reduction_fragment_ = reinterpret_cast<ReductionAccumulatorAccessType*>(&reduction_fragment);
        reduction_fragment_[column_idx] = reduction_op(reduction_fragment_[column_idx], reduction_converter(result));

        return result;
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) {
        visitor_.end_row(row_idx);
    }

    CUTLASS_DEVICE
    void end_step(int step_idx) {
        visitor_.end_step(step_idx);
    }

    CUTLASS_DEVICE
    void end_epilogue() {
        visitor_.end_epilogue();
         //
        // Store the partially reduced value to SMEM
        //

        // Guard against uses of the existing SMEM tile
        __syncthreads();

        using AccessType = AlignedArray<ElementReductionAccumulator, ThreadMap::kElementsPerAccess>;

        //
        // Determine a compact thread arrangement to store to SMEM
        //

        MatrixCoord thread_offset(
            thread_idx_ / ReductionDetail::kThreadsPerRow,
            (thread_idx_ % ReductionDetail::kThreadsPerRow) * ThreadMap::kElementsPerAccess
        );

        //
        // Each thread store its fragment to a SMEM
        //
        AccessType *aligned_reduction_ptr = reinterpret_cast<AccessType *>(
            &reduction_smem_ptr_[thread_offset.row() * ThreadblockShape::kN + thread_offset.column()]
        );

        AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(
            &reduction_fragment
        );

        CUTLASS_PRAGMA_UNROLL
        for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            int col_idx = column * ThreadMap::Delta::kColumn / ThreadMap::kElementsPerAccess;

            aligned_reduction_ptr[col_idx] = frag_ptr[column];
        }

        __syncthreads();

        //
        // Now, threads are assigned several columns of the output. The fetch over all rows from
        // the compacted SMEM tile and perform a reduction.
        //

        NumericConverter<ElementReduction, ElementReductionAccumulator> output_converter;

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < ReductionDetail::kThreadAccessesPerRow; ++j) {
            int column_idx = thread_idx_ + j * ReductionDetail::kThreadCount;

            ReductionOpScalar reduction_op;
            ElementReductionAccumulator reduction_element = ElementReductionAccumulator();

            int output_column_idx = threadblock_offset.column() + column_idx;

            if (column_idx < ThreadblockShape::kN && output_column_idx < problem_size_.column()) {
                
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ReductionDetail::kThreadRows; ++row) {
                    if (row) {
                        auto frag = reduction_smem_ptr_[row * ThreadblockShape::kN + column_idx];
                        reduction_element = reduction_op(reduction_element, frag);
                    }
                    else {
                        
                        reduction_element = reduction_smem_ptr_[column_idx];
                    }
                }

                // Store
                reduction_output_ptr_[column_idx + threadblock_offset.column() + threadblock_offset.row() / ThreadblockShape::kM * problem_size_.column()] = output_converter(reduction_element);
            }
        }
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
