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
  
  \brief A file contains the epilogue visitor Op with Tensor Output
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
///  ElementOutput T = ElementOutput(Visitor)
///  T-> device memory
///
template <
    typename ElementAccumulator_,  ///< Data type of the Accumulator
    typename OutputTileIterator_,  ///< Tile iterator type to write the tensor
    typename Visitor_              ///< Child visitor that produces the output tensor
>
class VisitorOpTensorOutput {
public:
    using ElementAccumulator = ElementAccumulator_;
    using OutputTileIterator = OutputTileIterator_;

    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
    using ElementOutput = typename OutputTileIterator::Element;

    using Visitor = Visitor_;

    /// Fragment type returned from Visitor
    using VisitAccessTypeVisitor = typename Visitor::VisitAccessType;
    using ElementVisitor = typename VisitAccessTypeVisitor::Element;

    using VisitAccessType = VisitAccessTypeVisitor;

    /// Fragment type of accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    /// Fragment type of output
    using OutputAccessType = Array<ElementOutput, kElementsPerAccess>;

    static_assert(kElementsPerAccess==VisitAccessTypeVisitor::kElements, "kElementsPerAccess mismatches with Visitor");

    struct SharedStorage {
        typename Visitor::SharedStorage storage_visitor;

        CUTLASS_HOST_DEVICE
        SharedStorage() { }
    };

    /// Host-constructable Argument structure
    struct Arguments {
        ElementOutput *output_ptr;                 ///< Pointer to the output tensor in device memory
        int ldt;                                   ///< Leading dimension of the output tensor operand
        int64_t batch_stride;                      ///< batch stride
        typename Visitor::Arguments visitor_arg;   ///< Argument type of visitor

        /// Methods
        CUTLASS_HOST_DEVICE
        Arguments(): output_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(
            ElementOutput *output_ptr,
            int ldt,
            int64_t batch_stride,
            typename Visitor::Arguments visitor_arg
        ):
            output_ptr(output_ptr),
            ldt(ldt),
            batch_stride(batch_stride),
            visitor_arg(visitor_arg)
        { }
    };

    /// Param structure
    struct Params {
        typename OutputTileIterator::Params params_output;
        ElementOutput *output_ptr;
        int64_t batch_stride;
        typename Visitor::Params visitor_param;

        /// Method
        CUTLASS_HOST_DEVICE
        Params():
            output_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            params_output(args.ldt),
            output_ptr(args.output_ptr),
            batch_stride(args.batch_stride),
            visitor_param(args.visitor_arg)
        { }
    };

private:
    OutputTileIterator iterator_T_;
    typename OutputTileIterator::Fragment fragment_T_;
    MatrixCoord problem_size;
    Visitor visitor_;
    int64_t batch_stride_;

public:

    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpTensorOutput(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ):
        visitor_(params.visitor_param, shared_storage.storage_visitor, thread_idx, threadblock_offset, problem_size),
        iterator_T_(
            OutputTileIterator(
                params.params_output,
                params.output_ptr,
                problem_size,
                thread_idx,
                threadblock_offset
            )
        ),
        problem_size(problem_size),
        batch_stride_(params.batch_stride) { }
    
    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {
        iterator_T_.add_pointer_offset(batch_idx * batch_stride_);
        visitor_.set_batch_index(batch_idx);
    }
    
    CUTLASS_DEVICE
    void begin_epilogue() {
        visitor_.begin_epilogue();
    }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {
        fragment_T_.clear();
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

        // Column guard
        MatrixCoord thread_offset_ = iterator_T_.thread_start() + OutputTileIterator::ThreadMap::iteration_offset(frag_idx);
        bool column_guard = (thread_offset_.column() < problem_size.column());

        if (column_guard) {
            NumericArrayConverter<ElementOutput, ElementVisitor, kElementsPerAccess> output_converter;
            OutputAccessType &output = reinterpret_cast<OutputAccessType *>(&fragment_T_)[frag_idx];
            output = output_converter(result);
        }

        return result;
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) {
        visitor_.end_row(row_idx);
    }

    CUTLASS_DEVICE
    void end_step(int step_idx) {
        visitor_.end_step(step_idx);
        iterator_T_.store(fragment_T_);
        ++iterator_T_;
    }

    CUTLASS_DEVICE
    void end_epilogue() {
        visitor_.end_epilogue();
    }

};



/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
