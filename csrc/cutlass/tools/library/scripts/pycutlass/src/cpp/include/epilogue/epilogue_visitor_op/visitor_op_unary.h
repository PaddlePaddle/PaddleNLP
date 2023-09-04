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
  
  \brief A file contains the epilogue visitor Op with Unary operation
*/

#pragma once
#include "cutlass/cutlass.h"
#include "unary_ops.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////


/// Epilogue Visitor operator for the following computation:
///
///  ElementCompute alpha;
///  ElementCompute beta;
///  ElementCompute C = UnaryOp(ElementCompute(Visitor)) 
///  Return C;
///
template <
    typename ElementAccumulator_,  ///< Data type of the Accumulator
    typename ElementCompute_,      ///< Data type used to compute linear combination
    int      kElementsPerAccess_,  ///< Number of elements computed per operation
    typename Visitor_,              ///< Child node
    template<typename T, int N> typename UnaryOp_
>
class VisitorOpUnary{
public:
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    static int const kElementsPerAccess = kElementsPerAccess_;

    using Visitor = Visitor_;

    /// Fragment type returned from Visitor.visit
    using VisitAccessTypeVisitor = typename Visitor::VisitAccessType;
    using ElementVisit = typename VisitAccessTypeVisitor::Element;

    /// Fragment type returned by this visitor
    using VisitAccessType = Array<ElementCompute, kElementsPerAccess>; 

    /// Fragment type of accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    /// Combination Op
    using UnaryOp = UnaryOp_<ElementCompute, kElementsPerAccess>;

    static_assert(kElementsPerAccess==VisitAccessTypeVisitor::kElements, "kElementsPerAccess mismatches with Visitor");

    /// SMEM buffer class required in the epilogue visitor
    struct SharedStorage {
        typename Visitor::SharedStorage storage_visitor;

        CUTLASS_HOST_DEVICE
        SharedStorage() {}
    };


    /// Host-constructable Arguments structure
    struct Arguments {
        typename UnaryOp::Arguments unary_arg;
        typename Visitor::Arguments visitor_arg;    ///< Argument type for visitor

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Arguments():unary_arg() { }
        
        CUTLASS_HOST_DEVICE
        Arguments(
            typename UnaryOp::Arguments unary_arg,
            typename Visitor::Arguments visitor_arg
        ):
            unary_arg(unary_arg),
            visitor_arg(visitor_arg)
        { }
    };

    /// Parameter structure
    struct Params {
        typename UnaryOp::Params unary_param;
        typename Visitor::Params visitor_param;    ///< Argument type for visitor

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Params():unary_param() { }
        
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            unary_param(args.unary_arg),
            visitor_param(args.visitor_arg)
        { }
    };

private:
    //
    // Data members
    //
    UnaryOp unary_op;

    Visitor visitor_op;

public:

    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpUnary(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ):
        unary_op(params.unary_param),
        visitor_op(params.visitor_param, shared_storage.storage_visitor, thread_idx, threadblock_offset, problem_size)
    { }

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {
        visitor_op.set_batch_index(batch_idx);
    }

    CUTLASS_DEVICE
    void begin_epilogue() {
        if (unary_op.guard()) visitor_op.begin_epilogue();
    }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {
        if (unary_op.guard()) visitor_op.begin_step(step_idx);
    }

    CUTLASS_DEVICE
    void begin_row(int row_idx) {
        if (unary_op.guard()) visitor_op.begin_row(row_idx);
    }

    CUTLASS_DEVICE
    VisitAccessType visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frag_idx,
        AccumulatorAccessType const &accum
    ) { 
        /// Get result from visitor A and visitor B
        VisitAccessTypeVisitor result;

        if (unary_op.guard()){
            result = visitor_op.visit(iter_idx, row_idx, column_idx, frag_idx, accum);
        } else {
            result.clear();
        }

        /// Type conversion
        NumericArrayConverter<ElementCompute, ElementVisit, kElementsPerAccess> source_converter;

        cutlass::multiplies<VisitAccessType> multiply_op;

        return unary_op(source_converter(result));
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) {
        if (unary_op.guard()) visitor_op.end_row(row_idx);
    }

    CUTLASS_DEVICE
    void end_step(int step_idx) {
        if (unary_op.guard()) visitor_op.end_step(step_idx);
    }

    CUTLASS_DEVICE
    void end_epilogue() {
        if (unary_op.guard()) visitor_op.end_epilogue();
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
