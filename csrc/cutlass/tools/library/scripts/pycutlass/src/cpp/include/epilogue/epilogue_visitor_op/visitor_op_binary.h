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
  
  \brief A file contains the epilogue visitor Op with Binary op
*/

#pragma once
#include "cutlass/cutlass.h"
#include "binary_ops.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue Visitor operator for the following computation:
///
///  ElementCompute alpha;
///  ElementCompute beta;
///  ElementCompute C = BinaryOp(alpha * ElementCompute(Visitor_A), beta * ElementCompute(Visitor_B) 
///  Return C;
///
template <
    typename ElementAccumulator_,  ///< Data type of the Accumulator
    typename ElementCompute_,      ///< Data type used to compute linear combination
    int      kElementsPerAccess_,   ///< Number of elements computed per operation
    typename VisitorA_,            ///< Child node A      
    typename VisitorB_,            ///< Child node B
    template<typename T, int N> typename BinaryOp_
>
class VisitorOpBinary{
public:
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    static int const kElementsPerAccess = kElementsPerAccess_;

    using VisitorA = VisitorA_;
    using VisitorB = VisitorB_;

    /// Fragment type returned from VisitorA.visit
    using VisitAccessTypeA = typename VisitorA::VisitAccessType;
    using ElementA = typename VisitAccessTypeA::Element;

    /// Fragment type returned from VisitorB.visit
    using VisitAccessTypeB = typename VisitorB::VisitAccessType;
    using ElementB = typename VisitAccessTypeB::Element;

    /// Fragment type returned by this visitor
    using VisitAccessType = Array<ElementCompute, kElementsPerAccess>; 

    /// Fragment type of accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    using BinaryOp = BinaryOp_<ElementCompute, kElementsPerAccess>;

    static_assert(kElementsPerAccess==VisitAccessTypeA::kElements, "kElementsPerAccess mismatches with Visitor A");
    static_assert(kElementsPerAccess==VisitAccessTypeB::kElements, "kElementsPerAccess misnatches with Visitor B");

    /// SMEM buffer class required in the epilogue visitor
    struct SharedStorage {
        typename VisitorA::SharedStorage storage_a;
        typename VisitorB::SharedStorage storage_b;

        CUTLASS_HOST_DEVICE
        SharedStorage() {}
    };


    /// Host-constructable Arguments structure
    struct Arguments {
        typename BinaryOp::Arguments binary_arg;
        typename VisitorA::Arguments visitor_a_arg;    ///< Argument type for visitor_a
        typename VisitorB::Arguments visitor_b_arg;    ///< Argument type for visitor_b

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Arguments():binary_arg() { }
        
        CUTLASS_HOST_DEVICE
        Arguments(
            typename BinaryOp::Arguments binary_arg,
            typename VisitorA::Arguments visitor_a_arg,
            typename VisitorB::Arguments visitor_b_arg
        ):
            binary_arg(binary_arg),
            visitor_a_arg(visitor_a_arg),
            visitor_b_arg(visitor_b_arg)
        { }
    };

    /// Parameter structure
    struct Params {
        typename BinaryOp::Params binary_param;
        typename VisitorA::Params visitor_a_param;    ///< Argument type for visitor_a
        typename VisitorB::Params visitor_b_param;    ///< Argument type for visitor_b

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Params() { }
        
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            binary_param(args.binary_arg),
            visitor_a_param(args.visitor_a_arg),
            visitor_b_param(args.visitor_b_arg)
        { }
    };

private:
    //
    // Data members
    //

    BinaryOp binary_op;

    VisitorA visitor_a_op;
    VisitorB visitor_b_op;

public:

    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpBinary(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ):
        binary_op(params.binary_param),
        visitor_a_op(params.visitor_a_param, shared_storage.storage_a, thread_idx, threadblock_offset, problem_size),
        visitor_b_op(params.visitor_b_param, shared_storage.storage_b, thread_idx, threadblock_offset, problem_size)
    { }


    CUTLASS_DEVICE
    void begin_epilogue() {
        visitor_a_op.begin_epilogue();
        visitor_b_op.begin_epilogue();
    }

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {
        visitor_a_op.set_batch_index(batch_idx);
        visitor_b_op.set_batch_index(batch_idx);
    }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {
        visitor_a_op.begin_step(step_idx);
        visitor_b_op.begin_step(step_idx);
    }

    CUTLASS_DEVICE
    void begin_row(int row_idx) {
        visitor_a_op.begin_row(row_idx);
        visitor_b_op.begin_row(row_idx);
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
        VisitAccessTypeA result_A = visitor_a_op.visit(iter_idx, row_idx, column_idx, frag_idx, accum);
        VisitAccessTypeB result_B = visitor_b_op.visit(iter_idx, row_idx, column_idx, frag_idx, accum);

        /// Type conversion
        NumericArrayConverter<ElementCompute, ElementA, kElementsPerAccess> source_converter_A;
        NumericArrayConverter<ElementCompute, ElementB, kElementsPerAccess> source_converter_B;

        return binary_op(
            source_converter_A(result_A),
            source_converter_B(result_B)
        );
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) {
        visitor_a_op.end_row(row_idx);
        visitor_b_op.end_row(row_idx);
    }

    CUTLASS_DEVICE
    void end_step(int step_idx) {
        visitor_a_op.end_step(step_idx);
        visitor_b_op.end_step(step_idx);
    }

    CUTLASS_DEVICE
    void end_epilogue() {
        visitor_a_op.end_epilogue();
        visitor_b_op.end_epilogue();
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
