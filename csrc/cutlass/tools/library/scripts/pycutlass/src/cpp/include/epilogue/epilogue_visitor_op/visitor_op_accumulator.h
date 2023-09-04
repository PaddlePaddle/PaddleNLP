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
  
  \brief A file contains the epilogue visitor Op with accumulator
*/

#pragma once
#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue Visitor operator for the following Computation
///
/// ElementAccumulator accum;
/// return accum;
///
/// It can only be the leaf node of the epilogue tree

template <
    typename ElementAccumulator_,  ///< Data type of the Accumulator
    int      kElementsPerAccess_    ///< Number of elements computed per operation
>
class VisitorOpAccumulator{
public:
    using ElementAccumulator = ElementAccumulator_;
    static int const kElementsPerAccess = kElementsPerAccess_;

    /// Fragment type for Accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    /// Fragment type returned by this visitor
    using VisitAccessType = AccumulatorAccessType;

    /// SMEM buffer class required in the epilogue visitor
    struct SharedStorage {
        CUTLASS_HOST_DEVICE
        SharedStorage() {}
    };

    /// Host-constructable Arguments structure
    struct Arguments {
        // Note: it is strange that ctypes will return issue with empty arguments
        int tmp;

        CUTLASS_HOST_DEVICE
        Arguments() { }

        CUTLASS_HOST_DEVICE
        Arguments(int tmp): tmp(tmp) { }
    };

    /// Parameter structure
    struct Params {

        CUTLASS_HOST_DEVICE
        Params() { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args) { }
    };

public:
    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpAccumulator(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ) { }

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) { }

    CUTLASS_DEVICE
    void begin_epilogue() { }

    CUTLASS_DEVICE
    void begin_step(int step_idx) { }

    CUTLASS_DEVICE
    void begin_row(int row_idx) { }

    CUTLASS_DEVICE
    VisitAccessType visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frag_idx,
        AccumulatorAccessType const &accum
    ) {
        return accum;
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) { }

    CUTLASS_DEVICE
    void end_step(int step_idx) { }

    CUTLASS_DEVICE
    void end_epilogue() { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
