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
  
  \brief A file contains the epilogue visitor Op with broadcasting vector to all columns
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
///  ElementVector T[i][j] <- device-memory Td[i]
///
/// It can only be a leaf node in the epilogue tree
template <
    typename ElementAccumulator_,    ///< Data type of the Accumulator
    typename ElementFragment_,       ///< Data type used to cache vector in register
    typename InputTileIterator_      ///< Tile iterator type to read the broadcasted tensor
>
class VisitorOpColumnBroadcast {
public:
    using InputTileIterator = InputTileIterator_;

    static int const kElementsPerAccess = InputTileIterator::kElementsPerAccess;
    using ElementAccumulator = ElementAccumulator_;
    using ElementVector = typename InputTileIterator::Element;
    using ElementFragment = ElementFragment_;

    using VisitAccessType = Array<ElementFragment, kElementsPerAccess>;

    /// Thread map used by input tile iterators
    using ThreadMap = typename InputTileIterator::ThreadMap;

    /// Fragment object used to store the broadcast values
    using BroadcastFragment = Array<
        ElementFragment, kElementsPerAccess>;

    /// Fragment type of accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    /// Used for the broadcast
    struct BroadcastDetail {
        /// Number of threads per warp
        static int const kWarpSize = 32;

        static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

        /// Number of distinct scalar column indices handled by each thread
        static int const kColumnsPerThread = ThreadMap::Iterations::kColumn * ThreadMap::kElementsPerAccess;

        /// Number of distinct scalar row indices handled by each thread
        static int const kRowsPerThread = ThreadMap::Iterations::kCount / ThreadMap::Iterations::kColumn;

        /// Number of threads per threadblock
        static int const kThreadCount = ThreadMap::kThreads;

        /// Number of distinct threads per row of output tile
        static int const kThreadsPerRow = (InputTileIterator::Shape::kN / kColumnsPerThread);

        /// Number of distinct threads which must be reduced during the final reduction phase within the threadblock.
        static int const kThreadRows = kThreadCount / kThreadsPerRow;

        // /// Number of iterations (accesses) the threadblock takes to reduce a row
        // static int const kThreadAccessesPerRow = const_max(1, (Shape::kN + kThreadCount - 1) / kThreadCount);
    };

    // using ComputeFragmentType = Array<ElementVector, BroadcastDetail::kElementsPerAccess>;

    struct SharedStorage {
        CUTLASS_HOST_DEVICE
        SharedStorage() { }
    };

    /// Host-constructable Argument structure
    struct Arguments {
        ElementVector *broadcast_ptr;      ///< Pointer to the additional tensor operand
        int64_t batch_stride;

        /// Methods
        CUTLASS_HOST_DEVICE
        Arguments():
            broadcast_ptr(nullptr) { }
        
        CUTLASS_HOST_DEVICE
        Arguments(
            ElementVector *broadcast_ptr,
            int64_t batch_stride
        ):
            broadcast_ptr(broadcast_ptr),
            batch_stride(batch_stride) { }
    };

    /// Param structure
    struct Params {
        ElementVector *broadcast_ptr;      ///< Pointer to the additional tensor operand
        int64_t batch_stride;

        /// Method
        CUTLASS_HOST_DEVICE
        Params():
            broadcast_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            broadcast_ptr(args.broadcast_ptr),
            batch_stride(args.batch_stride) { }
    };

private:
    ElementVector *broadcast_ptr;
    BroadcastFragment broadcast_fragment;   ///< Array holds the loaded broadcast fragment
    MatrixCoord threadblock_offset_;
    int thread_idx_;
    MatrixCoord problem_size;
    
    int thread_start_row_;
    int state_[3];
    int thread_offset_row_;

    int64_t batch_stride_;

public:
    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpColumnBroadcast(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ):
        broadcast_ptr(params.broadcast_ptr),
        threadblock_offset_(threadblock_offset),
        thread_idx_(thread_idx),
        problem_size(problem_size),
        thread_start_row_(ThreadMap::initial_offset(thread_idx).row() + threadblock_offset.row()),
        batch_stride_(params.batch_stride)
    {
        state_[0] = state_[1] = state_[2] = 0;
    }

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {
        broadcast_ptr += batch_idx * batch_stride_;
    }
    
    CUTLASS_DEVICE
    void begin_epilogue() { }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {}

    CUTLASS_DEVICE
    void begin_row(int row_idx) {}

    CUTLASS_DEVICE
    VisitAccessType visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frag_idx,
        AccumulatorAccessType const &accum
    ) {
        // get pointer
        thread_offset_row_ = thread_start_row_ + ThreadMap::iteration_offset(frag_idx).row();
        
        ElementFragment broadcast_data = ElementFragment(*(broadcast_ptr + thread_offset_row_));

        broadcast_fragment.fill(broadcast_data);

        return broadcast_fragment;
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) { }

    CUTLASS_DEVICE
    void end_step(int step_idx) {
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
    void end_epilogue() { }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
