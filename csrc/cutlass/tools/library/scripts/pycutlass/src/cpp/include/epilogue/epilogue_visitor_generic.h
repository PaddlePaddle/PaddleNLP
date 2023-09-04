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

  \brief A generic wrapper around an epilogue visitor operation
*/


#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

#include "epilogue_visitor_op/visitor_op_linear_combination.h"
#include "epilogue_visitor_op/visitor_op_tensor_input.h"
#include "epilogue_visitor_op/visitor_op_accumulator.h"
#include "epilogue_visitor_op/visitor_op_row_broadcast.h"
#include "epilogue_visitor_op/visitor_op_tensor_output.h"
#include "epilogue_visitor_op/visitor_op_column_reduction.h"
#include "epilogue_visitor_op/visitor_op_row_reduction.h"
#include "epilogue_visitor_op/visitor_op_column_broadcast.h"
#include "epilogue_visitor_op/visitor_op_unary.h"
#include "epilogue_visitor_op/visitor_op_binary.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Generic Epilogue Visitor.
template <
  typename OutputOp_
>
class EpilogueVisitorGeneric {
public:

  using OutputOp = OutputOp_;
  using AccumulatorAccessType = typename OutputOp::AccumulatorAccessType;
  static int const kElementsPerAccess = OutputOp::kElementsPerAccess;
  using ElementOutput = typename OutputOp::ElementOutput;
  using OutputTileIterator = typename OutputOp::OutputTileIterator;

  static int const kIterations = OutputTileIterator::kIterations;

  ///
  /// End Epilogue Tree
  ///

  /// Additional SMEM bufer is not required in the broadcast epilogue visitor
  struct SharedStorage {

    typename OutputOp::SharedStorage output_smem;
    CUTLASS_HOST_DEVICE
    SharedStorage() { }
  };

public:

  /// Argument structure
  struct Arguments {
    typename OutputOp::Arguments output_op_args;
    //
    // Methods
    //
    Arguments() { }

    Arguments(
      typename OutputOp::Arguments output_op_args
    ):
      output_op_args(output_op_args)
    {

    }
  };

  struct Params {
    typename OutputOp::Params output_op_params;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args):
      output_op_params(args.output_op_args)
    {

    }
  };



private:

  OutputOp output_op;

public:

  /// Constructor
  CUTLASS_DEVICE
  EpilogueVisitorGeneric(
    Params const &params,                                         ///< Parameters routed to the epilogue
    SharedStorage &shared_storage,                                ///< Shared storage needed by the functors here
    MatrixCoord threadblock_offset,
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    MatrixCoord problem_size
  ):
    output_op(params.output_op_params, shared_storage.output_smem, thread_idx, threadblock_offset, problem_size)
  { }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(
    int split_k_index,                                            ///< Index of this threadblock within split-K partitioned scheme
    int split_k_slices) {                                         ///< Total number of split-K slices

  }

  /// Called to set the batch index
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {
    output_op.set_batch_index(batch_idx);
  }

  /// Called at the start of the epilogue just before iterating over accumulator slices
  CUTLASS_DEVICE
  void begin_epilogue() {
    output_op.begin_epilogue();
  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    output_op.begin_step(step_idx);
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {
    output_op.begin_row(row_idx);
  }

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(
    int iter_idx,
    int row_idx,
    int column_idx,
    int frag_idx,
    AccumulatorAccessType const &accum) {
      output_op.visit(iter_idx, row_idx, column_idx, frag_idx, accum);
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void end_row(int row_idx) {
    output_op.end_row(row_idx);

  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {
    output_op.end_step(step_idx);
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {
    output_op.end_epilogue();
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
