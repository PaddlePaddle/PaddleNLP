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
    \brief Epilogue visitor type used for partial computation of a layernorm operation

    GemmLayernorm example =  GEMM0 with partial reduction fused in epilogue (EpilogueVisitorLayerNorm)
                          +  lightweight full reduction kernel (ApplyFinalReduction)
                          +  GEMM1 with elementwise operations fused in mainloop (GemmLayernormMainloopFusion)
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////


namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ThreadblockShape_,
  int ThreadCount,
  typename OutputTileIterator_,
  typename AccumulatorTile_,
  typename ElementAccumulator_,
  typename ElementVariance_,
  typename ElementMean_,
  typename ElementLayernormCompute_,
  typename ElementwiseFunctor_,
  bool IsShiftedVariance_ = false
>
class EpilogueVisitorLayerNorm {
public:

  using ElementVariance = ElementVariance_;
  using ElementMean = ElementMean_;
  using ElementLayernormCompute = ElementLayernormCompute_;

  using AccumulatorTile = AccumulatorTile_;

  using ThreadblockShape   = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static int const kRowIterations = OutputTileIterator::ThreadMap::Iterations::kRow;

  static int const kThreads = OutputTileIterator::ThreadMap::kThreads;

  static bool const kIsShiftedVariance = IsShiftedVariance_;

  using ElementOutput = typename OutputTileIterator::Element;

  static int const kDeltaRow = OutputTileIterator::ThreadMap::Delta::kRow;

  /// Array type used in Shift-K Layernorm
  static int const kRowAccessCount = kIterations * kRowIterations;

  using ConvertedShiftFragment = Array<ElementLayernormCompute, kRowAccessCount>;

  // Conducts manual transpose externally (already supported) for column major
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementAccumulator = ElementAccumulator_;

  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using LayernormFragment = Array<ElementLayernormCompute, kElementsPerAccess>;
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;
  using TensorRefD = TensorRef<ElementOutput, LayoutOutput>;

  static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::kAccessWidth;
  static int const kThreadsInColumn = kThreads / kThreadsPerRow;
  static int const kHalfThreadsPerRow = (kThreadsPerRow >> 1);

  /// Argument structure
  struct Arguments {

    typename ElementwiseFunctor::Params   elementwise;
    ElementVariance                       *ptr_Variance;
    ElementMean                           *ptr_Mean;
    ElementOutput                         *ptr_Shifted_K;
    MatrixCoord                           extent;

    //
    // Methods
    //
    Arguments():
      ptr_Variance(nullptr),
      ptr_Mean(nullptr),
      ptr_Shifted_K(nullptr)
    {

    }

    Arguments(
      typename ElementwiseFunctor::Params   elementwise_,
      ElementVariance                       *ptr_Variance,
      ElementMean                           *ptr_Mean_,
      ElementOutput                         *ptr_Shifted_K_ = nullptr,
      MatrixCoord                           extent = MatrixCoord(0, 0)
    ):
      elementwise(elementwise_),
      ptr_Variance(ptr_Variance),
      ptr_Mean(ptr_Mean_),
      ptr_Shifted_K(ptr_Shifted_K_),
      extent(extent)
    {

    }
  };

  struct Params {

    typename ElementwiseFunctor::Params   elementwise;
    ElementVariance                       *ptr_Variance;
    ElementMean                           *ptr_Mean;
    ElementOutput                         *ptr_Shifted_K;
    MatrixCoord                           extent;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params():
      ptr_Variance(nullptr),
      ptr_Mean(nullptr)
    {

    }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args):
      elementwise(args.elementwise),
      ptr_Variance(args.ptr_Variance),
      ptr_Mean(args.ptr_Mean),
      ptr_Shifted_K(args.ptr_Shifted_K),
      extent(args.extent)
    {

    }
  };

  /// Shared storage
  struct SharedStorage {

  };

private:

  Params const &                        params_;
  SharedStorage &                       shared_storage_;
  MatrixCoord                           extent_;
  ElementwiseFunctor                    elementwise_;

  OutputTileIterator                    iterator_C_;
  OutputTileIterator                    iterator_D_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator                    alpha_;
  ElementAccumulator                    beta_;
  ConvertedShiftFragment                shift_k_frag_;

  ElementLayernormCompute               accum_sum_square_;
  ElementLayernormCompute               accum_sum_element_;
  int                                   thread_idx_;

  MatrixCoord                           thread_offset_;

  gemm::GemmCoord                       threadblock_tile_offset_;

public:

  CUTLASS_DEVICE
  EpilogueVisitorLayerNorm(
    Params const &params,                                         ///< Parameters routed to the epilogue
    SharedStorage &shared_storage,                                ///< Shared storage needed by the functors here
    MatrixCoord threadblock_offset,
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    OutputTileIterator destination_iterator,                      ///< Tile iterator for destination
    OutputTileIterator source_iterator                            ///< Threadblock tile coordinate in GEMMM
  ):
    params_(params),
    shared_storage_(shared_storage),
    elementwise_(params.elementwise),
    extent_(params.extent),
    iterator_C_(source_iterator),
    iterator_D_(destination_iterator),
    threadblock_tile_offset_(threadblock_tile_offset),
    thread_idx_(thread_idx)
  {
    alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr : params.elementwise.alpha);
    beta_ =  (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr : params.elementwise.beta);

    if (beta_ == ElementAccumulator()) {
      iterator_C_.clear_mask();
    }
  }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(
    int split_k_index,                                            ///< Index of this threadblock within split-K partitioned scheme
    int split_k_slices) {                                         ///< Total number of split-K slices

  }

  /// Called to set the batch index
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {

  }

  /// Called at the start of the epilogue just before iterating over accumulator slices
  CUTLASS_DEVICE
  void begin_epilogue() {

    // If shift-K feature is enabled, we load shift-k fragment
    // at the very beginning of an epilogue
    if (kIsShiftedVariance && params_.ptr_Shifted_K != nullptr) {
      shift_k_frag_.clear();
      int thread_offset_row_base = iterator_D_.thread_start_row();

      CUTLASS_PRAGMA_UNROLL
      for (int iter_idx = 0; iter_idx < kIterations; ++iter_idx) {
        int step_offset = iter_idx * OutputTileIterator::Shape::kRow;
        CUTLASS_PRAGMA_UNROLL
        for (int rid = 0; rid < kRowIterations; ++rid) {
          int row_step_offset = rid * kDeltaRow;
          int row_offset = thread_offset_row_base + step_offset + row_step_offset;
          bool is_load = (row_offset < extent_.row());  
          shift_k_frag_[iter_idx * kRowIterations + rid] = load_shift_k_(row_offset, is_load);
        }

      }

    }

  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();

    if (elementwise_.kScale != cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      fragment_C_.clear();
      iterator_C_.load(fragment_C_);
      ++iterator_C_;
    }
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {
    /// set the accumulator to 0
    accum_sum_element_ = ElementLayernormCompute(0);
    accum_sum_square_ = ElementLayernormCompute(0);
  }

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(
    int iter_idx,
    int row_idx,
    int column_idx,
    int frag_idx,
    AccumulatorFragment const &accum) {

    using Mul = cutlass::multiplies<ElementLayernormCompute>;
    using Minus = cutlass::minus<ElementLayernormCompute>;
    using Exp   = cutlass::fast_exp_op<ElementLayernormCompute>;

    Minus     minus;
    Mul       mul;
    Exp       exponential;

    LayernormFragment result;

    thread_offset_ =
      iterator_D_.thread_start() +
      OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    NumericArrayConverter<ElementLayernormCompute, ElementOutput, kElementsPerAccess> source_converter;
    OutputVector &source_vector = reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];

    bool column_guard = (thread_offset_.column() < extent_.column());

    if (elementwise_.kScale == cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      result = source_converter(elementwise_(accum));
    }else{
      result = source_converter(elementwise_(accum, source_vector));
    }


    ElementLayernormCompute inv_scalar = cutlass::constants::one<ElementLayernormCompute>() / ElementLayernormCompute(extent_.column());

    // Fragment is cleared for non-reachable columns so no need to check against column guard
    ElementLayernormCompute accum_sum_element_tmp = element_sum_accumulator_(result);

    // Square sum is different. Non-reachable columns should've been computed for shift-k
    // Otherwise we will incorrectly have some extra k^2 added into square sum.
    ElementLayernormCompute accum_sum_square_tmp = ElementLayernormCompute(0);

    if (column_guard) {
      accum_sum_square_tmp = (kIsShiftedVariance) ? \
                        square_sum_accumulator_(result, shift_k_frag_[iter_idx * kRowIterations + row_idx]) : \
                        square_sum_accumulator_(result);
    }

    accum_sum_element_tmp *= inv_scalar;
    accum_sum_square_tmp *= inv_scalar;

    // After performing the in-thread reduction, we then perform cross-thread / in-warp reduction
    CUTLASS_PRAGMA_UNROLL
    for (int i = kHalfThreadsPerRow; i > 0; i >>= 1) {
      accum_sum_element_tmp += __shfl_xor_sync(0xFFFFFFFF, accum_sum_element_tmp, i);
      accum_sum_square_tmp += __shfl_xor_sync(0xFFFFFFFF, accum_sum_square_tmp, i);
    }
    accum_sum_element_ += accum_sum_element_tmp;
    accum_sum_square_ += accum_sum_square_tmp;

    // Convert to the output
    NumericArrayConverter<ElementOutput, ElementLayernormCompute, kElementsPerAccess> output_converter;
    OutputVector &output = reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
    output = output_converter(result);
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void end_row(int row_idx) {

    using ConvertVarianceOutput = cutlass::NumericConverter<ElementVariance, ElementLayernormCompute>;
    using ConvertMeanOutput = cutlass::NumericConverter<ElementMean, ElementLayernormCompute>;

    ConvertVarianceOutput   convert_variance_output;
    ConvertMeanOutput  convert_mean_output;

    bool is_write_thread = (thread_offset_.row() < extent_.row() && (threadIdx.x % kThreadsPerRow) == 0);
    int row_offset = thread_offset_.row() + threadblock_tile_offset_.n() * extent_.row();

    ElementVariance *curr_ptr_sum_square = params_.ptr_Variance + row_offset;
    ElementMean *curr_ptr_element_sum = params_.ptr_Mean + row_offset;

    arch::global_store<ElementVariance, sizeof(ElementVariance)>(
              convert_variance_output(accum_sum_square_),
              (void *)curr_ptr_sum_square,
              is_write_thread);

    arch::global_store<ElementMean, sizeof(ElementMean)>(
              convert_mean_output(accum_sum_element_),
              (void *)curr_ptr_element_sum,
              is_write_thread);
  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {

    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {

  }

private:

  CUTLASS_DEVICE
  ElementLayernormCompute load_shift_k_(int row_offset, bool is_load) {
    using ConvertShiftK = cutlass::NumericConverter<ElementLayernormCompute, ElementOutput>;
    ConvertShiftK convert_shift_k;    
    ElementOutput shift_k_val;

    // Computes the address to load shift_k element
    ElementOutput *curr_ptr_shift_k = params_.ptr_Shifted_K + row_offset;
    // Conditionally loads from global memory
    arch::global_load<ElementOutput, sizeof(ElementOutput)>(shift_k_val, (void *)curr_ptr_shift_k, is_load);
    // Converts data type to return
    ElementLayernormCompute converted_shift_k_val = convert_shift_k(shift_k_val);
    
    return converted_shift_k_val;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute square_sum_accumulator_(LayernormFragment const &accum) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      auto accum_ = accum[i];
      sum_ += accum_ * accum_;
    }

    return sum_;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute square_sum_accumulator_(LayernormFragment const &accum, ElementLayernormCompute shift_k_val) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      auto accum_ = accum[i] - shift_k_val;
      sum_ += accum_ * accum_;
    }

    return sum_;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute element_sum_accumulator_(LayernormFragment const &accum) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      sum_ += accum[i];
    }

    return sum_;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
