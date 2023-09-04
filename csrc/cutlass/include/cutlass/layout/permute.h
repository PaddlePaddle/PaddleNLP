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
    \brief Defines layout functions used by GEMM+permute path for common tensor or matrix formats.

    Like Layout functions, permute layout functions map logical coordinates to linear memory. They often require additional
    data to describe strides between elements.

    Permute layout functions must implement all members in the interface of NoPermute<> defined in this file. Address offset
    computation lies in operator() with private member variables  {col_permute_, row_permute_ and stride_permute_} as new addresses after permute op.
*/
#pragma once
#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include "assert.h"
#endif
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/coord.h"
#include "cutlass/tensor_coord.h"

namespace cutlass {
namespace layout {

class NoPermute {
public:
  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

private:
  //
  // Data members
  //

  MatrixCoord extent_;

  Index stride_unit_; //  sizeof(AccessType) / kElementsPerAccess in epilogue's predicated_tile_iterator

  Index stride_permute_;

public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  NoPermute() { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  NoPermute(MatrixCoord extent, Index stride_init): extent_(extent) { }

  /// Computes the address offset after Permute Op in Bytes
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord offset_init) { return 0; }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines permute layouts of various tensor formats.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Permute layout function for 4-D permuted tensors with output matrix (dimension as [M, N]) reshaped
/// as [M/D1, D1, D2, N/D2]. Then perform permute([0, 2, 1, 3]) on the corresponding output tensor.
template <int D1, int D2>
class Tensor4DPermute0213 {
public:
  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

private:
  //
  // Data members
  //

  MatrixCoord extent_;

  Index stride_permute_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermute0213() { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermute0213(MatrixCoord extent, Index stride_init): extent_(extent) {

    /// Update stride_permute with stride_init
    stride_permute_ = stride_init / D2 * D1; // stride in Elements

  }
  
  /// Computes the address offset after Permute Op in Bytes
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord offset_init) {
    // Permute as torch.permute(X1, [0, 2, 1, 3]) -> 4D Tensor indices as [i,j,k,l], the dimension of X
    // is [D0, D1, D2, D3], after permutation the dim of X1 is [D0, D2, D1, D3].
    assert(extent_.row() % D1 == 0);
    assert(extent_.column() % D2 == 0);

    int D3 = extent_.column() / D2;

    Index col_init = offset_init.column();
    Index row_init = offset_init.row();

    int l = col_init % D3;
    int k = col_init / D3;
    int j = row_init % D1;
    int i = row_init / D1;

    // After the Permute Op
    Index col_permute = l + j * D3;
    Index row_permute = k + i * D2;

    return LongIndex(row_permute) * LongIndex(stride_permute_) + LongIndex(col_permute);
  }

  /// Return D1
  CUTLASS_HOST_DEVICE
  Index d1() const {
    return D1;
  }

  /// Return D2
  CUTLASS_HOST_DEVICE
  Index d2() const {
    return D2;
  }
};

/// Permute layout function for 4-D permuted tensors for BMM with BMM output tensor (dimension as [B, M, N]) reshaped
/// as [B/D1, D1, M, N]. Then perform permute([0, 2, 1, 3]) on the corresponding whole BMM output tensor.
template <int D1>
class Tensor4DPermuteBMM0213 {
public:
  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

private:
  //
  // Data members
  //

  MatrixCoord extent_;

  Index stride_permute_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0213() { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor4DPermuteBMM0213(MatrixCoord extent, Index stride_init): extent_(extent) {

    /// Update stride_permute with stride_init
    stride_permute_ = stride_init * D1; // stride in Elements 

  }
  
  /// Computes the address offset after Permute Op in Bytes
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord offset_init) {

    // The batch index for BMM
    Index BMM_batch_idx = blockIdx.z;
    
    // Permute as torch.permute(X1, [0, 2, 1, 3]) -> 4D Tensor indices as [i,j,k,l], the dimension of X 
    // is [D0, D1, D2, D3], after permutation the dim of X1 is [D0, D2, D1, D3].
    int D2 = extent_.row();
    int D3 = extent_.column();

    Index col_init = offset_init.column();
    Index row_init = offset_init.row();

    int l = col_init;
    int k = row_init;
    int j = BMM_batch_idx % D1;
    int i = BMM_batch_idx / D1;

    // After the Permute Op
    Index col_permute = l + j * D3;
    Index row_permute = k + i * D2;

    return LongIndex(row_permute) * LongIndex(stride_permute_) + LongIndex(col_permute);
  }

  /// Return D1
  CUTLASS_HOST_DEVICE
  Index d1() const {
    return D1;
  }
};

/// Permute layout function for 5-D permuted tensors with output matrix (dimension as [M, N]) reshaped
/// as [M/T1, T1, T2, T3, N/T2/T3]. Then perform permute([2, 0, 3, 1, 4]) on the corresponding output tensor.
template <int T1, int T2, int T3>
class Tensor5DPermute20314 {
public:
  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

private:
  //
  // Data members
  //

  MatrixCoord extent_;

  Index stride_permute_;
  
public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute20314() { }

  /// Constructor
  CUTLASS_HOST_DEVICE
  Tensor5DPermute20314(MatrixCoord extent, Index stride_init): extent_(extent) {

    /// Update stride_permute with stride_init
    stride_permute_ = stride_init / T2 * T1; // stride in Elements 

  }
  
  /// Computes the address offset after Permute Op in Bytes
  CUTLASS_HOST_DEVICE
  LongIndex operator()(MatrixCoord offset_init) {

    // Permute as torch.permute(X1, [2, 0, 3, 1, 4]) -> 5D Tensor indices as [i,j,k,l,m], the dimension of X 
    // is [T0, T1, T2, T3, T4], after permutation the dim of X1 is [T2, T0, T3, T1, T4].
    int T0 = extent_.row() / T1;
    int T4 = extent_.column() / T2 / T3;

    Index col_init = offset_init.column();
    Index row_init = offset_init.row();

    int m = col_init % T4;
    int l = int(col_init / T4) % T3;
    int k = int(col_init / T4) / T3;
    int j = row_init % T1;
    int i = row_init / T1;

    // After the Permute Op
    Index col_permute = m + j * T4 + l * T1 * T4;
    Index row_permute = i + k * T0;

    return LongIndex(row_permute) * LongIndex(stride_permute_) + LongIndex(col_permute);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace layout
} // namespace cutlass
