/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>

#include <cute/util/type_traits.hpp>
#include <cute/tensor.hpp>

#include <cute/numeric/half.hpp>
#include <cute/numeric/complex.hpp>

#include <cutlass/layout/layout.h>

// The computed infinity norm does not include
// any NaN column absolute-value sums.
struct matrix_inf_norm_result {
  // Accumulate errors in double, as this is generally
  // the highest precision that the examples use.
  double inf_norm = 0.0;
  bool found_nan = false;
};

// In theory, cute::Tensor<ViewEngine<T*>, T> could be treated as a view type,
// and thus passed by value (as std::span or std::string_view would be).
// However, generic cute::Tensor are more like containers
// and thus are best passed by reference or const reference.
template <typename EngineType, typename LayoutType>
matrix_inf_norm_result
matrix_inf_norm(const cute::Tensor<EngineType, LayoutType>& host_matrix)
{
  using std::abs;
  using error_type = decltype(std::declval<matrix_inf_norm_result>().inf_norm);

  error_type inf_norm = 0.0;
  bool found_nan = false;

  const auto shape = host_matrix.shape();
  using index_type = std::decay_t<decltype(cute::get<0>(shape))>;
  // Computing the infinity norm requires that we be able
  // to treat the input as a matrix, with rows and columns.
  static_assert(std::is_integral_v<index_type>);
  const index_type num_rows = cute::get<0>(shape);
  const index_type num_cols = cute::get<1>(shape);

  for(index_type i = 0; i < num_rows; ++i) {
    error_type row_abs_sum = 0.0;
    for(index_type j = 0; j < num_cols; ++j) {
      row_abs_sum += abs(host_matrix(i, j));
    }
    if(std::isnan(row_abs_sum)) {
      found_nan = true;
    } else {
      inf_norm = row_abs_sum > inf_norm ? row_abs_sum : inf_norm;
    }
  }

  return {inf_norm, found_nan};
}

// Infinity norm of (X - Y).
template <typename EngineType, typename LayoutType>
matrix_inf_norm_result
matrix_diff_inf_norm(const cute::Tensor<EngineType, LayoutType>& X,
                     const cute::Tensor<EngineType, LayoutType>& Y)
{
  using std::abs;
  using error_type = decltype(std::declval<matrix_inf_norm_result>().inf_norm);

  const auto X_shape = X.shape();
  const auto Y_shape = Y.shape();

  using index_type = std::decay_t<decltype(cute::get<0>(X_shape))>;
  // Computing the infinity norm requires that we be able
  // to treat the input as a matrix, with rows and columns.
  static_assert(std::is_integral_v<index_type>);
  const index_type num_rows = cute::get<0>(X_shape);
  const index_type num_cols = cute::get<1>(X_shape);

  assert(num_rows == cute::get<0>(Y_shape));
  assert(num_cols == cute::get<1>(Y_shape));

  auto matrix_ij = [&](const auto& A, std::size_t i, std::size_t j) {
    return A(i, j); 
  };
  auto diff_ij = [&](std::size_t i, std::size_t j) {
    return matrix_ij(X, i, j) - matrix_ij(Y, i, j);
  };

  error_type inf_norm = 0.0;
  bool found_nan = false;

  for(index_type i = 0; i < num_rows; ++i) {
    error_type row_abs_sum = 0.0;
    for(index_type j = 0; j < num_cols; ++j) {
      row_abs_sum += abs(diff_ij(i, j));
    }
    if(std::isnan(row_abs_sum)) {
      found_nan = true;
    } else {
      inf_norm = row_abs_sum > inf_norm ? row_abs_sum : inf_norm;
    }
  }

  return {inf_norm, found_nan};
}

template <typename EngineType_A, typename LayoutType_A,
          typename EngineType_B, typename LayoutType_B,
          typename EngineType_C_computed, typename LayoutType_C_computed,
          typename EngineType_C_expected, typename LayoutType_C_expected>
void
print_matrix_multiply_mollified_relative_error(
  const char A_value_type_name[],
  const cute::Tensor<EngineType_A, LayoutType_A>& A,
  const char B_value_type_name[],
  const cute::Tensor<EngineType_B, LayoutType_B>& B,
  const char C_value_type_name[],
  const cute::Tensor<EngineType_C_computed, LayoutType_C_computed>& C_computed,
  const cute::Tensor<EngineType_C_expected, LayoutType_C_expected>& C_expected)
{
  const auto [A_norm, A_has_nan] = matrix_inf_norm(A);
  const auto [B_norm, B_has_nan] = matrix_inf_norm(B);
  const auto [C_norm, C_has_nan] = matrix_inf_norm(C_expected);
  const auto [diff_norm, diff_has_nan] = matrix_diff_inf_norm(C_computed, C_expected);

  const auto A_norm_times_B_norm = A_norm * B_norm;
  const auto relative_error = A_norm_times_B_norm == 0.0 ?
    diff_norm : (diff_norm / A_norm_times_B_norm);

  // For expected error bounds, please refer to the LAPACK Users' Guide,
  // in particular https://netlib.org/lapack/lug/node108.html .
  // Printing the infinity norm of C is a way to check
  // that both the function being tested (C_computed)
  // and the reference implementation (C_expected)
  // don't just do nothing (or fill with zeros).
  using std::cout;
  cout << "Value type of A: " << A_value_type_name << '\n'
       << std::scientific
       << "Infinity norm of A: " << A_norm << '\n'
       << "Value type of B: " << B_value_type_name << '\n'
       << "Infinity norm of B: " << B_norm << '\n'
       << "Value type of C: " << C_value_type_name << '\n'
       << "Infinity norm of C_expected: " << C_norm << '\n'
       << "Infinity norm of (C_computed - C_expected): " << diff_norm << '\n';

  if(A_norm_times_B_norm == 0.0) {
    cout << "Mollified relative error: " << relative_error << '\n';
  } else {
    cout << "Relative error: " << relative_error << '\n';
  }

  cout << "Did we encounter NaN in A? " << (A_has_nan ? "yes" : "no") << '\n' 
       << "Did we encounter NaN in B? " << (B_has_nan ? "yes" : "no") << '\n'
       << "Did we encounter NaN in C_expected? " << (C_has_nan ? "yes" : "no") << '\n'
       << "Did we encounter NaN in (C_computed - C_expected)? "
       << (diff_has_nan ? "yes" : "no") << '\n';
}

template <typename EngineType, typename LayoutType>
void
print_matrix_multiply_mollified_relative_error(
  const char value_type_name[],
  const cute::Tensor<EngineType, LayoutType>& A,
  const cute::Tensor<EngineType, LayoutType>& B,
  const cute::Tensor<EngineType, LayoutType>& C_computed,
  const cute::Tensor<EngineType, LayoutType>& C_expected)
{
  print_matrix_multiply_mollified_relative_error(value_type_name, A, value_type_name, B,
                                                 value_type_name, C_computed, C_expected);
}

// Take a CUTLASS HostTensor (or the like) as input,
// and return a const CuTe Tensor.
// This is useful for use with the above error printing functions.
// This implicitly "transposes" if the layout is RowMajor.
// Note that the HostTensor must be captured by nonconst reference
// in order for X.host_ref().data() to compile.
// (CUTLASS is a bit more container-y than CuTe.)
template<class CutlassHostTensorType>
auto host_matrix_to_const_cute_tensor(CutlassHostTensorType& X)
{
  // The tensors were created with post-transposed extents.
  const auto extents = X.extent();
  const auto shape = cute::Shape<int, int>{extents[0], extents[1]};
  // Both RowMajor and ColumnMajor only store one stride.
  const int LDX = X.stride(0);
  const auto strides = [&]() {
      using input_layout_type = typename std::decay_t<decltype(X)>::Layout;
      if constexpr (std::is_same_v<input_layout_type, cutlass::layout::ColumnMajor>) {
        return cute::Stride<int, int>{1, LDX};
      }
      else {
        static_assert(std::is_same_v<input_layout_type, cutlass::layout::RowMajor>);
        return cute::Stride<int, int>{LDX, 1};
      }
    }();
  const auto layout = cute::make_layout(shape, strides);
  auto X_data = X.host_ref().data();
  auto X_data_const = const_cast<std::add_const_t< decltype(X_data)> >(X_data);
  return cute::make_tensor(X_data_const, layout);
};
