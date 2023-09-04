/***************************************************************************************************
 * Copyright (c) 2023 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Utilities for packing a rank-X shape into a rank-(X-1) stride in CuTe.
*/

#pragma once

#include "cute/stride.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

// Strides without batch mode

template <class StrideIntT>
cute::Stride<StrideIntT, cute::Int<1>>
make_cute_packed_stride(cute::Stride<StrideIntT, cute::Int<1>> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<StrideIntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
  return s_copy;
}

template <class StrideIntT>
cute::Stride<cute::Int<1>, StrideIntT>
make_cute_packed_stride(cute::Stride<cute::Int<1>, StrideIntT> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<StrideIntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL));
  return s_copy;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Strides with batch mode

template <class StrideIntT>
cute::Stride<StrideIntT, cute::Int<1>, int64_t>
make_cute_packed_stride(cute::Stride<StrideIntT, cute::Int<1>, int64_t> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<StrideIntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
  int batch_count =  cute::get<2>(shape_MKL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL) * cute::get<1>(shape_MKL));
  }
  else {
    cute::get<2>(s_copy) = static_cast<StrideIntT>(0);
  }
  return s_copy;
}

template <class StrideIntT>
cute::Stride<cute::Int<1>, StrideIntT, int64_t>
make_cute_packed_stride(cute::Stride<cute::Int<1>, StrideIntT, int64_t> s, cute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<StrideIntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL));
  int batch_count =  cute::get<2>(shape_MKL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL) * cute::get<1>(shape_MKL));
  }
  else {
    cute::get<2>(s_copy) = static_cast<StrideIntT>(0);
  }
  return s_copy;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
