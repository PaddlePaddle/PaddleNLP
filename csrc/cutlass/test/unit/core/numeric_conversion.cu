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
    \brief Unit tests for conversion operators.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/numeric_conversion.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace core {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Simple conversion function
template <typename Destination, typename Source, int Count>
__global__ void convert(
  cutlass::Array<Destination, Count> *destination,
  cutlass::Array<Source, Count> const *source) {

  cutlass::NumericArrayConverter<Destination, Source, Count> convert;

  *destination = convert(*source);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Destination, typename Source, int Count>
void run_test() {
  const int kN = Count;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  cutlass::HostTensor<Destination, cutlass::layout::RowMajor> destination({1, kN});
  cutlass::HostTensor<Source, cutlass::layout::RowMajor> source({1, kN});

  for (int i = 0; i < kN; ++i) {
    source.host_data()[i] = Source(i % 4);
  }

  source.sync_device();

  convert<Destination, Source, kN><<< grid, block >>>(
    reinterpret_cast<cutlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<cutlass::Array<Source, kN> const *>(source.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(float(destination.host_data()[i]) == float(source.host_data()[i]));
  }
}

} // namespace kernel
} // namespace core
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f32_to_f16_rn) {
  int const kN = 1;
  using Source = float;
  using Destination = cutlass::half_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f32x8_to_f16x8_rn) {
  int const kN = 8;
  using Source = float;
  using Destination = cutlass::half_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f16_to_f32_rn) {  
  int const kN = 1;
  using Source = cutlass::half_t;
  using Destination = float;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f16x8_to_f32x8_rn) {
  int const kN = 8;
  using Source = cutlass::half_t;
  using Destination = float;
  test::core::kernel::run_test<Destination, Source, kN>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f32_to_fe4m3_rn) {
  int const kN = 1;
  using Source = float;
  using Destination = cutlass::float_e4m3_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f32_to_fe4m3_rn_array) {
  int const kN = 27;
  using Source = float;
  using Destination = cutlass::float_e4m3_t;

  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f32_to_fe5m2_rn) {
  int const kN = 1;
  using Source = float;
  using Destination = cutlass::float_e5m2_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f32_to_fe5m2_rn_array) {
  int const kN = 27;
  using Source = float;
  using Destination = cutlass::float_e5m2_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f16_to_fe4m3_rn) {
  int const kN = 1;
  using Source = cutlass::half_t;
  using Destination = cutlass::float_e4m3_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f16_to_fe4m3_rn_array) {
  int const kN = 27;
  using Source = cutlass::half_t;
  using Destination = cutlass::float_e4m3_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f16_to_fe5m2_rn) {
  int const kN = 1;
  using Source = cutlass::half_t;
  using Destination = cutlass::float_e5m2_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, f16_to_fe5m2_rn_array) {
  int const kN = 27;
  using Source = cutlass::half_t;
  using Destination = cutlass::float_e5m2_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, bf16_to_fe4m3_rn) {
  int const kN = 1;
  using Source = cutlass::bfloat16_t;
  using Destination = cutlass::float_e4m3_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, bf16_to_fe4m3_rn_array) {
  int const kN = 27;
  using Source = cutlass::bfloat16_t;
  using Destination = cutlass::float_e4m3_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, bf16_to_fe5m2_rn) {
  int const kN = 1;
  using Source = cutlass::bfloat16_t;
  using Destination = cutlass::float_e5m2_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, bf16_to_fe5m2_rn_array) {
  int const kN = 27;
  using Source = cutlass::bfloat16_t;
  using Destination = cutlass::float_e5m2_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, fe4m3_to_fe5m2_rn) {
  int const kN = 1;
  using Source = cutlass::float_e4m3_t;
  using Destination = cutlass::float_e5m2_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe4m3_to_fe5m2_array) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  using Destination = cutlass::float_e5m2_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe5m2_to_fe4m3_rn) {
  int const kN = 1;
  using Source = cutlass::float_e5m2_t;
  using Destination = cutlass::float_e4m3_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe5m2_to_fe4m3_array) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  using Destination = cutlass::float_e4m3_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe4m3_to_f32_rn) {
  int const kN = 1;
  using Source = cutlass::float_e4m3_t;
  using Destination = float;
  test::core::kernel::run_test<Destination, Source, kN>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f32x8_to_s8x8_rn) {

  int const kN = 8;
  using Source = float;
  using Destination = int8_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe4m3_to_f32_array) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  using Destination = float;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe5m2_to_f32_array) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  using Destination = float;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe4m3_to_f16_rn) {
  int const kN = 1;
  using Source = cutlass::float_e4m3_t;
  using Destination = cutlass::half_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe4m3_to_f16_array) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  using Destination = cutlass::half_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe5m2_to_f16_rn) {
  int const kN = 1;
  using Source = cutlass::float_e5m2_t;
  using Destination = cutlass::half_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe5m2_to_f16_array) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  using Destination = cutlass::half_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe4m3_to_bf16_rn) {
  int const kN = 1;
  using Source = cutlass::float_e4m3_t;
  using Destination = cutlass::bfloat16_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe4m3_to_bf16_array) {
  int const kN = 27;
  using Source = cutlass::float_e4m3_t;
  using Destination = cutlass::bfloat16_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe5m2_to_bf16_rn) {
  int const kN = 1;
  using Source = cutlass::float_e5m2_t;
  using Destination = cutlass::bfloat16_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

TEST(NumericConversion, fe5m2_to_bf16_array) {
  int const kN = 27;
  using Source = cutlass::float_e5m2_t;
  using Destination = cutlass::bfloat16_t;
  test::core::kernel::run_test<Destination, Source, kN>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
