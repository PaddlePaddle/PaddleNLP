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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/epilogue.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f32, 64x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f32, 128x128x32) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_32>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f32, 64x64x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f32, 64x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f32, 128x128x32) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_32>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f32, 64x64x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f16n_tensor_op_gmma_f32, 64x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16n_f16t_f16n_tensor_op_gmma_f32, 128x128x32) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_32>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16n_f16t_f16n_tensor_op_gmma_f32, 64x64x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f16n_tensor_op_gmma_f32, 64x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16n_f16n_f16n_tensor_op_gmma_f32, 128x128x32) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_32>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16n_f16n_f16n_tensor_op_gmma_f32, 64x64x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f16, 64x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f16, 128x128x32) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_128,_128,_32>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f16n_tensor_op_gmma_f16, 64x64x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f16, 64x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f16, 128x128x32) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_128,_128,_32>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f16, 64x64x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f16n_tensor_op_gmma_f16, 64x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16n_f16t_f16n_tensor_op_gmma_f16, 128x128x32) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_128,_128,_32>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16n_f16t_f16n_tensor_op_gmma_f16, 64x64x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f16n_tensor_op_gmma_f16, 64x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16n_f16n_f16n_tensor_op_gmma_f16, 128x128x32) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_128,_128,_32>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16n_f16n_f16n_tensor_op_gmma_f16, 64x64x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f16_Epilogue, 64x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits<cutlass::half_t>::value>, Layout<Shape<_64,_128>,Stride<_1,_64>>>,
      Copy_Atom<SM90_U16x8_STSM_T, cutlass::half_t>,
      TiledCopy<Copy_Atom<DefaultCopy, cutlass::half_t>,Layout<Shape<_128,_8>,Stride<_8,_1>>,Shape<_64,_16>>,
      Copy_Atom<DefaultCopy, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f16_Epilogue, 128x64x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_128,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits<cutlass::half_t>::value>, Layout<Shape<Shape<_64,_2>,_64>,Stride<Stride<_1,_4096>,_64>>>,
      Copy_Atom<SM90_U16x8_STSM_T, cutlass::half_t>,
      TiledCopy<Copy_Atom<DefaultCopy, cutlass::half_t>,Layout<Shape<_128,_8>,Stride<_8,_1>>,Shape<_128,_8>>,
      Copy_Atom<DefaultCopy, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f16_Epilogue, 64x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits<cutlass::half_t>::value>, Layout<Shape<_64,Shape<_64,_2>>,Stride<_64,Stride<_1,_4096>>>>,
      Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>,
      TiledCopy<Copy_Atom<DefaultCopy, cutlass::half_t>,Layout<Shape<_128,_8>,Stride<_8,_1>>,Shape<_8,_128>>,
      Copy_Atom<DefaultCopy, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f16_Epilogue, 128x64x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      cutlass::half_t,
      Shape<_128,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, cutlass::half_t, cutlass::half_t>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits<cutlass::half_t>::value>, Layout<Shape<_128,_64>,Stride<_64,_1>>>,
      Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>,
      TiledCopy<Copy_Atom<DefaultCopy, cutlass::half_t>,Layout<Shape<_128,_8>,Stride<_8,_1>>,Shape<_16,_64>>,
      Copy_Atom<DefaultCopy, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f32_Epilogue, 64x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits<float>::value>, Layout<Shape<_64,_128>,Stride<_1,_64>>>,
      Copy_Atom<DefaultCopy, float>,
      TiledCopy<Copy_Atom<DefaultCopy, float>,Layout<Shape<_128,_8>,Stride<_8,_1>>,Shape<_64,_16>>,
      Copy_Atom<DefaultCopy, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16n_tensor_op_gmma_f32_Epilogue, 128x64x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits<float>::value>, Layout<Shape<Shape<_64,_2>,_64>,Stride<Stride<_1,_4096>,_64>>>,
      Copy_Atom<DefaultCopy, float>,
      TiledCopy<Copy_Atom<DefaultCopy, float>,Layout<Shape<_128,_8>,Stride<_8,_1>>,Shape<_128,_8>>,
      Copy_Atom<DefaultCopy, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f32_Epilogue, 64x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits<float>::value>, Layout<Shape<_64,Shape<_64,_2>>,Stride<_64,Stride<_1,_4096>>>>,
      Copy_Atom<DefaultCopy, float>,
      TiledCopy<Copy_Atom<DefaultCopy, float>,Layout<Shape<_128,_8>,Stride<_8,_1>>,Shape<_8,_128>>,
      Copy_Atom<DefaultCopy, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16n_f16t_tensor_op_gmma_f32_Epilogue, 128x64x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_64,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = cutlass::epilogue::collective::Epilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>,
      ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag_bits<sizeof_bits<float>::value>, Layout<Shape<_128,_64>,Stride<_64,_1>>>,
      Copy_Atom<DefaultCopy, float>,
      TiledCopy<Copy_Atom<DefaultCopy, float>,Layout<Shape<_128,_8>,Stride<_8,_1>>,Shape<_16,_64>>,
      Copy_Atom<DefaultCopy, cutlass::half_t>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      EpilogueOp
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
