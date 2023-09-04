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
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Cluster 2x2x1  ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_2x2x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_2,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_2x2x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_2,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized 
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_2x2x1) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_2,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_2x2x1) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_2,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Cluster 4x1x1  ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_4x1x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_4,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_4x1x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_4,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_4x1x1) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_4,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_4x1x1) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_4,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}


///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Cluster 1x4x1  ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_1x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_4,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_1x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_4,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_1x4x1) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_4,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_1x4x1) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_1,_4,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}


///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Cluster 2x4x1  ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


TEST(SM90_Device_Gemm_f16t_f16t_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_2x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_4,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_2x4x1) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_4,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_2x4x1) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_4,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f32n_tensor_op_gmma_f32_warpspecialized, 64x128x64_2x4x1) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_64,_128,_64>, Shape<_2,_4,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelTmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
