// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "fp8_common.h"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass_kernels/gemm/collective/collective_builder_gated.hpp"
#include "cutlass_kernels/gemm/kernel/gemm_universal_gated.hpp"

using namespace cute;

template <typename InputType, typename CTAShape, typename ClusterShape,
    typename MainloopScheduleType, typename EpilogueScheduleType, typename TileSchedulerType = void,
    template <class /* ElementCompute */> class Activation = cutlass::epilogue::thread::SiLu, bool SwapAB = true>
bool dispatch_dual_gemm_act_sm90(DualGemmEpilogueAllParams params) {
  using ElementA = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
    using LayoutA = cutlass::layout::RowMajor;         // Layout type for A matrix operand
    static constexpr int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A
                                                       // matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = ElementA;                      // Element type for B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;      // Layout type for B matrix operand
    static constexpr int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B
                                                       // matrix in units of elements (up to 16 bytes)

    using ElementC = ElementA; // Element type for C matrix operands

    using LayoutC = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    static constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C matrices in units of
                                                       // elements (up to 16 bytes)

    // Output matrix configuration
    using ElementOutput = ElementA; // Element type for output matrix operands
    // using LayoutOutput = cutlass::layout::RowMajor; // Layout type for output matrix operands
    using LayoutOutput = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    // Multiply-accumulate blocking/pipelining details
    using ElementAccumulator = float; // Element type for internal accumulation
    using ElementCompute = float;                // Element type for compute
    using ArchTag = cutlass::arch::Sm90;         // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
    using TileShape = CTAShape;                           // Threadblock-level tile size
    using KernelSchedule = MainloopScheduleType;
    using EpilogueSchedule = EpilogueScheduleType;
    using TileScheduler = TileSchedulerType;

    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using FusionOperation = cutlass::epilogue::fusion::ScaledAcc<ElementOutput, ElementCompute>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
        TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator, ElementC, LayoutC,
        AlignmentC, ElementOutput, LayoutOutput, AlignmentOutput, EpilogueSchedule, FusionOperation>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderGated<ArchTag, OperatorClass,
        ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule, Activation, SwapAB>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversalGated<Shape<int, int, int, int>, // Indicates ProblemShape
        CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    int arg_m = params.M;
    int arg_n = params.N;
    ElementA const* ptr_A = reinterpret_cast<ElementA const*>(params.A);
    ElementB const* ptr_B0 = reinterpret_cast<ElementB const*>(params.B0);
    ElementB const* ptr_B1 = reinterpret_cast<ElementB const*>(params.B1);
    if constexpr (SwapAB)
    {
        arg_m = params.N;
        arg_n = params.M;
        ptr_A = reinterpret_cast<ElementB const*>(params.B0);
        ptr_B0 = reinterpret_cast<ElementA const*>(params.A);
    }
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(arg_m, params.K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(arg_n, params.K, 1));
    StrideC stride_C;
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(arg_m, arg_n, 1));

    typename Gemm::Arguments arguments = {cutlass::gemm::GemmUniversalMode::kGemm, {arg_m, arg_n, params.K, 1},
        {ptr_A, stride_A, ptr_B0, ptr_B1, stride_B, params.scale0, params.scale1},
        {{}, // epilogue.thread
            nullptr, stride_C, reinterpret_cast<ElementOutput*>(params.D), stride_D}};
    arguments.epilogue.thread.alpha = params.scale_out;

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::can_implement() failed" << std::endl;
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  phi::Allocator* allocator = paddle::GetAllocator(params.place);
  auto workspace = allocator->Allocate(workspace_size);

  //
  // Run the GEMM
  //
  status = gemm_op(arguments, workspace->ptr(), params.stream);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::run() failed" << std::endl;
    return false;
  }
  return true;
}

