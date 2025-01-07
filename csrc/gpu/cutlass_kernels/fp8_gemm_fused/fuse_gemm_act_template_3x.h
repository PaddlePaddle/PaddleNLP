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
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

template <
  typename InputType = phi::dtype::float8_e4m3fn,
  typename OutType = phi::dtype::float16,
  bool hasbias = false,
  template <class> typename Activation = cutlass::epilogue::thread::Identity,
  typename TileShape = Shape<_128, _128, _128>,
  typename ClusterShape = Shape<_2, _1, _1>,
  typename KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
  typename EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized,
  typename SM = cutlass::arch::Sm90
>
bool dispatch_fuse_gemm_act_sm90(GemmEpilogueAllParams params){
  using ElementA = typename std::conditional_t<std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
                                                              cutlass::float_e4m3_t,
                                                              cutlass::float_e5m2_t>;
  using ElementB = ElementA;
  using ElementD = typename std::conditional_t<std::is_same_v<OutType, phi::dtype::bfloat16>,
                                  cutlass::bfloat16_t,
                                  cutlass::half_t>;
  using ElementC = std::conditional_t<
        hasbias,
        ElementD,
        void>;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;

  // 16B alignment lets us use TMA
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = hasbias ? 16 / sizeof(ElementC) : 8;
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  using FusionOperation = cutlass::epilogue::fusion::LinCombEltAct<Activation, ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      SM, cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      SM, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A{params.lda, cute::Int<1>{}, 0};
  StrideB stride_B{params.ldb, cute::Int<1>{}, 0};
  StrideC stride_C{0, cute::Int<1>{}, 0};
  StrideD stride_D{params.ldd, cute::Int<1>{}, 0};

  auto a_ptr = reinterpret_cast<ElementA*>(const_cast<void*>(params.A));
  auto b_ptr = reinterpret_cast<ElementB*>(const_cast<void*>(params.B));
  auto c_ptr = reinterpret_cast<ElementC*>(const_cast<void*>(params.bias));
  auto d_ptr = reinterpret_cast<ElementD*>(params.D);

  ProblemShapeType problem_size = ProblemShapeType{params.M, params.N, params.K, 1};

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    {a_ptr, stride_A, b_ptr, stride_B},
    {{params.scale}, // epilogue.thread
      c_ptr, stride_C, d_ptr, stride_D}
  };
  if constexpr (hasbias){
    arguments.epilogue.thread.beta = 1.0;
  }
  

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cout << "Gemm::can_implement() failed. " << cutlassGetStatusString(status) << std::endl;
    return false;
  }
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  phi::Allocator* allocator = paddle::GetAllocator(params.place);
  auto workspace = allocator->Allocate(workspace_size);

  status = gemm_op(arguments, workspace->ptr(), params.stream);
  if (status != cutlass::Status::kSuccess) {
    std::cout << "Gemm::run() failed." << cutlassGetStatusString(status) << std::endl;
    return false;
  }
  return true;
}
