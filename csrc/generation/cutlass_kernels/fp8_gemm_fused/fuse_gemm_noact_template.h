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

#include "fp8_fp8_gemm_scale_bias_act.h"  // NOLINT

#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

template <typename InputType, typename OutType, 
            typename ThreadBlockShape, typename WarpShape, 
            typename MMAShape, int Stages, bool hasbias, typename SM>
bool dispatch_fuse_gemm_noact(GemmEpilogueAllParams params) {
  using ElementInputA = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
  using ElementInputB = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
  using ElementOutput =
      typename std::conditional_t<std::is_same_v<OutType, phi::dtype::bfloat16>,
                                  cutlass::bfloat16_t,
                                  cutlass::half_t>;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementComputeEpilogue = float;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;
  static int const kAlignmentA = 16;
  static int const kAlignmentB = 16;

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = SM;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock = ThreadBlockShape; 
      
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = WarpShape;  
      
  // This code section describes the size of MMA op
  using ShapeMMAOp = MMAShape;  // <- MMA Op tile 

  static constexpr auto ScaleType =
              hasbias? cutlass::epilogue::thread::ScaleType::NoBetaScaling
                       : cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling;


  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;  // <- ??

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,  // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::
                value,     // <- the number of elements per vectorized
                           // memory access. For a byte, it's 16
                           // elements. This becomes the vector width of
                           // math instructions in the epilogue too
      ElementAccumulator,  // <- data type of accumulator
      ElementComputeEpilogue,
      ScaleType>;  // <- data type for alpha/beta in linear
                              // combination function

  // Number of pipelines you want to use
  constexpr int NumStages = Stages;

  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      kAlignmentA,
      kAlignmentB,
      cutlass::arch::OpMultiplyAddFastAccum>;  // NOLINT

  cutlass::gemm::GemmCoord problem_size =
      cutlass::gemm::GemmCoord{params.M, params.N, params.K};
  // cutlass::gemm::GemmUniversalMode mode =
  // cutlass::gemm::GemmUniversalMode::kGemm;

  cutlass::gemm::GemmUniversalMode mode =
      cutlass::gemm::GemmUniversalMode::kGemm;
  // cutlass::gemm::BatchedGemmCoord problem_size =
  // cutlass::gemm::BatchedGemmCoord{params.M, params.N, params.K,
  // params.batch_count};

  using EpilogueOutputOp = typename Gemm::GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(ElementCompute(params.scale),
                                                ElementCompute(1.0));
  typename Gemm::Arguments arguments{
      mode,
      problem_size,
      params.batch_count,
      epilogue_op,
      reinterpret_cast<ElementInputA*>(const_cast<void*>(params.A)),
      reinterpret_cast<ElementInputB*>(const_cast<void*>(params.B)),
      reinterpret_cast<ElementOutput*>(const_cast<void*>(params.bias)),
      reinterpret_cast<ElementOutput*>(params.D),
      params.lda * params.M,
      params.ldb * params.N,
      (int64_t)0,
      params.ldd * params.M,
      params.lda,
      params.ldb,
      (int64_t)0,
      params.ldd,
  };

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


template <typename InputType, typename OutType, 
            typename ThreadBlockShape, typename WarpShape, 
            typename MMAShape, int Stages, bool hasbias, typename SM>
bool dispatch_fuse_gemm_split_k_noact(GemmEpilogueAllParams params) {
  using ElementInputA = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
  using ElementInputB = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
  using ElementOutput =
      typename std::conditional_t<std::is_same_v<OutType, phi::dtype::bfloat16>,
                                  cutlass::bfloat16_t,
                                  cutlass::half_t>;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementComputeEpilogue = float;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;
  static int const kAlignmentA = 16;
  static int const kAlignmentB = 16;

  // This code section describes whether you want to use tensor cores or regular
  // SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = SM;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock = ThreadBlockShape; 
      
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = WarpShape;  
      
  // This code section describes the size of MMA op
  using ShapeMMAOp = MMAShape; 

  static constexpr auto ScaleType =
              hasbias? cutlass::epilogue::thread::ScaleType::NoBetaScaling
                       : cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,  // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::
                value,     // <- the number of elements per vectorized
                           // memory access. For a byte, it's 16
                           // elements. This becomes the vector width of
                           // math instructions in the epilogue too
      ElementAccumulator,  // <- data type of accumulator
      ElementComputeEpilogue,
      ScaleType>;  // <- data type for alpha/beta in linear
                              // combination function

  // Number of pipelines you want to use
  constexpr int NumStages = Stages;

    using ConvertScaledOp = cutlass::epilogue::thread::Convert<
        ElementAccumulator,
        cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassSimt, SmArch, ElementInputA, ElementInputB,
                                    ElementAccumulator,
                                    ElementAccumulator>::EpilogueOutputOp::kCount,
        ElementAccumulator>;

    /// Reduction operator
    using ReductionOp = cutlass::reduction::thread::ReduceAdd<
        ElementAccumulator, typename EpilogueOp::ElementAccumulator,
        EpilogueOp::kCount>;

    /// Threadblock-level swizzling operator
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle;

    /// Operation performed by GEMM
    using Operator = cutlass::arch::OpMultiplyAddFastAccum;

    using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
                                                        LayoutInputA,
                                                        ElementInputB,
                                                        LayoutInputB,
                                                        ElementOutput,
                                                        LayoutOutput,
                                                        ElementAccumulator,
                                                        MMAOp,
                                                        SmArch,
                                                        ShapeMMAThreadBlock,
                                                        ShapeMMAWarp,
                                                        ShapeMMAOp,
                                                        EpilogueOp,
                                                        ConvertScaledOp,
                                                        ReductionOp,
                                                        ThreadblockSwizzle,
                                                        NumStages,
                                                        kAlignmentA,
                                                        kAlignmentB,
                                                        Operator>;


  cutlass::gemm::GemmCoord problem_size =
      cutlass::gemm::GemmCoord{params.M, params.N, params.K};

  ElementComputeEpilogue alpha = ElementComputeEpilogue(params.scale);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 16 partitions
  int split_k_slices = params.split_k;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {reinterpret_cast<ElementInputA*>(const_cast<void*>(params.A)),params.lda},
                                     {reinterpret_cast<ElementInputB*>(const_cast<void*>(params.B)),params.ldb},
                                     {reinterpret_cast<ElementOutput*>(const_cast<void*>(params.bias)),0},
                                     {reinterpret_cast<ElementOutput*>(params.D),params.ldd},
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

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
