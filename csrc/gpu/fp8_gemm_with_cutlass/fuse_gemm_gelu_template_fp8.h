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

#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"

#include "paddle/extension.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/common/flags.h"

typedef struct {
  const void *A;
  const void *B;
  void *D;
  float scale = 1.0;
  void* scale_out = nullptr;
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldd;
  int batch_count = 1;
  const phi::GPUPlace &place;
  cudaStream_t stream;
  int sm_version = 89;
  float leaky_alpha = 1.0;
  const void *bias = nullptr;
  std::vector<int64_t> &bias_dims;
  std::string &fuse_gemm_config;
  int split_k = 1;
} GemmEpilogueAllParamsFP8;

typedef bool (*func_fp8)(GemmEpilogueAllParamsFP8);

template <typename InputType, typename BiasType, typename OutType, 
            typename ThreadBlockShape, typename WarpShape, 
            typename MMAShape, int Stages, bool hasbias, typename SM>
bool dispatch_fuse_gemm_gelu_fp8_noact(GemmEpilogueAllParamsFP8 params) {
  using ElementInputA = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
  using ElementInputB = typename std::conditional_t<
      std::is_same_v<InputType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;
  using ElementInputC =
      typename std::conditional_t<std::is_same_v<BiasType, phi::dtype::bfloat16>,
                                  cutlass::bfloat16_t,
                                  cutlass::half_t>;
  using ElementOutput = typename std::conditional_t<
      std::is_same_v<OutType, phi::dtype::float8_e4m3fn>,
      cutlass::float_e4m3_t,
      cutlass::float_e5m2_t>;

  using ElementAccumulator = float;
  
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

  using ElementAuxOutput = float;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    8,
    ElementAccumulator,
    ElementAccumulator,
    ScaleType
    >;
  // Number of pipelines you want to use
  constexpr int NumStages = Stages;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput, LayoutOutput,
    ElementAccumulator, MMAOp, SmArch,
    ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, NumStages,
    kAlignmentA, kAlignmentB, cutlass::arch::OpMultiplyAddFastAccum
  >;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  
  cutlass::gemm::GemmCoord problem_size =
      cutlass::gemm::GemmCoord{params.M, params.N, params.K};
  cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;

  
  typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{
      ElementCompute(params.scale),
      ElementCompute(1.0)
    };

  typename Gemm::EpilogueOutputOp::Params epilogue_params{
    activation_params,
    nullptr,
    nullptr,
    nullptr,
    static_cast<ElementAuxOutput*>(params.scale_out),
    nullptr,
    nullptr,
    nullptr
  };

  typename Gemm::Arguments arguments{
    mode,
    problem_size,
    params.batch_count,
    epilogue_params,
    reinterpret_cast<ElementInputA*>(const_cast<void*>(params.A)),
    reinterpret_cast<ElementInputB*>(const_cast<void*>(params.B)),
    reinterpret_cast<ElementInputC*>(const_cast<void*>(params.bias)),
    reinterpret_cast<ElementOutput*>(params.D),
    nullptr,
    nullptr,
    params.M * params.K,
    params.N * params.K,
    params.M * params.N,
    params.M * params.N,
    params.M, // Batch stride vector
    params.lda,
    params.ldb,
    params.ldd,
    params.ldd,
    (int64_t)0 // Leading dimension of vector. This must be 0
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

