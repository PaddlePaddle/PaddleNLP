// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "helper.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_universal_streamk.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"
// #include "epilogue_tensor_op_int32.h"


#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
  }


template<class Engine, class Layout>
void print_device_tensor(cute::Tensor<Engine, Layout> const& t)
{
  // Assumes size = cosize, i.e. compact tensor
  std::vector<typename Engine::value_type> data_host(t.size());
  cutlass::device_memory::copy_to_host(data_host.data(), t.data(), t.size());
  auto t_host = cute::make_tensor(data_host.data(), t.layout());
  cute::print_tensor(t_host);
}

/// Make A structured sparse by replacing elements with 0 and compress it
template <typename ElementA_, typename ElementAcc_>
bool cutlass_sparse_compress(paddle::Tensor& a_nzs, paddle::Tensor& a_meta,
                             paddle::Tensor const& a) {
  int m = a.dims()[0];
  int k = a.dims()[1];

  // Sparse kernel setup; this kernel is not used for matmul,
  // but just for setting up the compressor utility
  // A matrix configuration
  using ElementA = ElementA_;
  using LayoutTagA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  // B matrix configuration
  using ElementB = ElementA;
  using LayoutTagB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  // C/D matrix configuration
  using ElementC = float;
  using LayoutTagC = cutlass::layout::ColumnMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  // Core kernel configurations
  using ElementAccumulator = ElementAcc_;
  using TileShape = Shape<_128, _128, _128>;
  using TileShapeRef = Shape<_128, _128, _64>;
  using ClusterShape = Shape<_1, _2, _1>;

  using KernelSchedule      = cutlass::gemm::collective::KernelScheduleAuto;        // Kernel schedule policy
  using EpilogueSchedule    = cutlass::epilogue::collective::EpilogueScheduleAuto;  // Epilogue schedule policy
  using ProblemShape = Shape<int, int, int, int>; 
  

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator, ElementC, LayoutTagC,
          AlignmentC, ElementC, LayoutTagC, AlignmentC,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassSparseTensorOp, ElementA,
          LayoutTagA, AlignmentA, ElementB, LayoutTagB, AlignmentB,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
  using StrideE = StrideA;

  using StrideA = Stride<int64_t, Int<1>, int64_t>;

  // The n (=1) dimension does not matter for the compressor
  typename GemmKernel::ProblemShape prob_shape{m, 1, k, 1};

  using LayoutA = typename GemmKernel::CollectiveMainloop::LayoutA;
  using LayoutE = typename GemmKernel::CollectiveMainloop::LayoutE;

  using ElementE = typename GemmKernel::CollectiveMainloop::ElementE;
  using SparseConfig = typename GemmKernel::CollectiveMainloop::SparseConfig;

  // Offline compressor kernel
  using CompressorUtility =
      cutlass::transform::kernel::StructuredSparseCompressorUtility<
          ProblemShape, ElementA, LayoutTagA, SparseConfig>;

  using CompressorKernel =
      cutlass::transform::kernel::StructuredSparseCompressor<
          ProblemShape, ElementA, LayoutTagA, SparseConfig,
          cutlass::arch::Sm90>;

  using Compressor =
      cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  auto [M, N, K, L] = prob_shape;

  StrideA stride_A;
  stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  

  CompressorUtility compressor_utility(prob_shape, stride_A);

  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int KC = compressor_utility.get_tensorA_k_physical();
  StrideA stride_A_compressed = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, KC, L));
  StrideE stride_E = cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(ME, KE, L));

  auto a_ptr = static_cast<ElementA*>(const_cast<void*>(a.data()));

  auto a_nzs_ptr = static_cast<ElementA*>(a_nzs.data());
  auto a_meta_ptr = static_cast<typename Gemm::CollectiveMainloop::ElementE*>(a_meta.data());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  typename Compressor::Arguments arguments{
      prob_shape, {a_ptr, stride_A, a_nzs_ptr, a_meta_ptr}, {hw_info}};

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(compressor_op.can_implement(arguments));
  CUTLASS_CHECK(compressor_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(compressor_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());
  return true;
}


std::vector<paddle::Tensor> SparseCompressor(const paddle::Tensor& w) {
    int m = w.dims()[0];
    int k = w.dims()[1];
    
    auto place = w.place();
    paddle::Tensor a_nzs = paddle::empty({m, k/2}, w.dtype(), place);
    paddle::Tensor a_meta = paddle::empty({m, k/8}, paddle::DataType::UINT8, place);
    cutlass_sparse_compress<cutlass::float_e4m3_t, float>(a_nzs, a_meta, w);
    return {a_nzs, a_meta};
}

std::vector<std::vector<int64_t>> SparseCompressorInferShape(const std::vector<int64_t>& w) {
    int64_t k = w[0];
    int64_t n = w[0];
    return {{k, n/2}, {k, n/8}};
}

std::vector<paddle::DataType> SparseCompressorInferDtype(const paddle::DataType& w) {
    return {w, paddle::DataType::UINT8};
}

PD_BUILD_OP(sparse_compressor)
    .Inputs({"w"})
    .Outputs({"a_nzs", "a_meta"})
    .SetKernelFn(PD_KERNEL(SparseCompressor))
    .SetInferShapeFn(PD_INFER_SHAPE(SparseCompressorInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SparseCompressorInferDtype));