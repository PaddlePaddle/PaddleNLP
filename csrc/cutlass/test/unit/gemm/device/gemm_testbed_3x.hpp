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

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

#include "testbed_utils.h"

#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/gemm.h"

#include "cute/int_tuple.hpp"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail{

template <typename Gemm>
struct TestbedImpl {
  // Kernel data types
  using ElementA = typename Gemm::GemmKernel::ElementA;
  using StrideA  = typename Gemm::GemmKernel::StrideA;
  using ElementB = typename Gemm::GemmKernel::ElementB;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using ElementC = typename Gemm::GemmKernel::ElementC;
  using StrideC  = typename Gemm::GemmKernel::StrideC;
  using ElementD = typename Gemm::GemmKernel::ElementD;
  using StrideD  = typename Gemm::GemmKernel::StrideD;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::CollectiveEpilogue::ElementCompute;
  using ElementScalar = typename Gemm::GemmKernel::CollectiveEpilogue::ElementScalar;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  // Looks at Cute Stride to check Row / Column Major
  template<typename Stride>
  static constexpr bool is_row_or_col_major(){
    int stride_0 = int(cute::size<0>(Stride{}));
    int stride_1 = int(cute::size<1>(Stride{}));
    int depth = cute::depth(Stride{});
    return ((stride_0 == 1) || (stride_1 == 1)) && (depth == 1);
  }

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<StrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideB>(),
    "ERROR : B Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideD>(),
    "ERROR : D Layout is neither Row / Column Major)");

  // Deduce Cutlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagA = decltype(cutlass::gemm::detail::stride_to_layout_tag_A<StrideA>());
  using LayoutTagB = decltype(cutlass::gemm::detail::stride_to_layout_tag_B<StrideB>());
  using LayoutTagC = decltype(cutlass::gemm::detail::stride_to_layout_tag_A<StrideC>());
  using LayoutTagD = decltype(cutlass::gemm::detail::stride_to_layout_tag_A<StrideD>());
  using LayoutTagPackedVector = cutlass::layout::PackedVectorLayout;

  /// Initialization
  StrideA stride_a;
  StrideB stride_b;
  StrideC stride_c;
  StrideD stride_d;
  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;
  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  cutlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  cutlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  cutlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  cutlass::HostTensor<ElementD, LayoutTagD> tensor_D;
  cutlass::HostTensor<ElementD, LayoutTagD> reference_D;
  uint32_t sm_count;

  // Used to force multi-wave tests for persistent kernel schedules
  constexpr static int MaxSmCount = 16;

  //
  // Methods
  //

  TestbedImpl(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ):
    stride_factor_A(typename LayoutTagA::Stride()),
    stride_factor_B(typename LayoutTagB::Stride()),
    stride_factor_C(typename LayoutTagC::Stride()),
    stride_factor_D(typename LayoutTagD::Stride()),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  TestbedImpl(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ):
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    stride_factor_C(stride_factor_C_),
    stride_factor_D(stride_factor_D_),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {
      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<ElementD>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      }
      else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      }
      else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      }
      else {
        scope_max = 8;
        scope_min = -8;
      }
      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }

    else if (dist_kind == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(view);
    }

    else if (dist_kind == cutlass::Distribution::Gaussian) {
      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }

    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    }

    else {
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(ProblemShapeType problem_size) {
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    stride_a = make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_b = make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_c = make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_d = make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto a_coord = cutlass::make_Coord(M * L, K);
    auto c_coord = cutlass::make_Coord(M * L, N);
    // Cutlass has Row/Col major refers to MxK times KxN matrix product, 
    // so the HostTensorB should be treated as KxN in "coord"'s view
    auto b_coord = cutlass::make_Coord(K, N * L);
    

    tensor_A.resize(a_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
    tensor_B.resize(b_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
    tensor_C.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
    tensor_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
    reference_D.resize(c_coord, cutlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2020));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);
    tensor_C.host_view().at(cutlass::make_Coord(0, 0)) = ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
      cute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta
    ) {
    auto [M, N, K, L] = problem_shape_MNKL;

    tensor_D.sync_host();
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    if (tensor_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    }

    if (reference_D.size() > 1) {
      EXPECT_GT(cutlass::reference::host::TensorNorm(reference_D.host_view()), 0);
    }

    bool passed = cutlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    EXPECT_TRUE(passed);
    if (!passed) {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << cute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << cute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".txt";

      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K << ", Batch count = " << L
        << ", alpha: " << float(alpha) << ", beta: " << float(beta) << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\n\nComputed =\n" << tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
      ProblemShapeType problem_size,
      ElementScalar alpha,
      ElementScalar beta
    ) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto M = cute::size<0>(problem_shape_MNKL);
    auto N = cute::size<1>(problem_shape_MNKL);
    auto K = cute::size<2>(problem_shape_MNKL);
    auto L = cute::size<3>(problem_shape_MNKL);

    auto A = cute::make_tensor(tensor_A.host_data(),
        cute::make_layout(cute::make_shape(M, K, L), stride_a));
    auto B = cute::make_tensor(tensor_B.host_data(),
        cute::make_layout(cute::make_shape(N, K, L), stride_b));
    auto C = cute::make_tensor(tensor_C.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), stride_c));
    auto D = cute::make_tensor(reference_D.host_data(),
        cute::make_layout(cute::make_shape(M, N, L), stride_d));
    cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

    cutlass::reference::host::GettEpilogueParams<
        ElementScalar,
        ElementAccumulator,
        ElementCompute,
        decltype(C),
        decltype(D)
        >
        epilogue_params{
          alpha, beta,
          C, D
        };

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    return compare_reference(
      problem_shape_MNKL, alpha, beta
    );
  }

	/// Determine if the CUDA device is sufficient to run the kernel
  bool sufficient() {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    int smem_size = Gemm::GemmKernel::SharedStorageSize;

    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() API call failed.");
    }

    cudaDeviceProp properties;
    result = cudaGetDeviceProperties(&properties, device_idx);
    this->sm_count = properties.multiProcessorCount;

    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      return false;
    }

    return true;
  }

  bool profile(
    ProblemShapeType problem_size,
    int iterations,
    Gemm& gemm_op,
    typename Gemm::Arguments& arguments,
    cutlass::device_memory::allocation<uint8_t>& workspace) {
    int M = cute::size<0>(problem_size);
    int N = cute::size<1>(problem_size);
    int K = cute::size<2>(problem_size);
    int L = 1;
    if constexpr(cute::rank(ProblemShapeType{}) == 4) {
      L = cute::size<3>(problem_size);
    }


    cutlass::Status status;
    //
    // Run the GEMM
    //
    cudaError_t result;

    for (int iter = 0; iter < iterations; ++iter) {
      status = gemm_op(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
        return false;
      }
    }

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
      ProblemShapeType problem_size,
      ElementScalar alpha = ElementScalar(1),
      ElementScalar beta = ElementScalar(0),
      bool profiling = false,
      int iterations = 20
    ) {
    // Fail test if insufficient CUDA device
    if (!sufficient()) {
      std::cout << "Test failed due to insufficient CUDA device." << std::endl;
      return false;
    }

    this->initialize(problem_size);

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      this->sm_count = min(MaxSmCount, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = this->sm_count;
    }
    else {
      this->sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = this->sm_count;
    }

      // DefaultEpilogue
      arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        tensor_A.device_data(),
        stride_a,
        tensor_B.device_data(),
        stride_b,
        {tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d, {alpha, beta}},
        hw_info
      };
    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);

    if (status != cutlass::Status::kSuccess) {
      cudaError_t error = cudaGetLastError();
      std::cerr << "This test is not supported: " << cudaGetErrorString(error) << "\n";
      return true;
    }

    //
    // Run the GEMM
    //

    if (profiling) {
      return profile(problem_size, iterations, gemm_op, arguments, workspace);
    }
    else {
      cudaError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = cudaDeviceSynchronize();
      if (result != cudaSuccess) {
        EXPECT_EQ(result, cudaSuccess) << "Error at Kernel Sync.";
        return false;
      }

      EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

      //
      // Verify
      //
      bool passed = this->verify(
          problem_size, alpha, beta
        );
      if (!passed) {
        std::cout << "Error : Failed : with alpha: " << float(alpha) << ", beta: " << float(beta)
                  << "\n";
      }

      return passed;
    }
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct Testbed {

  using TestBedImplementation = typename detail::TestbedImpl<Gemm>;

  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::CollectiveEpilogue::ElementCompute;
  using ElementScalar = typename Gemm::GemmKernel::CollectiveEpilogue::ElementScalar;
  using LayoutTagA = typename TestBedImplementation::LayoutTagA;
  using LayoutTagB = typename TestBedImplementation::LayoutTagB;
  using LayoutTagC = typename TestBedImplementation::LayoutTagC;
  using LayoutTagD = typename TestBedImplementation::LayoutTagD;

  // Detail Implementation
  TestBedImplementation impl_;

  //
  // Methods
  //
 Testbed(
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint64_t seed_ = TestBedImplementation::kDefaultSeed)
     : impl_(init_A_, init_B_, init_C_, seed_) {}

 Testbed(    
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
         uint64_t seed_ = TestBedImplementation::kDefaultSeed)
     : impl_(stride_factor_A_,
             stride_factor_B_,
             stride_factor_C_,
             stride_factor_D_,
             init_A_,
             init_B_,
             init_C_,
             seed_) {}

 /// Executes one test
 bool run(
  typename TestBedImplementation::ProblemShapeType problem_size,
  ElementScalar alpha = ElementScalar(1),
  ElementScalar beta = ElementScalar(0),
  bool profiling = false,
  int iterations = 20
  ) {
   return impl_.run(
       problem_size, alpha, beta, profiling, iterations
      );
 }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestAll() {
  using ElementScalar = typename Gemm::GemmKernel::CollectiveEpilogue::ElementScalar;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_m = {max_alignment, 512 - 3 * max_alignment};
  std::vector<int> problem_size_n = {max_alignment, 512 - 2 * max_alignment};

  if constexpr (std::is_same_v<typename Gemm::GemmKernel::DispatchPolicy::Schedule,
                cutlass::gemm::KernelTmaWarpSpecializedPersistent>) {
    problem_size_m.push_back(768);
    problem_size_n.push_back(768);
  }

  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = cute::size<2>(typename Gemm::GemmKernel::TileShape{});

  std::vector<int> problem_size_k = {max_alignment, TileShapeK * (Stages + 1) - max_alignment};

  Testbed<Gemm> testbed;
  bool passed = true;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        ProblemShapeType problem_size;
        if constexpr (cute::rank(ProblemShapeType{}) == 4) {
          problem_size = ProblemShapeType{m, n, k, /* l */ 1};
        }
        else {
          problem_size = ProblemShapeType{m, n, k};
        }

        passed = testbed.run(
          problem_size,
          cutlass::from_real<ElementScalar>(1),
          cutlass::from_real<ElementScalar>(0)
        );

        if (!passed) {
          return false;
        }
      }
    }
  }

  // if we do support batched GEMM, just run one test on it to save on test time
  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
    auto problem_size = ProblemShapeType{256 + max_alignment, 256 + max_alignment, 160 + max_alignment, /* l */ 3};
    passed = testbed.run(
      problem_size,
      cutlass::from_real<ElementScalar>(1),
      cutlass::from_real<ElementScalar>(0)
    );

    if (!passed) {
      return false;
    }
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Gemm>
bool TestGemmPerf(int iterations = 20) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalar = ElementAccumulator;
  bool passed = true;

  std::vector<int> problem_size_m = { 4608 };
  std::vector<int> problem_size_n = { 4608 };
  std::vector<int> problem_size_k = { 8192 };

  Testbed<Gemm> testbed;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        ProblemShapeType problem_size;
        if constexpr (cute::rank(ProblemShapeType{}) == 4) {
          problem_size = ProblemShapeType{m, n, k, /* l */ 1};
        }
        else {
          problem_size = ProblemShapeType{m, n, k};
        }

        passed = testbed.run(
          problem_size,
          cutlass::from_real<ElementScalar>(1),
          cutlass::from_real<ElementScalar>(0),
          true,
          iterations
        );

        if (!passed) {
          return false;
        }
      }
    }
  }


  // if we do support batched GEMM, just run it once
  if constexpr (cute::rank(ProblemShapeType{}) == 4) {
    auto problem_size = ProblemShapeType{problem_size_m[0], problem_size_n[0], problem_size_k[0], /* l */ 4};
    passed = testbed.run(
      problem_size,
      cutlass::from_real<ElementScalar>(1),
      cutlass::from_real<ElementScalar>(0),
      true,
      iterations
    );

    if (!passed) {
      return false;
    }
  }

  return passed;
}


} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
