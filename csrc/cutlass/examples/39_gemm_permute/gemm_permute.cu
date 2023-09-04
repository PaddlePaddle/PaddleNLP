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
    \brief GEMM Permute Example.

    This example computes batched GEMM operations with output results permuted as reshaped tensors.

    We provide layout plugin as a flexible tool for users to add any customized output tensor permute operation, 
    or any other generalized global memory writeout address computation. To add a customized layout, add new class
    in include/cutlass/layout/permute.h

    In this example, we used Tensor4DPermuteBMM0213 layout to perform Batched GEMM with permute([0, 2, 1, 3]) on BMM
    whole output tensor, and used Tensor5DPermute20314 layout to perform Normal GEMM with permute([2, 0, 3, 1, 4]) on
    output matrix. The address computations are performed in compute(col_init, row_init, stride_init, 
    BMM_batch_idx) with {col_permute, row_permute and stride_permute} as new addresses after permute op.
    (check include/cutlass/layout/permute.h)

    Tips:
    
      1) Make sure to set batch_stride_D to zero for BMM permute; Also the BMM GEMM should be in mode
      cutlass::gemm::GemmUniversalMode::kBatched instead of kArray

      2) When the last dimension is touched in permute op (for example permute([0, 2, 3, 1])), AlignmentC should 
      be set to 1. If the last dimension is untouched, one can set AlignmentC to be larger like 8 in our example.
      As a result, permute op without touching the last dimension is recommended to obtain the best performance gain.

    Examples:

      # Runs a batched GEMM with 96 batches
      $ ./examples/39_gemm_permute/39_gemm_permute --problem-count=96

      # Runs a batched GEMM with 96 batches (with GEMM-K dimension equal to 1024)
      $ ./examples/39_gemm_permute/39_gemm_permute --problem-count=96 --k=1024 --verbose=true

      # Execute batched GEMM and profile with NSight
      $ nv-nsight-cu-cli ./examples/39_gemm_permute/39_gemm_permute --m=256 --n=192 --k=256 --verbose=true --iterations=1 --reference-check=false

*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "cutlass/layout/permute.h"

/// Tensor4DPermuteBMM0213 --->
/// Permute layout function for 4-D permuted tensors for BMM with BMM output tensor (dimension as [B, M, N]) reshaped
/// as [B/D1, D1, M, N]. Then perform permute([0, 2, 1, 3]) on the corresponding whole BMM output tensor.
const int D1 = 12;

/// Tensor5DPermute20314 --->
/// Permute layout function for 5-D permuted tensors with output matrix (dimension as [M, N]) reshaped
/// as [M/T1, T1, T2, T3, N/T2/T3]. Then perform permute([2, 0, 3, 1, 4]) on the corresponding output tensor.
const int T1 = 16; 
const int T2 = 3;
const int T3 = 8;

// Alignment C
const int AlignmentC = 8;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;

  cutlass::gemm::GemmCoord problem_each;

  int batch_count;
  int iterations;
  int cuda_streams;
  bool verbose;
  float alpha;
  float beta;

  //
  // Methods
  // 

  Options():
    help(false),
    error(false),
    reference_check(true),
    batch_count(-1),
    iterations(20),
    cuda_streams(0),
    verbose(false),
    alpha(1),
    beta()
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
    cmd.get_cmd_line_argument("beta", beta, 0.0f);    
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("streams", cuda_streams, 0);
    cmd.get_cmd_line_argument("verbose", verbose, false);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);

    int m, n, k;

    cmd.get_cmd_line_argument("m", m, 128);
    cmd.get_cmd_line_argument("n", n, 192);
    cmd.get_cmd_line_argument("k", k, 128);
    cmd.get_cmd_line_argument("batch-count", batch_count, 768);

    cutlass::gemm::GemmCoord problem(m, n, k);
    problem_each = problem;

    if (batch_count % D1 != 0){
      std::cerr << "\nProblem count error (problem-count = " << batch_count << "). " 
        << "problem-count needs to be divided with no remain by " << D1 << " (D1)."
        << " (Required by the Batched GEMM permute Tensor4DPermuteBMM0213)\n\n";
      error = true;
    }

    if (m % (AlignmentC * T1) != 0){
      std::cerr << "\nProblem m size error (m = " << m << "). " 
        << "m needs to be divided with no remain by " << (AlignmentC * T1) << " (AlignmentC * T1)."
        << " (Required by the normal GEMM permute Tensor5DPermute20314)\n\n";
        error = true;
    }

    if (n % (AlignmentC * (T2 * T3)) != 0){
      std::cerr << "\nProblem n size error (n = " << n << "). " 
        << "n needs to be divided with no remain by " << (AlignmentC * (T2 * T3)) << " (AlignmentC * T2 * T3)."
        << " (Required by the normal GEMM permute Tensor5DPermute20314)\n\n";
        error = true;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "39_gemm_permute\n\n"
      << " 1) This example firstly profiles the performance of a batched GEMM kernel with BMM whole output"
      << " (including output matrices for each batch) as permuted 4D Tensor."
      << " The BMM tensor output in shape of [B, M, N] is reshaped as [B/D1, D1, M, N] and then permuted with"
      << " permute([0, 2, 1, 3]) to be in shape of [B/D1, M, D1, N].\n\n"
      << " 2) This example also profiles the performance of a normal GEMM kernel with output as permuted 5D Tensor."
      << " The GEMM matrix output in shape of [M, N]  is reshaped as [M/T1, T1, T2, T3, N/T2/T3] and then permuted"
      << " with permute([2, 0, 3, 1, 4]) to be in shape of [T2, M/T1, T3, T1, N/T2/T3].\n\n"
      << " Note: D1, T1, T2, T3 are compile-time constants defined in gemm_permute.cu\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --batch-count=<int>         Sets the number of batches in batched GEMM (batch number for BMM). (default: --batch-count=768)\n"
      << "  --m=<int>                   Sets the M dimension for both batched GEMM and normal GEMM problems. (default: --m=128)\n"
      << "  --n=<int>                   Sets the N dimension for both batched GEMM and normal GEMM problems. (default: --n=192)\n"
      << "  --k=<int>                   Sets the K dimension for both batched GEMM and normal GEMM problems. (default: --k=128)\n"
      << "  --alpha=<f32>               Epilogue scalar alpha (real part)\n"
      << "  --beta=<f32>                Epilogue scalar beta (real part)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n"
      << "  --reference-check=<bool>    If true, performs reference check.\n"
      << "  --verbose=<bool>            If true, prints problem sizes and batching structure.\n";

    out << "\n\nExamples:\n\n"

      << "# Runs a batched GEMM with 96 batches\n"
      << "$ ./examples/39_gemm_permute/39_gemm_permute --problem-count=96\n\n"

      << "# Runs a batched GEMM with 96 batches (with GEMM-K dimension equal to 1024)\n"
      << "$ ./examples/39_gemm_permute/39_gemm_permute --problem-count=96 --k=1024 --verbose=true\n\n"

      << "# Execute batched GEMM and profile with NSight\n"
      << "$ nv-nsight-cu-cli ./examples/39_gemm_permute/39_gemm_permute --m=256 --n=192 --k=256 --verbose=true --iterations=1 --reference-check=false\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = int64_t();

   fmas += problem_each.product() * batch_count;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmBatched, typename GemmPermute>
class Testbed {
public:

  //
  // Type definitions
  //

  using ElementA = typename GemmBatched::ElementA;
  using ElementB = typename GemmBatched::ElementB;
  using ElementC = typename GemmBatched::ElementC;
  using ElementAccumulator = typename GemmBatched::ElementAccumulator;

  using EpilogueOutputOp = typename GemmBatched::GemmKernel::Epilogue::OutputOp;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using LayoutA = typename GemmBatched::LayoutA;
  using LayoutB = typename GemmBatched::LayoutB;
  using LayoutC = typename GemmBatched::LayoutC;

  using MatrixCoord = typename LayoutC::TensorCoord;

private:

  //
  // Data members
  //

  Options & options;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint32_t seed;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D;

public:

  //
  // Methods
  //

  Testbed(
    Options &options_,
    cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
    cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
    uint32_t seed_ = 3090
  ):
    options(options_), init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Verbose BMM info
  void print_BMM_info_() {

    // Print batched GEMM
    std::cout << "Batched GEMM with permute([0, 2, 1, 3]) on BMM whole output tensor:\n";

    auto problem = options.problem_each;
    std::cout 
      << problem.m() << "-by-" << problem.n() << "-by-" << problem.k() 
      << ", batch count: " << options.batch_count << "\n";

    std::cout << "output tensor shape: [" << options.batch_count << ", " << problem.m() << ", "
      << problem.n() <<"]\n";
    std::cout << "reshaped as: [" << options.batch_count / D1 << ", " << D1 << ", "
      << problem.m() << ", " << problem.n() <<"]\n";
    std::cout << "finally permuted as: [" << options.batch_count / D1 << ", " << problem.m() << ", "
      << D1 << ", " << problem.n() <<"]\n";

    std::cout << "----------------------------------------------------\n";

  }

  /// Verbose normal GEMM info
  void print_GEMM_info_() {

    // Print batched GEMM
    std::cout << "Normal GEMM with permute([2, 0, 3, 1, 4]):\n";

    auto problem = options.problem_each;
    std::cout 
      << problem.m() << "-by-" << problem.n() << "-by-" << problem.k() << "\n";

    std::cout << "output tensor shape: [" << problem.m() << ", " << problem.n() <<"]" << std::endl;
    std::cout << "reshaped as: [" << problem.m() / T1 << ", " << T1 << ", "
      << T2 << ", " << T3 << ", " << problem.n() / T2 / T3 <<"]" << std::endl;
    std::cout << "finally permuted as: [" << T2 << ", " << problem.m() / T1 << ", "
      << T3 << ", " << T1 << ", " << problem.n() / T2 / T3 <<"]" << std::endl;

    std::cout << "----------------------------------------------------\n";

  }

private:

  /// Helper to initialize a tensor view
  template <typename Element>
  void initialize_tensor_(
    Element *ptr,
    size_t capacity, 
    cutlass::Distribution::Kind dist_kind,
    uint32_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      Element scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename GemmBatched::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        if (cutlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope_max = 5;
          scope_min = -5;
        }
        else {
          scope_max = 8;
          scope_min = -8;
        }
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::device::BlockFillRandomUniform(
        ptr, capacity, seed, scope_max, scope_min, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::device::BlockFillRandomGaussian(
        ptr, capacity, seed, Element(), Element(0.5f));
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {

      // Fill with increasing elements
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(1), Element());
    } 
    else {

      // Fill with all 1s
      cutlass::reference::device::BlockFillSequential(
        ptr, capacity, Element(), Element(1));
    }
  }

  /// Initializes data structures
  void initialize_(int batch_count) {

    //
    // Choose random problem sizes
    //

    // construct a few problems of random sizes
    srand(seed);

    int64_t total_elements_A = options.problem_each.m() * options.problem_each.k() * batch_count;
    int64_t total_elements_B = options.problem_each.n() * options.problem_each.k() * batch_count;
    int64_t total_elements_C = options.problem_each.m() * options.problem_each.n() * batch_count;
    int64_t total_elements_D = options.problem_each.m() * options.problem_each.n() * batch_count;

    //
    // Assign space
    //

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);

    //
    // Initialize the problems of the workspace
    //

    initialize_tensor_(block_A.get(), total_elements_A, init_A, seed * 2021);
    initialize_tensor_(block_B.get(), total_elements_B, init_B, seed * 2022);
    initialize_tensor_(block_C.get(), total_elements_C, init_C, seed * 2023);

    cutlass::reference::device::BlockFillSequential(
      block_D.get(), total_elements_D, ElementC(), ElementC());
  }

  /// Verifies the BMM GEMM result
  bool verify_BMM_() {

    bool passed = true;

    cutlass::gemm::GemmCoord problem = options.problem_each;

    LayoutA layout_A(LayoutA::packed({problem.m(), problem.k()}).stride(0));
    LayoutB layout_B(LayoutB::packed({problem.k(), problem.n()}).stride(0));
    LayoutC layout_C(LayoutC::packed({problem.m(), problem.n()}).stride(0));
    LayoutC layout_D(LayoutC::packed({problem.m(), problem.n()}).stride(0));

    MatrixCoord extent_A{problem.m(), problem.k()};
    MatrixCoord extent_B{problem.k(), problem.n()};
    MatrixCoord extent_C{problem.m(), problem.n()};
    
    cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get(), layout_A, extent_A);
    cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get(), layout_B, extent_B);
    cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get(), layout_C, extent_C);

    cutlass::DeviceAllocation<ElementC>    block_Ref(layout_D.capacity(extent_C) * options.batch_count);
    cutlass::TensorView<ElementC, LayoutC> view_Ref_device(block_Ref.get(), layout_D, extent_C);

    // Reference GEMM
    cutlass::reference::device::GemmComplex<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC, 
        ElementCompute, ElementAccumulator
    >(
      problem,
      options.alpha, 
      view_A,
      GemmBatched::kTransformA,
      view_B,
      GemmBatched::kTransformB,
      options.beta, 
      view_C, 
      view_Ref_device, 
      ElementAccumulator(0),
      options.batch_count,
      options.problem_each.m() * options.problem_each.k(),
      options.problem_each.n() * options.problem_each.k(),
      options.problem_each.m() * options.problem_each.n(),
      options.problem_each.m() * options.problem_each.n()
    );

    // Copy to host memory
    std::vector<ElementC> matrix_D(layout_D.capacity(extent_C) * options.batch_count);
    std::vector<ElementC> matrix_Ref(layout_D.capacity(extent_C) * options.batch_count);

    cutlass::device_memory::copy_to_host(matrix_D.data(), block_D.get(), matrix_D.size());
    cutlass::device_memory::copy_to_host(matrix_Ref.data(), block_Ref.get(), matrix_D.size());

    // Print out the results and reference in 4D Tensor
    // [options.batch_count, options.problem_each.m() * options.problem_each.n()] -> [D0, D1, D2, D3].
    // After permute Op, -> [D0, D2, D1, D3].
    int D0 = options.batch_count / D1;
    int D2 = options.problem_each.m();
    int D3 = options.problem_each.n();

    cutlass::TensorView<ElementC, cutlass::layout::TensorNHWC> view_D_Tensor(matrix_D.data(),   // if LayoutC = cutlass::layout::ColumnMajor, view_D_Tensor should be constructed differently
      cutlass::layout::TensorNHWC().packed(cutlass::Tensor4DCoord({D0, D2, D1, D3})), cutlass::Tensor4DCoord({D0, D2, D1, D3})); 

    cutlass::TensorView<ElementC, cutlass::layout::TensorNHWC> view_Ref_Tensor(matrix_Ref.data(), 
      cutlass::layout::TensorNHWC().packed(cutlass::Tensor4DCoord({D0, D1, D2, D3})), cutlass::Tensor4DCoord({D0, D1, D2, D3}));

    // Tensor Permute Op on reference tensor
    cutlass::HostTensor<ElementC, cutlass::layout::TensorNHWC> view_Ref_Permute_Tensor(cutlass::Tensor4DCoord({D0, D2, D1, D3}));
    for (int n = 0; n < D0; ++n) {
      for (int h = 0; h < D1; ++h) {
        for (int w = 0; w < D2; ++w) {
          for (int c = 0; c < D3; ++c) {
            view_Ref_Permute_Tensor.at({n, w, h, c}) = view_Ref_Tensor.at({n, h, w, c});
          }
        }
      }
    }

    // Reference check
    passed = cutlass::reference::host::TensorEquals(view_Ref_Permute_Tensor.host_view(), view_D_Tensor);

    if (!passed) {
      std::cerr << "\n***\nError - problem failed the QA check\n***\n" << std::endl;
      return passed;
    }

    std::cout << "Passed verification" << std::endl;
    return passed;
  }

  bool verify_GEMM_normal_() {

    bool passed = true;

    cutlass::gemm::GemmCoord problem = options.problem_each;

    LayoutA layout_A(LayoutA::packed({problem.m(), problem.k()}).stride(0));
    LayoutB layout_B(LayoutB::packed({problem.k(), problem.n()}).stride(0));
    LayoutC layout_C(LayoutC::packed({problem.m(), problem.n()}).stride(0));
    LayoutC layout_D(LayoutC::packed({problem.m(), problem.n()}).stride(0));

    MatrixCoord extent_A{problem.m(), problem.k()};
    MatrixCoord extent_B{problem.k(), problem.n()};
    MatrixCoord extent_C{problem.m(), problem.n()};
    
    cutlass::TensorView<ElementA, LayoutA> view_A(block_A.get(), layout_A, extent_A);
    cutlass::TensorView<ElementB, LayoutB> view_B(block_B.get(), layout_B, extent_B);
    cutlass::TensorView<ElementC, LayoutC> view_C(block_C.get(), layout_C, extent_C);

    cutlass::DeviceAllocation<ElementC>    block_Ref(layout_D.capacity(extent_C));
    cutlass::TensorView<ElementC, LayoutC> view_Ref_device(block_Ref.get(), layout_D, extent_C);

    // Reference GEMM
    cutlass::reference::device::GemmComplex<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC, 
        ElementCompute, ElementAccumulator
    >(
      problem,
      options.alpha, 
      view_A,
      GemmBatched::kTransformA,
      view_B,
      GemmBatched::kTransformB,
      options.beta, 
      view_C, 
      view_Ref_device, 
      ElementAccumulator(0)
    );

    // Copy to host memory
    std::vector<ElementC> matrix_D(layout_D.capacity(extent_C));
    std::vector<ElementC> matrix_Ref(layout_D.capacity(extent_C));

    cutlass::device_memory::copy_to_host(matrix_D.data(),   block_D.get(), matrix_D.size());
    cutlass::device_memory::copy_to_host(matrix_Ref.data(), block_Ref.get(),                matrix_D.size());

    // Print out the results and reference in 5D Tensor
    // [options.problem_each.m(),  options.problem_each.n()] -> [T0, T1, T2, T3, T4].
    // options.problem_each.m() == T0 * T1
    // options.problem_each.n() == T2 * T3 * T4
    // After permute Op, -> [T2, T0, T3, T1, T4].
    int T0 = options.problem_each.m() / T1;
    int T4 = options.problem_each.n() / T2 / T3;

    cutlass::TensorView<ElementC, cutlass::layout::TensorNDHWC> view_D_Tensor(matrix_D.data(),   // if LayoutC = cutlass::layout::ColumnMajor, view_D_Tensor should be constructed differently
      cutlass::layout::TensorNDHWC().packed(cutlass::Tensor5DCoord({T2, T0, T3, T1, T4})), cutlass::Tensor5DCoord({T2, T0, T3, T1, T4})); 
    cutlass::TensorView<ElementC, cutlass::layout::TensorNDHWC> view_Ref_Tensor(matrix_Ref.data(), 
      cutlass::layout::TensorNDHWC().packed(cutlass::Tensor5DCoord({T0, T1, T2, T3, T4})), cutlass::Tensor5DCoord({T0, T1, T2, T3, T4}));

    // Tensor Permute Op on reference tensor
    cutlass::HostTensor<ElementC, cutlass::layout::TensorNDHWC> view_Ref_Permute_Tensor(cutlass::Tensor5DCoord({T2, T0, T3, T1, T4}));
    for (int n = 0; n < T0; ++n) {
      for (int d = 0; d < T1; ++d) {
        for (int h = 0; h < T2; ++h) {
          for (int w = 0; w < T3; ++w) {
            for (int c = 0; c < T4; ++c) {
              view_Ref_Permute_Tensor.at({h, n, w, d, c}) = view_Ref_Tensor.at({n, d, h, w, c}); // permute([2,0,3,1,4])
            }
          }
        }
      }
    }

    // Reference check
    passed = cutlass::reference::host::TensorEquals(view_Ref_Permute_Tensor.host_view(), view_D_Tensor);

    if (!passed) {
      std::cerr << "\n***\nError - problem failed the QA check\n***\n" << std::endl;
      return passed;
    }

    std::cout << "Passed verification" << std::endl;
    return passed;
}

public:
  /// Executes a conventional batched GEMM kernel.
  Result profile_batched_kBatched() {

    std::cout << "\n====================================================" << std::endl;
    std::cout << "Batched GEMM (CUTLASS):\n"
      << "====================================================" << std::endl;
    
    if (options.verbose) {
      print_BMM_info_();
    }

    Result result;

    result.passed = false;

    // Initialize the problem
    initialize_(options.batch_count);

    // Configure the GEMM arguments
    typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

    // Please make sure all problem_sizes are the same for kBatched mode
    auto problem = options.problem_each;

    // For regular BMM
    int64_t batch_stride_C = problem.m() * problem.n();
    // For BMM permute output ---> make sure to set batch_stride_D to zero for BMM permute op
    int64_t batch_stride_D = 0;

    // Configure GEMM arguments
    typename GemmBatched::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kBatched,
      options.problem_each,
      options.batch_count,
      epilogue_op,
      (void*)block_A.get(),
      (void*)block_B.get(),
      (void*)block_C.get(),
      (void*)block_D.get(),
      problem.m() * problem.k(),
      problem.n() * problem.k(),
      batch_stride_C,
      batch_stride_D,
      problem.k(),
      problem.n(),
      problem.n(),
      problem.n()
    };

    // Initialize the GEMM object
    GemmBatched gemm;

    result.status = gemm.initialize(arguments, nullptr);

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS Batched GEMM kernel." << std::endl;
      return result;
    }

    // Run the batched GEMM object
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Batched GEMM kernel." << std::endl;
      return result;
    }

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (options.reference_check) {
      result.passed = verify_BMM_();
    }

    //
    // Warm-up run of the batched GEMM object
    //
    result.status = gemm.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Batched GEMM kernel." << std::endl;
      return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm();
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    std::cout << "    " << 1 << " batched GEMMs launched\n";

    std::cout << std::endl;
    std::cout << "    " << "Batched Runtime: " << result.runtime_ms << " ms\n";
    std::cout << "    " << "Batched  GFLOPs: " << result.gflops << "\n";

    return result;
  }

  Result profile_GEMM_permute() {

    std::cout << "\n====================================================" << std::endl;
    std::cout << "Normal GEMM (CUTLASS):\n"
      << "====================================================" << std::endl;

    if (options.verbose) {
      print_GEMM_info_();
    }

    Result result;

    result.passed = false;

    // Initialize the problem
    initialize_(1);

    // Configure the GEMM arguments
    typename EpilogueOutputOp::Params epilogue_op(options.alpha, options.beta);

    // Please make sure all problem_sizes are the same for kBatched mode
    auto problem = options.problem_each;

    // Configure GEMM arguments
    typename GemmPermute::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      options.problem_each,
      1,
      epilogue_op,
      (void*)block_A.get(),
      (void*)block_B.get(),
      (void*)block_C.get(),
      (void*)block_D.get(),
      0,
      0,
      0,
      0,
      problem.k(),
      problem.n(),
      problem.n(),
      problem.n()
    };

    // Initialize the GEMM object
    GemmPermute gemm_normal;

    result.status = gemm_normal.initialize(arguments, nullptr);

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS Batched GEMM kernel." << std::endl;
      return result;
    }

    // Run the normal GEMM object
    result.status = gemm_normal.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Batched GEMM kernel." << std::endl;
      return result;
    }

    // Wait for completion
    result.error = cudaDeviceSynchronize();

    if (result.error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error);
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (options.reference_check) {
      result.passed = verify_GEMM_normal_();
    }

    //
    // Warm-up run of the normal GEMM object
    //
    result.status = gemm_normal.run();

    if (result.status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Batched GEMM kernel." << std::endl;
      return result;
    }

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMM operations
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm_normal();
    }

    //
    // Stop profiling loop
    //

    // Record an event when the GEMM operations have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    std::cout << std::endl;
    std::cout << "    " << "Normal Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "Normal  GFLOPs: " << result.gflops << "\n";

    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  //
  // This example uses mma.sync to directly access Tensor Cores to achieve peak performance.
  //

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 11 || props.major < 8) {
  
    //
    // This example requires an NVIDIA Ampere-architecture GPU.
    //

    std::cout 
      << "CUTLASS's Grouped GEMM example requires a GPU of NVIDIA's Ampere Architecture or "
      << "later (compute capability 80 or greater).\n";

    return 0;
  }

  //
  // Parse options
  //

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Define the GEMM types
  //

  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  //
  // Define a conventional batched GEMM type
  //

  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_align8
  using GemmBatched = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, LayoutA,
    cutlass::half_t, LayoutB,
    ElementOutput,   LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 
      AlignmentC, //128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, 
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,     /*alignmentA*/
    8,     /*alignmengB*/
    cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    false,  /*GatherA*/
    false,  /*GatherB*/
    false,  /*ScatterD*/
    cutlass::layout::Tensor4DPermuteBMM0213<D1>   /*PermuteDLayout*/
  >;

  // Gemm operator cutlass_tensorop_f16_s16816gemm_f16_128x128_32x4_nt_align8
  using GemmPermute = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t, LayoutA,
    cutlass::half_t, LayoutB,
    ElementOutput,   LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 
      AlignmentC, //128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, 
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4,
    8,     /*alignmentA*/
    8,     /*alignmengB*/
    cutlass::arch::OpMultiplyAdd,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    false,  /*GatherA*/
    false,  /*GatherB*/
    false,  /*ScatterD*/
    cutlass::layout::Tensor5DPermute20314<T1, T2, T3>   /*PermuteDLayout*/
  >;

  //
  // Profile it
  //

  Testbed<GemmBatched, GemmPermute> testbed(options);

  Result result;
  result = testbed.profile_batched_kBatched();
  if (!result.passed) {
    std::cout << "Profiling batched GEMM has failed.\n";
    std::cout << "\nFailed\n";
  } else {
    std::cout << "\nPassed CUTLASS batched GEMM\n";
  }

  result = testbed.profile_GEMM_permute();
  if (!result.passed) {
    std::cout << "Profiling normal GEMM has failed.\n";
    std::cout << "\nFailed\n";
  } else {
    std::cout << "\nPassed CUTLASS normal GEMM\n";
  }

  std::cout << "\n====================================================" << std::endl;
  std::cout << "Finished\n";
  std::cout << "====================================================" << std::endl;

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
