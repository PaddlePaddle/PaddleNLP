/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Hopper GEMM example leveraging collective operation builders.

    This example showcases the use of CUTLASS's CollectiveBuilder to easily construct performant kernels
    targetting the NVIDIA Hopper architecture.

    Background and motivation
    -------------------------
    CUTLASS kernels are highly parameterizable via template parameters. To ease the selection of template
    parameters, CUTLASS 2 leveraged DefaultGemmConfigurations. Given a small set of parameters, such as
    the data types of operands and the compute capability of the GPU, DefaultGemmConfigurations defined sensible
    defaults for the many other parameters to the kernel (e.g., warp shape, stage count).

    However, DefaultGemmConfigurations leave multiple opportunities for improvement, which are addressed
    in CUTLASS 3:
      (1) DefaultGemmConfigurations do not allow one to use a more-performant set of parameters without
          specifying every parameter. For example, the DefaultGemmConfigurations for GEMMs targetting
          Ampere specify that three pipeline stages should be used regardless of the sizes of operands.
          If one wished to increase this value, one would also need to specify all other template parameters.
          This leaves a gap between a high-level ease-of-use interface and a lower-level detailed interface.
      (2) A new DefaultGemmConfiguration was required for each combination of operand types, GPU architecture,
          and operation type (e.g., Tensor Core or SIMT). This led to increased code size to cover each unique
          configuration and a lack of extensibility from one DefaultGemmConfiguration to another.

    Alongside these opportunities for improvement, the Hopper architecture offers new features that increase
    the number of valid configurations of a kernel. In addition to the many template parameters already available
    in CUTLASS 2 kernels, CUTLASS 3 kernels targetting Hopper also have various scheduling modes to select from that control:
      (1) how data is to be loaded (e.g., using the Hopper TMA feature or Ampere cp.async)
      (2) how work is to be divided among warps in a thread block (e.g., whether to use "warp specialization")
      (3) whether persistent thread blocks should be used
    This increased configuration space further motivates rethinking DefaultGemmConfigurations.

    Introduction to the CollectiveBuilder
    -------------------------------------
    CUTLASS 3 introduces the CollectiveBuilder to further ease the process of selecting template parameters
    for kernels targetting Hopper. Similar to the DefaultGemmConfigurations used in CUTLASS 2, the CollectiveBuilder
    takes in a small set of template parameters (e.g., the data types of operands A and B). It then automatically
    determines the data loading strategy to use depending on whether the Hopper TMA feature can be used with the provided
    parameters. If one does not indicate a particular scheduling policy or stage count to use (by using `Auto` template
    parameters), the CollectiveBuilder will also automatically select these.

    Unlike DefaultGemmConfigurations a parital specialization of the CollectiveBuilder is not needed for many
    configurations of operand types. Instead the CollectiveBuilder "builds" a configuration based on generic
    properties of the specified operands, layouts, and other parameters. For example, when the stage count
    is set to `Auto`, the CollectiveBuilder may automatically calculate the maximum number of stages that
    will fit in shared memory given the types of operands and the thread block shape, rather than simply using
    a single default value.

    Note that one does not need to use the CollectiveBuilder to declare CUTLASS 3 kernels; one can still provide
    every template parameter to the gemm::collective::CollectiveMma. Specifying every template parameter in this
    manner remains the primary API for using CUTLASS 3 kernels. The CollectiveBuilder is simply meant to be
    a convenience interface.

    Note also that, while the selections made by CollectiveBuilder attempt to maximize performance, this is not
    a guarantee. Furthermore, the behavior of the CollectiveBuilder when `Auto` parameters are provided is subject
    to change in future CUTLASS releases -- do not rely on `Auto` if you require a specific scheduling policy and/or
    stage count to be used.

    Details of this example
    -----------------------
    This example walks through the use of the CollectiveBuilder with various schedules and stage counts specified.
    This example also illustrates how CUTLASS 3 GEMMs targetting Hopper automatically support batched GEMMs by simply
    extending the problem size with an additional tensor rank.

    Example usage:
      $ ./examples/49_hopper_gemm_schedules_with_collective_builder/49_hopper_gemm_schedules_with_collective_builder \
            --m=2048 --n=2048 --k=2048 --l=2
*/

#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(2048), n(2048), k(2048), l(1),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 2048);
    cmd.get_cmd_line_argument("n", n, 2048);
    cmd.get_cmd_line_argument("k", k, 2048);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "49_hopper_gemm_schedules_with_collective_builder\n\n"
      << "  This example showcases the use of CUTLASS's collective operation builders to easily construct\n"
      << "  performant kernels targetting NVIDIA's Hopper architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

// Wrapper to construct, run, and verify a GEMM. This example showcases CUTLASS's collective
// operation builders by specializing the GEMM only on the kernel schedule it will use and the
// number of pipeline stages.
//
// For either option, one can use a special `Auto` type that tells the CollectiveBuilder
// to select an appropriate value on its own. The CollectiveBuilder will attempt to select
// values that will result in the most-performant kernel, but this is not a guarantee. Furthermore,
// the behavior of the CollectiveBuilder with `Auto` types is subject to change in future releases
// -- do not rely on `Auto` if you require a specific scheduling policy.
template <
  // Type of kernel schedule to generate
  class KernelScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
  // Number of pipeline stages to use
  class StageCountType = cutlass::gemm::collective::StageCountAuto
>
struct ExampleRunner {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  static constexpr int kAlignmentA = 8;
  static constexpr int kAlignmentB = 8;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, kAlignmentA,
      cutlass::half_t, LayoutB, kAlignmentB,
      float,
      Shape<_128,_128,_64>, Shape<_2,_1,_1>,
      StageCountType,
      KernelScheduleType
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 1, float, float>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutTagA = decltype(cutlass::gemm::detail::stride_to_layout_tag_A<StrideA>());
  using LayoutTagB = decltype(cutlass::gemm::detail::stride_to_layout_tag_B<StrideB>());
  using LayoutTagC = decltype(cutlass::gemm::detail::stride_to_layout_tag_A<StrideC>());
  using LayoutTagD = decltype(cutlass::gemm::detail::stride_to_layout_tag_A<StrideD>());

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, float alpha, float beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          typename Gemm::EpilogueOutputOp::ElementCompute(alpha),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          typename Gemm::EpilogueOutputOp::ElementCompute(beta),
          ref_C,
          ref_D,
          typename Gemm::EpilogueOutputOp::ElementAccumulator(0.f),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Reference kernel failed. Last CUDA error: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(M * K * L);
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  bool run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      block_A.get(),
      stride_A,
      block_B.get(),
      stride_B,
      {block_C.get(), stride_C, block_D.get(), stride_D, {options.alpha, options.beta}},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    // Run the GEMM
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return false;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
    }

    return passed;
  }

};

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, bool passed) {
  std::cout << description << ": " << (passed ? "Passed" : "Failed") << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (__CUDACC_VER_MAJOR__ < 12 || props.major < 9) {
    std::cout
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater) and CUDA 12.0 or greater.\n";
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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of SMs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool passed;

  // This first example constructs a GEMM using the default schedule and stage count provided by
  // the CollectiveBuilder. The scheduling policy that is expected to be most performant will be
  // selected and the maximum number of stages that can fit in shared memory will be selected.
  //
  // This example is equivalent to declaring
  //    ExampleRunner<cutlass::gemm::collective::KernelScheduleAuto, cutlass::gemm::collective::StageCountAuto>
  // Each of the `Auto` types indicate that the CollectiveBuilder should determine the scheduling policy and
  // stage count. Note that the behavior of the CollectiveBuilder with `Auto` parameters is subject to change
  // -- do not rely on `Auto` if you require a specific scheduling policy.
  ExampleRunner<> auto_schedule_auto_stage_runner;
  passed = auto_schedule_auto_stage_runner.run(options, hw_info);
  print_result("Automatically-selected schedule and stage count", passed);

  // One can override the stage count used in the GEMM by replacing cutlass::gemm::collective::StageCountAuto
  // with the number of stages to use (5 in this case).
  ExampleRunner<cutlass::gemm::collective::KernelScheduleAuto, _5> auto_schedule_5_stage_runner;
  passed = auto_schedule_5_stage_runner.run(options, hw_info);
  print_result("Automatically-selected schedule with 5 stages", passed);

  // One can also override the scheduling policy to use. In this case, use the KernelTma scheduling
  // policy, which specifies that the Hopper TMA feature should be used.
  ExampleRunner<cutlass::gemm::KernelTma> tma_schedule_auto_stage_runner;
  passed = tma_schedule_auto_stage_runner.run(options, hw_info);
  print_result("TMA schedule with automatically-selected stage count", passed);

  // Here, we override the scheduling policy to use Hopper's TMA feature alongside the warp-specialized
  // scheduling policy.
  //
  // Note that, as of the CUTLASS 3.0 release, this is the default scheduling policy
  // used by the CollectiveBuilder, so this declaration is equivalent to ExampleRunner<> and
  // ExampleRunner<cutlass::gemm::collective::KernelScheduleAuto>. However, this default is subject to
  // change in future releases -- do not rely on `Auto` if you require a specific scheduling policy.
  ExampleRunner<cutlass::gemm::KernelTmaWarpSpecialized> ws_schedule_auto_stage_runner;
  passed = ws_schedule_auto_stage_runner.run(options, hw_info);
  print_result("Warp-specialized TMA schedule with automatically-selected stage count", passed);

  // Finally, we override the scheduling policy to use Hopper's TMA feature, alongside the warp-specialized
  // scheduling policy, leveraging persistent thread blocks.
  ExampleRunner<cutlass::gemm::KernelTmaWarpSpecializedPersistent> ws_persistent_schedule_auto_stage_runner;
  passed = ws_persistent_schedule_auto_stage_runner.run(options, hw_info);
  print_result("Persistent warp-specialized TMA schedule with automatically-selected stage count", passed);

#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
