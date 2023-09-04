#################################################################################################
#
# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

import pycutlass
from pycutlass import *
from pycutlass.test import *
from time import sleep
from bfloat16 import bfloat16
import subprocess
from typeguard import typechecked
import re



def getTensorRef(tensor, tensor_layout, conv_kind, problem_size, operand):
    ptr = tensor.__array_interface__['data'][0]
    if operand == "a":
        tensor_coord = cutlass.conv.implicit_gemm_tensor_a_extent(conv_kind, problem_size)
    elif operand == "b":
        tensor_coord = cutlass.conv.implicit_gemm_tensor_b_extent(conv_kind, problem_size)
    elif operand in ["c", "d"]:
        tensor_coord = cutlass.conv.implicit_gemm_tensor_c_extent(conv_kind, problem_size)
    else:
        raise ValueError("unknown operand: " + operand)
    
    layout = tensor_layout.packed(tensor_coord)

    if tensor.dtype == np.float64:
        return cutlass.TensorRefF64NHWC(ptr, layout)
    elif tensor.dtype == np.float32:
        return cutlass.TensorRefF32NHWC(ptr, layout)
    elif tensor.dtype == np.float16:
        return cutlass.TensorRefF16NHWC(ptr, layout)
    if tensor.dtype == bfloat16:
        return cutlass.TensorRefBF16NHWC(ptr, layout)
    elif tensor.dtype == np.int32:
        return cutlass.TensorRefS32NHWC(ptr, layout)
    elif tensor.dtype == np.int8:
        if tensor_layout == cutlass.TensorNC32HW32:
            return cutlass.TensorRefS8NC32HW32(ptr, layout)
        elif tensor_layout == cutlass.TensorC32RSK32:
            return cutlass.TensorRefS8C32RSK32(ptr, layout)
        else:
            return cutlass.TensorRefS8NHWC(ptr, layout)
    else:
        raise ValueError("unsupported data type")

def getTensorView(tensor, tensor_layout, conv_kind, problem_size, operand):
    tensor_ref = getTensorRef(tensor, tensor_layout, conv_kind, problem_size, operand)

    if operand == "a":
        tensor_coord = cutlass.conv.implicit_gemm_tensor_a_extent(conv_kind, problem_size)
    elif operand == "b":
        tensor_coord = cutlass.conv.implicit_gemm_tensor_b_extent(conv_kind, problem_size)
    elif operand in ["c", "d"]:
        tensor_coord = cutlass.conv.implicit_gemm_tensor_c_extent(conv_kind, problem_size)
    else:
        raise ValueError("unknown operand: " + operand)

    if tensor.dtype == np.float64:
        return cutlass.TensorViewF64NHWC(tensor_ref, tensor_coord)
    elif tensor.dtype == np.float32:
        return cutlass.TensorViewF32NHWC(tensor_ref, tensor_coord)
    elif tensor.dtype == np.float16:
        return cutlass.TensorViewF16NHWC(tensor_ref, tensor_coord)
    elif tensor.dtype == bfloat16:
        return cutlass.TensorViewBF16NHWC(tensor_ref, tensor_coord)
    elif tensor.dtype == np.int32:
        return cutlass.TensorViewS32NHWC(tensor_ref, tensor_coord)
    elif tensor.dtype == np.int8:
        if tensor_layout == cutlass.TensorNC32HW32:
            return cutlass.TensorViewS8NC32HW32(tensor_ref, tensor_coord)
        elif tensor_layout == cutlass.TensorC32RSK32:
            return cutlass.TensorViewS8C32RSK32(tensor_ref, tensor_coord)
        else:
            return cutlass.TensorViewS8NHWC(tensor_ref, tensor_coord)
        
    else:
        raise ValueError("unsupported data type")



# @typechecked
class Conv2dLauncher:
    """
    Launcher that runs the operation on given problem size
    """
    def __init__(self, operation: 'Conv2dOperation', seed: int=2080, interleaved=False,
        verification=True, profiling=False, warmup_iterations=500, iterations=500, **kwargs) -> None:

        self.enable_cached_results = True
        self.interleaved = interleaved

        # create the reduction kernel
        self.reduction_operation = ReductionOperation(
            shape=cutlass.MatrixCoord(4, 32 * operation.C.alignment),
            C=operation.C, element_accumulator=operation.tile_description.math_instruction.element_accumulator,
            element_compute=operation.epilogue_functor.element_epilogue, epilogue_functor=operation.epilogue_functor,
            count=operation.C.alignment
        )

        #: verify the output result
        self.verification = verification
        #: profile the kernel's runtime
        self.profiling = profiling

        self.timer = GpuTimer()

        self.warmup_iterations = warmup_iterations
        self.iterations = iterations

        if "sleep" in kwargs.keys():
            self.sleep_time = kwargs["sleep"]
        else:
            self.sleep_time = 0
        
        #
        # Compile the operator
        #

        pycutlass.compiler.add_module([operation, self.reduction_operation])

        self.operation = operation

        self.dtype_A = Conv2dLauncher.numpy_type(operation.A.element)
        self.layout_A = operation.A.layout
        self.dtype_B = Conv2dLauncher.numpy_type(operation.B.element)
        self.layout_B = operation.B.layout
        self.dtype_C = Conv2dLauncher.numpy_type(operation.C.element)
        self.layout_C = operation.C.layout
        self.dtype_D = Conv2dLauncher.numpy_type(operation.C.element)
        self.layout_D = operation.C.layout

        accumulator_size = DataTypeSize[operation.tile_description.math_instruction.element_accumulator]
        element_size = DataTypeSize[operation.A.element]

        if element_size <= 8:
            self.scope = 1
        elif element_size == 16:
            if accumulator_size <= 16:
                self.scope = 2
            else:
                self.scope = 4
        else:
            self.scope = 7

        # Seed
        self.seed = seed

        self.conv_kind = operation.conv_kind
        

        #
        # Get the host reference function
        #

        self.element_compute = operation.epilogue_functor.element_epilogue

        self.host_conv2d = cutlass.test.conv.host.conv2d

        self.timer = GpuTimer()

    @staticmethod
    def numpy_type(type):
        if type == cutlass.float64:
            return np.float64
        elif type == cutlass.float32:
            return np.float32
        elif type == cutlass.float16:
            return np.float16
        elif type == cutlass.bfloat16:
            return bfloat16
        elif type == cutlass.int32:
            return np.int32
        elif type == cutlass.int8:
            return np.int8
        else:
            raise ValueError("unsupported type: %s" % ShortDataTypeNames[type])

    def print_problem_size(self, p, split_k_mode=1):
        print("nhwc_%dx%dx%dx%d_krsc_%dx%dx%dx%d_padding_%dx%d_stride_%dx%d_dilation_%dx%d_splitkslices_%d_splitkmode_%d"
         % (p.N, p.H, p.W, p.C, p.K, p.R, p.S, p.C, p.pad_h,
          p.pad_w, p.stride_h, p.stride_w, p.dilation_h, p.dilation_w, p.split_k_slices, split_k_mode))
    
    def uniform_init(self, size, dtype):
        if dtype in [np.float32, np.float16, bfloat16, np.float64]:
            return np.ceil(
                np.random.uniform(
                    low=-self.scope - 0.5, high=self.scope - 0.5, 
                    size=size).astype(dtype)
                )
        else:
            return np.random.uniform(
                low=-self.scope - 1, high=self.scope + 1, 
                size=size).astype(dtype)
    
    def eq_gemm_size(self, problem_size):
        n = problem_size.N
        p = problem_size.P
        q = problem_size.Q
        k = problem_size.K
        r = problem_size.R
        s = problem_size.S
        c = problem_size.C
        h = problem_size.H
        w = problem_size.W
        if self.conv_kind == cutlass.conv.Operator.fprop:
            return cutlass.gemm.GemmCoord(n * p * q, k, r * s * c)
        elif self.conv_kind == cutlass.conv.Operator.dgrad:
            return cutlass.gemm.GemmCoord(n * h * w, c, k * r * s)
        else:
            return cutlass.gemm.GemmCoord(k, r * s * c, n * p * q)
    
    def bytes(self, problem_size, alpha, beta):
        mnk = self.eq_gemm_size(problem_size)

        bytes_ = \
            (DataTypeSize[self.operation.A.element] * mnk.m() // 8) * mnk.k() + \
            (DataTypeSize[self.operation.B.element] * mnk.n() // 8) * mnk.k() + \
            (DataTypeSize[self.operation.C.element] * mnk.m() // 8) * mnk.n()

        if beta != 0:
            bytes_ += (DataTypeSize[self.operation.C.element] * mnk.m() // 8) * mnk.n()
        
        return bytes_
    
    def flops(self, problem_size):
        mnk = self.eq_gemm_size(problem_size)

        flops_mainloop_ = mnk.m() * mnk.n() * mnk.k() * 2
        flops_epilogue_ = mnk.m() * mnk.n() * 2

        # Adjust mainloop flop for dgrad stride
        if self.conv_kind == cutlass.conv.Operator.dgrad:
            flops_mainloop_ = flops_mainloop_ // (problem_size.stride_h * problem_size.stride_w)
        
        flops_total_ = flops_mainloop_ + flops_epilogue_
        
        return flops_total_


    
    def host_reference(self, problem_size, tensor_A, tensor_B, tensor_C, alpha, beta):
        if self.element_compute == cutlass.float16:
            alpha = cutlass.float16(alpha)
            beta = cutlass.float16(beta)
        elif self.element_compute == cutlass.int32:
            alpha = int(alpha)
            beta = int(beta)
        else:
            alpha = alpha
            beta = beta

        # if cached result is loaded
        cached_result_loaded = False

        if self.enable_cached_results:
            # get problem key
            cached_test_key = cutlass.test.conv.host.CreateCachedConv2dTestKey(
                self.conv_kind, problem_size, alpha, beta, 
                getTensorView(tensor_A, self.layout_A, self.conv_kind, problem_size, "a"),
                getTensorView(tensor_B, self.layout_B, self.conv_kind, problem_size, "b"),
                getTensorView(tensor_C, self.layout_C, self.conv_kind, problem_size, "c"),
            )

            cached_test_result = cutlass.test.conv.host.CachedTestResult()

            conv2d_result_cache_name = "cached_results_SM%d_%d.txt" % (self.operation.arch, self.seed)

            cached_results = cutlass.test.conv.host.CachedTestResultListing(conv2d_result_cache_name)
            # CachedTestResultListing cached_results(conv2d_result_cache_name);
            cached = cached_results.find(cached_test_key)
            cached_result_loaded = cached[0]
            if cached_result_loaded :
                cached_test_result = cached[1]
        
        if not cached_result_loaded:
            # compute the conv2d on host
            tensor_D_ref = np.ones_like(tensor_C)
            tensor_ref_A = getTensorRef(tensor_A, self.layout_A, self.conv_kind, problem_size, "a")
            tensor_ref_B = getTensorRef(tensor_B, self.layout_B, self.conv_kind, problem_size, "b")
            tensor_ref_C = getTensorRef(tensor_C, self.layout_C, self.conv_kind, problem_size, "c")
            tensor_ref_D_ref = getTensorRef(tensor_D_ref, self.layout_D, self.conv_kind, problem_size, "d")

            self.host_conv2d(
                self.conv_kind, problem_size, 
                tensor_ref_A, tensor_ref_B, tensor_ref_C, tensor_ref_D_ref,
                alpha, beta
            )

            tensor_view_D_ref = getTensorView(tensor_D_ref, self.layout_D, self.conv_kind, problem_size, "d")

            if self.enable_cached_results:
                cached_test_result.D = cutlass.test.conv.host.TensorHash(tensor_view_D_ref)
                cached_results = cutlass.test.conv.host.CachedTestResultListing(conv2d_result_cache_name)
                cached_results.append(cached_test_key, cached_test_result)
                cached_results.write(conv2d_result_cache_name)
            else:
                return tensor_D_ref

        return cached_test_result.D
    
    def equal(self, tensor_D, tensor_D_ref, problem_size):
        if self.enable_cached_results:
            tensor_view_D = getTensorView(tensor_D, self.layout_D, self.conv_kind, problem_size, "d")
            tensor_D_hash = cutlass.test.conv.host.TensorHash(tensor_view_D)

            return tensor_D_hash == tensor_D_ref
        else:
            tensor_view_D = getTensorView(tensor_D, self.layout_D, self.conv_kind, problem_size, "d")
            tensor_view_D_ref = getTensorView(tensor_D_ref, self.layout_D, self.conv_kind, problem_size, "d")
            return cutlass.test.conv.host.equals(tensor_view_D, tensor_view_D_ref)
    
    def run_cutlass_profiler(self, problem_size, split_k_mode=cutlass.conv.SplitKMode.Serial, alpha=1.0, beta=0.0):

        if split_k_mode == cutlass.conv.SplitKMode.Serial:
            split_k_mode_ = "serial"
        else:
            split_k_mode_ = "parallel"

        cutlass_path = os.getenv('CUTLASS_PATH')
        assert cutlass_path is not None, "Environment variable 'CUTLASS_PATH' is not defined."

        values = {
            "profiler_path": cutlass_path + "/build/tools/profiler/cutlass_profiler",
            "kernel_name": self.operation.procedural_name(),
            "verification_providers": "device",
            "provider": "cutlass",
            'n': str(problem_size.N),
            'h': str(problem_size.H),
            'w': str(problem_size.W),
            'c': str(problem_size.C),
            'k': str(problem_size.K),
            'r': str(problem_size.R),
            's': str(problem_size.S),
            'p': str(problem_size.P),
            'q': str(problem_size.Q),
            'pad_h': str(problem_size.pad_h),
            'pad_w': str(problem_size.pad_w),
            'stride_h': str(problem_size.stride_h),
            'stride_w': str(problem_size.stride_w),
            'dilation_h': str(problem_size.dilation_h),
            'dilation_w': str(problem_size.dilation_w),
            'split_k_slices': str(problem_size.split_k_slices),
            'split_k_mode': split_k_mode_,
            'alpha': str(alpha),
            'beta': str(beta),
            'warmup': str(self.warmup_iterations),
            'profile': str(self.iterations)
        }

        cmd_template = \
            "${profiler_path} --kernels=${kernel_name} --verification-providers=${verification_providers}" \
            " --providers=${provider} --n=${n} --h=${h} --w=${w} --c=${c} --k=${k} --r=${r} --s=${s} --p=${p}" \
            " --q=${q} --pad_h=${pad_h} --pad_w=${pad_w} --stride_h={stride_h} --stride_w=${stride_w}" \
            " --dilation_h=${dilation_h} --dilation_w=${dilation_w} --warmup-iterations=${warmup} --profiling-iterations=${profile}" \
            " --split_k_slices=${split_k_slices} --alpha=${alpha} --beta=${beta} --split_k_mode=${split_k_mode}"
        
        cmd = SubstituteTemplate(cmd_template, values)
        result = subprocess.getoutput(cmd)

        m = re.search(r"Runtime:\s+(?P<runtime>\d+.\d+)", result)
        runtime = float(m.group('runtime'))

        m = re.search(r"Bytes:\s+(?P<bytes>\d+)", result)
        bytes = int(m.group('bytes'))

        m = re.search(r"FLOPs:\s+(?P<flops>\d+)", result)
        flops = int(m.group('flops'))

        # check if the problem size matches
        assert bytes == self.bytes(problem_size, alpha, beta)
        assert flops == self.flops(problem_size)

        return runtime



    def run(self, problem_size, split_k_mode=cutlass.conv.SplitKMode.Serial,
        alpha=1.0, beta=0.0):

        assert get_allocated_size() == 0, "%d byte of pool memory is not released in previous run" % get_allocated_size()

        #
        # Initialize input and output tensors
        #
        tensor_A_size = cutlass.conv.implicit_gemm_tensor_a_size(self.conv_kind, problem_size)
        tensor_B_size = cutlass.conv.implicit_gemm_tensor_b_size(self.conv_kind, problem_size)
        tensor_C_size = cutlass.conv.implicit_gemm_tensor_c_size(self.conv_kind, problem_size)
        
        np.random.seed(self.seed)

        tensor_A = self.uniform_init(size=(tensor_A_size,), dtype=self.dtype_A)
        tensor_B = self.uniform_init(size=(tensor_B_size,), dtype=self.dtype_B)
        tensor_C = self.uniform_init(size=(tensor_C_size,), dtype=self.dtype_C)
        tensor_D = np.zeros(shape=(tensor_C_size,), dtype=self.dtype_D)
        

        #
        # Launch kernel
        #

        arguments = Conv2dArguments(
            operation=self.operation, problem_size=problem_size, A=tensor_A,
            B=tensor_B, C=tensor_C, D=tensor_D, 
            output_op = self.operation.epilogue_type(alpha, beta), 
            split_k_slices=problem_size.split_k_slices,
            split_k_mode=split_k_mode
        )

        if split_k_mode == cutlass.conv.SplitKMode.Parallel:
            implicit_gemm_size = cutlass.conv.implicit_gemm_problem_size(self.operation.conv_kind, arguments.problem_size)
            reduction_arguments = ReductionArguments(
                self.reduction_operation,
                problem_size=[implicit_gemm_size.m(), implicit_gemm_size.n()], partitions=problem_size.split_k_slices,
                workspace=arguments.ptr_D,
                destination=tensor_D,
                source=tensor_C,
                output_op = self.reduction_operation.epilogue_type(alpha, beta)
            )

        self.operation.run(arguments)
        if split_k_mode == cutlass.conv.SplitKMode.Parallel:
            self.reduction_operation.run(reduction_arguments)
        
        passed = True
        if self.verification:
            if split_k_mode == cutlass.conv.SplitKMode.Parallel:
                reduction_arguments.sync()
            else:
                arguments.sync()

            tensor_D_ref = self.host_reference(problem_size, tensor_A, tensor_B, tensor_C, alpha, beta)
            
            passed = self.equal(tensor_D, tensor_D_ref, problem_size)

            try: 
                assert passed
            except AssertionError:
                self.print_problem_size(problem_size, split_k_mode)
        
        if self.profiling:
            sleep(self.sleep_time)
            for _ in range(self.warmup_iterations):
                self.operation.run(arguments)
                if split_k_mode == cutlass.conv.SplitKMode.Parallel:
                    self.reduction_operation.run(reduction_arguments)
            
            self.timer.start()
            for _ in range(self.warmup_iterations):
                self.operation.run(arguments)
                if split_k_mode == cutlass.conv.SplitKMode.Parallel:
                    self.reduction_operation.run(reduction_arguments)
            self.timer.stop_and_wait()
            runtime = self.timer.duration(self.iterations)
        
        # free memory
        del arguments
        if split_k_mode == cutlass.conv.SplitKMode.Parallel:
            del reduction_arguments
        
        assert get_allocated_size() == 0, "%d byte of pool memory is not released after current run" % get_allocated_size()
        if self.profiling:
            return runtime
        return passed



########################################################################################################
# TestAllConv: Runs cutlass::conv::device::ImplicitGemmConvolution operator and compares it with reference
# TestAllConv runs conv operator on default conv problem sizes from test::conv::device::TestbedConv2dProblemSizes
# Additionaly, each conv2d test can provide conv problem sizes (conv_test_sizes) and blacklist of sizes 
# (conv_blacklist_sizes)
############################################################################################################

def test_all_conv2d(operation: Conv2dOperation, conv_test_sizes = [], interleaved=False):
    passed = True
    #
    # Testbed object
    #

    testbed = Conv2dLauncher(operation, interleaved=interleaved)

    #
    # Get conv problem sizes to run conv operator
    #

    conv_problems = cutlass.test.conv.TestbedConv2dProblemSizes(64)

    # Vector of conv2d problem sizes to avoid duplicate runs
    conv_tested_sizes = []

    # Flatten 2D problem_vectors into a 1D problem sizes
    problem_sizes = conv_problems.conv2d_default_sizes
    
    problem_sizes = [conv_problem for conv_problem in problem_sizes] + conv_test_sizes

    # Sweep conv2d problem sizes (split-k-mode=kSerial, split-k-slices=1, alpha=1.0, beta=0.0)
    for conv_problem in problem_sizes:

        if conv_problem in conv_tested_sizes:
            continue
            
        # skip channel dimension % 32 != 0 for interleaved case
        if interleaved:
            if conv_problem.K % 32 != 0 or conv_problem.C % 32 != 0:
                continue
    
        #
        # Procedurally disable certain cases
        #

        # CUTLASS DGRAD's *unity* stride specialization only support stride {1, 1} 
        if operation.conv_kind == cutlass.conv.Operator.dgrad and operation.stride_support == StrideSupport.Unity:
            if not ((conv_problem.stride_h == 1) and (conv_problem.stride_w == 1)):
                continue
        
        if not interleaved:
            # Fixed channels algorithm requires channel count to match access size
            if operation.iterator_algorithm == cutlass.conv.IteratorAlgorithm.fixed_channels:
                if conv_problem.C != operation.A.alignment:
                    continue
            
            # Few channels algorithm requires channel count to match access size
            if operation.iterator_algorithm == cutlass.conv.IteratorAlgorithm.few_channels:
                if conv_problem.C % operation.A.alignment:
                    continue
            
            # CUTLASS DGRAD's *strided* stride specialization supports all stride {stride_h, stride_w} 
            # Although strided dgrad works for all stride combinations, we are only going 
            # to run strided dgrad for non-unity strides 

            if operation.conv_kind == cutlass.conv.Operator.dgrad and operation.stride_support == StrideSupport.Strided:
                if (conv_problem.stride_h == 1) and (conv_problem.stride_w == 1):
                    continue
            
        #
        # Test
        #

        # push back tested problem size to avoid re-running duplicates
        conv_tested_sizes.append(conv_problem)

        passed = testbed.run(conv_problem)

        if not passed:
            return False

    if interleaved:
        return True
    #
    # filter the cases for split K
    #

    # Small-channels convolution can't run here.
    if operation.iterator_algorithm in [cutlass.conv.IteratorAlgorithm.fixed_channels, cutlass.conv.IteratorAlgorithm.few_channels]:
        return True
    
    # CUTLASS DGRAD's *stride* specialization does not support split-k mode
    if operation.conv_kind == cutlass.conv.Operator.dgrad and operation.stride_support == StrideSupport.Strided:
        conv_problem = cutlass.conv.Conv2dProblemSize(
            cutlass.Tensor4DCoord(1, 56, 56, 8),
            cutlass.Tensor4DCoord(8, 1, 1, 8),
            cutlass.Tensor4DCoord(0, 0, 0, 0),
            cutlass.MatrixCoord(2, 2),
            cutlass.MatrixCoord(1, 1),
            cutlass.conv.Mode.cross_correlation,
            1, 1
        )
        passed = testbed.run(conv_problem)

        return passed
    
    # Sweep split-k-slice using serial and prallel reduction with non-unity alpha and non-zero beta for 
    # a single conv2d problem size. Convolution unit tests take a long time to run so only sweep parameters 
    # which are abolutely neccessary to catch functional bugs. The below code does provide option to sweep 
    # alpha and beta for local testing, but only runs one value for alpha and beta.

    conv2d_split_k_test_size = cutlass.conv.Conv2dProblemSize(
        cutlass.Tensor4DCoord(1, 17, 11, 288),
        cutlass.Tensor4DCoord(160, 3, 3, 288),
        cutlass.Tensor4DCoord(1, 1, 1, 1),
        cutlass.MatrixCoord(1, 1),
        cutlass.MatrixCoord(1, 1),
        cutlass.conv.Mode.cross_correlation,
        1, 1
    )

    split_k_modes = [cutlass.conv.SplitKMode.Parallel, cutlass.conv.SplitKMode.Serial]

    split_k_slices = [1, 2, 3, 4, 201]
    problem_alpha = [2.0,]
    problem_beta = [2.0,]

    for split_k_mode in split_k_modes:
        for split_k_slice in split_k_slices:
            for alpha in problem_alpha:
                for beta in problem_beta:
                    passed = testbed.run(conv2d_split_k_test_size.reset_split_k_slices(split_k_slice),
                    split_k_mode,
                    alpha, beta)
                
    return passed
