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

from time import sleep
import pycutlass
from pycutlass import *
import pycutlass.utils.datatypes as datatypes
import cutlass
from cuda import cudart
from cuda import cuda
from bfloat16 import bfloat16
from .profiler import GpuTimer
import subprocess


def transpose(layout):
    if layout == cutlass.RowMajor:
        return cutlass.ColumnMajor
    elif layout == cutlass.ColumnMajor:
        return cutlass.RowMajor
    elif layout == cutlass.ColumnMajorInterleaved32:
        return cutlass.RowMajorInterleaved32
    elif layout == cutlass.RowMajorInterleaved32:
        return cutlass.ColumnMajorInterleaved32


def getTensorRef(tensor: np.ndarray, problem_size: cutlass.gemm.GemmCoord, operand: str, layout: cutlass.layout, batch_offset: int = 0):
    ptr = tensor.__array_interface__['data'][0]
    if operand == "a":
        tensor_coord = problem_size.mk()
        batch_stride = problem_size.m() * problem_size.k()
    elif operand == "b":
        tensor_coord = problem_size.kn()
        batch_stride = problem_size.k() * problem_size.n()
    elif operand in ["c", "d"]:
        tensor_coord = problem_size.mn()
        batch_stride = problem_size.m() * problem_size.n()
    else:
        raise ValueError("Unknown operand: " + operand)

    elt_size = DataTypeSizeBytes[datatypes.to_cutlass(tensor.dtype)]
    ptr += batch_offset * batch_stride * elt_size

    if layout == cutlass.RowMajor:
        layout = cutlass.RowMajor.packed(tensor_coord)
        layout_tag = "RowMajor"
    elif layout == cutlass.ColumnMajor:
        layout = cutlass.ColumnMajor.packed(tensor_coord)
        layout_tag = "ColumnMajor"
    elif layout == cutlass.ColumnMajorInterleaved32:
        layout = cutlass.ColumnMajorInterleaved32.packed(tensor_coord)
        layout_tag = "ColumnMajorInterleaved32"
    elif layout == cutlass.RowMajorInterleaved32:
        layout = cutlass.RowMajorInterleaved32.packed(tensor_coord)
        layout_tag = "RowMajorInterleaved32"
    else:
        raise ValueError("unsupported layout")
    if tensor.dtype == np.float32:
        ref_name = "TensorRefF32" + layout_tag
    elif tensor.dtype == np.float64:
        ref_name = "TensorRefF64" + layout_tag
    elif tensor.dtype == np.float16:
        ref_name = "TensorRefF16" + layout_tag
    elif tensor.dtype == bfloat16:
        ref_name = "TensorRefBF16" + layout_tag
    elif tensor.dtype == np.int8:
        ref_name = "TensorRefS8" + layout_tag
    elif tensor.dtype == np.int32:
        ref_name = "TensorRefS32" + layout_tag
    else:
        raise ValueError("unsupported datatype %s" %
                         ShortDataTypeNames[tensor.dtype])

    return getattr(cutlass, ref_name)(ptr, layout)


def getTensorView(tensor: np.ndarray, problem_size: cutlass.gemm.GemmCoord, operand: str, layout: str, batch_offset: int = 0):
    tensor_ref = getTensorRef(tensor, problem_size, operand, layout, batch_offset)

    if operand == "a":
        tensor_coord = problem_size.mk()
    elif operand == "b":
        tensor_coord = problem_size.kn()
    elif operand in ["c", "d"]:
        tensor_coord = problem_size.mn()
    else:
        raise ValueError("Unknown operand: " + operand)

    if layout == cutlass.RowMajor:
        layout_tag = "RowMajor"
    elif layout == cutlass.ColumnMajor:
        layout_tag = "ColumnMajor"
    elif layout == cutlass.ColumnMajorInterleaved32:
        layout_tag = "ColumnMajorInterleaved32"
    elif layout == cutlass.RowMajorInterleaved32:
        layout_tag = "RowMajorInterleaved32"
    else:
        raise ValueError("unsupported layout")
    if tensor.dtype == np.float32:
        ref_name = "TensorViewF32" + layout_tag
    elif tensor.dtype == np.float64:
        ref_name = "TensorViewF64" + layout_tag
    elif tensor.dtype == np.float16:
        ref_name = "TensorViewF16" + layout_tag
    elif tensor.dtype == bfloat16:
        ref_name = "TensorViewBF16" + layout_tag
    elif tensor.dtype == np.int32:
        ref_name = "TensorViewS32" + layout_tag
    elif tensor.dtype == np.int8:
        ref_name = "TensorViewS8" + layout_tag
    else:
        raise ValueError("unsupported datatype")

    return getattr(cutlass, ref_name)(tensor_ref, tensor_coord)


class GemmUniversalLauncher:
    def __init__(self, operation: 'GemmOperationUniversal', seed: int = 2080, interleaved=False,
                 verification=True, profiling=False, warmup_iterations=500, iterations=500, **kwargs) -> None:
        # create the reduction kernel
        self.reduction_operation: ReductionOperation = ReductionOperation(
            shape=cutlass.MatrixCoord(4, 32 * operation.C.alignment),
            C=operation.C, element_accumulator=operation.tile_description.math_instruction.element_accumulator,
            element_compute=operation.epilogue_functor.element_epilogue, epilogue_functor=operation.epilogue_functor,
            count=operation.C.alignment
        )

        self.math_operation = operation.tile_description.math_instruction.math_operation

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

        op_list = [operation]
        if operation.arch < 90:
            # Split K via Python is currently only supported for pre-SM90 kernels
            op_list.append(self.reduction_operation)

        pycutlass.compiler.add_module(op_list)

        self.operation = operation

        self.dtype_A = GemmUniversalLauncher.numpy_type(operation.A.element)
        self.dtype_B = GemmUniversalLauncher.numpy_type(operation.B.element)
        self.dtype_C = GemmUniversalLauncher.numpy_type(operation.C.element)
        self.dtype_D = GemmUniversalLauncher.numpy_type(operation.C.element)

        accumulator_size = DataTypeSize[operation.tile_description.math_instruction.element_accumulator]
        element_size = DataTypeSize[operation.A.element]

        if element_size == 1:
            self.scope_max = 1
            self.scope_min = 0
        elif element_size <= 8:
            self.scope_max = 1
            self.scope_min = -1
        elif element_size == 16:
            self.scope_max = 4
            self.scope_min = -4
        else:
            self.scope_max = 8
            self.scope_min = -8

        #: seed
        self.seed: int = seed

        #: whether the layout is interleaved
        self.interleaved = interleaved

        #: compute type
        self.compute_type = operation.epilogue_functor.element_epilogue
        self.accumulator_type = operation.tile_description.math_instruction.element_accumulator

    def print_problem_size(self, p, mode, batch_count):
        if mode == cutlass.gemm.Mode.Gemm:
            mode = "Gemm"
        elif mode == cutlass.gemm.Mode.Batched:
            mode = "GemmBatched"
        elif mode == cutlass.gemm.Mode.GemmSplitKParallel:
            mode = "GemmSplitKParallel"
        problem_size = "problem: %d, %d, %d\n batch_count: %d\n mode: %s" % (
            p.m(), p.n(), p.k(), batch_count, mode)
        print(problem_size)

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

    def uniform_init(self, size, dtype):
        if dtype in [np.float32, np.float16, bfloat16, np.float64]:
            return np.ceil(
                np.random.uniform(
                    low=self.scope_min - 0.5, high=self.scope_max - 0.5,
                    size=size).astype(dtype)
            )
        else:
            return np.random.uniform(
                low=self.scope_min - 1, high=self.scope_max + 1,
                size=size).astype(dtype)

    def reorder_tensor_B(self, tensor_B, problem_size):
        reordered_tensor_B = np.empty_like(tensor_B)
        tensor_ref_B = getTensorRef(
            tensor_B, problem_size, "b", self.operation.B.layout)
        reordered_tensor_ref_B = getTensorRef(
            reordered_tensor_B, problem_size, "b", self.operation.B.layout)
        cutlass.gemm.host.reorder_column(
            tensor_ref_B, reordered_tensor_ref_B, problem_size)
        return reordered_tensor_B

    def host_reference(self, problem_size, batch_count, tensor_A, tensor_B, tensor_C, alpha, beta):
        tensor_D_ref = np.ones_like(tensor_C)
        alpha = self.numpy_type(self.compute_type)(alpha)
        beta = self.numpy_type(self.compute_type)(beta)
        init_acc = 0

        alpha = self.compute_type(alpha).value()
        beta = self.compute_type(beta).value()
        init_acc = self.accumulator_type(init_acc).value()

        for i in range(batch_count):
            if self.operation.switched:
                tensor_ref_A = getTensorRef(
                    tensor_A, problem_size, "a", transpose(self.operation.B.layout), batch_offset=i)
                tensor_ref_B = getTensorRef(
                    tensor_B, problem_size, "b", transpose(self.operation.A.layout), batch_offset=i)
                tensor_ref_C = getTensorRef(
                    tensor_C, problem_size, "c", transpose(self.operation.C.layout), batch_offset=i)
                tensor_ref_D_ref = getTensorRef(
                    tensor_D_ref, problem_size, "d", transpose(self.operation.C.layout), batch_offset=i)
            else:
                tensor_ref_A = getTensorRef(
                    tensor_A, problem_size, "a", self.operation.A.layout, batch_offset=i)
                tensor_ref_B = getTensorRef(
                    tensor_B, problem_size, "b", self.operation.B.layout, batch_offset=i)
                tensor_ref_C = getTensorRef(
                    tensor_C, problem_size, "c", self.operation.C.layout, batch_offset=i)
                tensor_ref_D_ref = getTensorRef(
                    tensor_D_ref, problem_size, "d", self.operation.C.layout, batch_offset=i)

            if self.math_operation in [MathOperation.multiply_add_saturate]:
                cutlass.test.gemm.host.gemm_saturate(
                    problem_size, alpha, tensor_ref_A, tensor_ref_B, beta, tensor_ref_C, tensor_ref_D_ref, init_acc)
            else:
                cutlass.test.gemm.host.gemm(problem_size, alpha, tensor_ref_A,
                                            tensor_ref_B, beta, tensor_ref_C, tensor_ref_D_ref, init_acc)

        return tensor_D_ref

    def equal(self, tensor_D, tensor_D_ref, problem_size, batch_count):
        for i in range(batch_count):
            tensor_view_D = getTensorView(
                tensor_D, problem_size, "d", self.operation.C.layout, batch_offset=i)
            tensor_view_D_ref = getTensorView(
                tensor_D_ref, problem_size, "d", self.operation.C.layout, batch_offset=i)

            if not cutlass.test.gemm.host.equals(tensor_view_D, tensor_view_D_ref):
                return False

        return True

    def bytes(self, problem_size, batch_count=1, alpha=1.0, beta=0.0):
        m = problem_size.m()
        n = problem_size.n()
        k = problem_size.k()

        bytes = \
            (DataTypeSize[self.operation.A.element] * m // 8) * k + \
            (DataTypeSize[self.operation.B.element] * n // 8) * k + \
            (DataTypeSize[self.operation.C.element] * m // 8) * n

        if beta != 0:
            bytes += (DataTypeSize[self.operation.C.element] * m // 8) * n

        bytes *= batch_count

        return bytes

    def flops(self, problem_size, batch_count=1):
        m = problem_size.m()
        n = problem_size.n()
        k = problem_size.k()

        flops_ = (m * n * k) * 2 * batch_count

        return flops_

    def run_cutlass_profiler(self, mode, problem_size, batch_count=1, alpha=1.0, beta=0.0):

        cutlass_path = os.getenv('CUTLASS_PATH')
        assert cutlass_path is not None, "Environment variable 'CUTLASS_PATH' is not defined."

        values = {
            "profiler_path": cutlass_path + "/build/tools/profiler/cutlass_profiler",
            "kernel_name": self.operation.procedural_name(),
            "verification_providers": "device",
            "provider": "cutlass",
            "m": str(problem_size.m()),
            "n": str(problem_size.n()),
            "k": str(problem_size.k()),
            'split_k_slices': str(batch_count),
            'alpha': str(alpha),
            'beta': str(beta),
            'warmup': str(self.warmup_iterations),
            'profile': str(self.iterations)
        }

        cmd_template = \
            "${profiler_path} --kernels=${kernel_name} --verification-providers=${verification_providers}" \
            " --providers=${provider} --m=${m} --n=${n} --k=${k}"

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

    def run(self, mode, problem_size, batch_count=1, split_k_slices=1, alpha=1.0, beta=0.0):
        assert get_allocated_size(
        ) == 0, "%d byte of pool memory is not released in previous run" % get_allocated_size()

        np.random.seed(self.seed)

        # Assign an actual batch count in cases where we are not running in batched mode.
        # This is to differentiate between the number of split K slices and the batch count,
        # which are overloaded within the single `batch_count` variable.
        true_batch_count = batch_count if mode == cutlass.gemm.Mode.Batched else 1

        tensor_A = self.uniform_init(
            size=(problem_size.m() * problem_size.k() * true_batch_count,), dtype=self.dtype_A)
        tensor_B = self.uniform_init(
            size=(problem_size.n() * problem_size.k() * true_batch_count,), dtype=self.dtype_B)
        tensor_C = self.uniform_init(
            size=(problem_size.m() * problem_size.n() * true_batch_count,), dtype=self.dtype_C)
        tensor_D = np.zeros(
            shape=(problem_size.m() * problem_size.n() * true_batch_count,), dtype=self.dtype_D)

        #
        # Launch kernel
        #

        arguments = GemmArguments(
            operation=self.operation, problem_size=problem_size,
            A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_D,
            output_op=self.operation.epilogue_type(alpha, beta),
            gemm_mode=mode, split_k_slices=split_k_slices, batch=batch_count
        )

        if mode == cutlass.gemm.Mode.GemmSplitKParallel:
            reduction_arguments = ReductionArguments(
                self.reduction_operation, problem_size=[
                    problem_size.m(), problem_size.n()],
                partitions=split_k_slices,
                workspace=arguments.ptr_D,
                destination=tensor_D,
                source=tensor_C,
                output_op=self.reduction_operation.epilogue_type(alpha, beta)
            )

        self.operation.run(arguments)

        if mode == cutlass.gemm.Mode.GemmSplitKParallel:
            self.reduction_operation.run(reduction_arguments)

        passed = True

        if self.verification:
            if mode == cutlass.gemm.Mode.GemmSplitKParallel:
                reduction_arguments.sync()
            else:
                arguments.sync()
            tensor_D_ref = self.host_reference(
                problem_size, true_batch_count, tensor_A, tensor_B, tensor_C, alpha, beta)
            passed = self.equal(tensor_D, tensor_D_ref, problem_size, true_batch_count)

            try:
                assert passed
            except AssertionError:
                self.print_problem_size(problem_size, mode, batch_count)

        if self.profiling:
            sleep(self.sleep_time)
            for _ in range(self.warmup_iterations):
                self.operation.run(arguments)
                if mode == cutlass.gemm.Mode.GemmSplitKParallel:
                    self.reduction_operation.run(reduction_arguments)

            self.timer.start()
            for _ in range(self.iterations):
                self.operation.run(arguments)
                if mode == cutlass.gemm.Mode.GemmSplitKParallel:
                    self.reduction_operation.run(reduction_arguments)
            self.timer.stop_and_wait()

            runtime = self.timer.duration(self.iterations)

        # free memory and clear buffers
        del arguments
        if mode == cutlass.gemm.Mode.GemmSplitKParallel:
            del reduction_arguments

        assert get_allocated_size(
        ) == 0, "%d byte of pool memory is not released after current run" % get_allocated_size()

        if self.profiling:
            return runtime
        return passed


def test_all_gemm(operation: 'GemmOperationUniversal', testcase="universal"):

    passed = True

    minimum_operand_element_size = min(
        DataTypeSize[operation.A.element], DataTypeSize[operation.B.element])
    opcode_class = operation.tile_description.math_instruction.opcode_class

    if opcode_class == cutlass.OpClass.Simt:
        alignment = 1
    else:
        alignment = 128 // minimum_operand_element_size

    # int8_t gemm alignment constrainst
    if opcode_class == cutlass.OpClass.Simt and operation.A.element == cutlass.int8 and operation.A.layout == cutlass.ColumnMajor:
        alignment_m = 4
    else:
        alignment_m = alignment

    if opcode_class == cutlass.OpClass.Simt and operation.B.element == cutlass.int8 and operation.A.layout == cutlass.RowMajor:
        alignment_n = 4
    else:
        alignment_n = alignment

    if opcode_class == cutlass.OpClass.Simt and operation.A.element == cutlass.int8 \
            and operation.B.element == cutlass.int8 \
            and (operation.A.layout == cutlass.RowMajor or operation.B.layout == cutlass.ColumnMajor):

        alignment_k = 4
    else:
        alignment_k = alignment

    threadblock_k = operation.tile_description.threadblock_shape[2]

    if testcase == "interleaved":
        if operation.A.layout in [cutlass.ColumnMajorInterleaved32, cutlass.RowMajorInterleaved32]:
            interleavedk = 32
        else:
            raise ValueError("Unknown layout")

    if testcase == "interleaved":
        modes = [cutlass.gemm.Mode.Gemm, ]
        problem_size_m = [interleavedk, 512+interleavedk]
        problem_size_n = [interleavedk, 512+interleavedk]
        problem_size_k = [interleavedk, threadblock_k *
                          operation.tile_description.stages + interleavedk]
        problem_alpha = [1.0]
        problem_beta = [0.0]
        batch_counts = [1, ]
    elif testcase == "multistage":
        modes = [cutlass.gemm.Mode.Gemm, ]
        problem_size_m = [16, 528]
        problem_size_n = [16, 528]
        problem_size_k = [threadblock_k, threadblock_k * operation.tile_description.stages +
                          operation.tile_description.math_instruction.instruction_shape[2]]
        problem_alpha = [1.0]
        problem_beta = [0.0]
        batch_counts = [1, ]
    else:  # universal
        modes = [cutlass.gemm.Mode.Gemm]
        batch_counts = [1, 2, 3, 5, 7]
        if operation.arch < 90:
            # Split K kernels via Python are currently only supported pre-SM90
            modes.append(cutlass.gemm.Mode.GemmSplitKParallel)

        problem_size_m = [alignment_m, 512 - 3 * alignment_m]
        problem_size_n = [alignment_n, 512 - 2 * alignment_n]
        if operation.tile_description.stages is None:
            stages_for_k_calc = 7
        else:
            stages_for_k_calc = operation.tile_description.stages
        problem_size_k = [
            alignment_k,
            threadblock_k * stages_for_k_calc - alignment_k,
            threadblock_k * stages_for_k_calc * 3 - alignment_k]
        problem_alpha = [1.0]
        problem_beta = [2.0]

    testbed = GemmUniversalLauncher(
        operation, interleaved=(testcase == "interleaved"))

    for mode in modes:
        for m in problem_size_m:
            for n in problem_size_n:
                for k in problem_size_k:
                    for batch_count in batch_counts:
                        for alpha in problem_alpha:
                            for beta in problem_beta:
                                # skip very small K problems
                                if testcase == "universal":
                                    if (k // batch_count < 2 * threadblock_k):
                                        continue

                                problem_size = cutlass.gemm.GemmCoord(m, n, k)

                                if operation.arch < 90:
                                    split_k_slices = batch_count
                                else:
                                    split_k_slices = 1

                                overridden_mode = mode
                                if mode == cutlass.gemm.Mode.Gemm and batch_count > 1:
                                    overridden_mode = cutlass.gemm.Mode.Batched

                                passed = testbed.run(
                                    overridden_mode, problem_size, batch_count, split_k_slices, alpha, beta)

                                err, = cudart.cudaDeviceSynchronize()
                                if err != cuda.CUresult.CUDA_SUCCESS:
                                    raise RuntimeError(
                                        "CUDA Error %s" % str(err))

                                if not passed:
                                    return False

    return passed
