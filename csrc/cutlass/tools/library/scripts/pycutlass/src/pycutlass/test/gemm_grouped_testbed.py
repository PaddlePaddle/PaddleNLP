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
from pycutlass.test.gemm_testbed import getTensorRef, getTensorView, transpose
from pycutlass import *
import numpy as np
import cutlass
from bfloat16 import bfloat16


class TestbedGrouped:
    def __init__(self, operation: GemmOperationGrouped, seed: int = 2080) -> None:

        pycutlass.compiler.add_module([operation])

        self.seed = seed

        self.operation = operation

        element_size = DataTypeSize[operation.A.element]

        self.dtype_A = self.numpy_type(operation.A.element)
        self.dtype_B = self.numpy_type(operation.B.element)
        self.dtype_C = self.numpy_type(operation.C.element)
        self.dtype_D = self.numpy_type(operation.C.element)

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

        #: compute type
        self.compute_type = operation.epilogue_functor.element_epilogue

        self.accumulator_type = operation.tile_description.math_instruction.element_accumulator

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

    def print_problem_size(self, p):
        problem_size = "problem: %d, %d, %d\n" % (p.m(), p.n(), p.k())
        print(problem_size)

    def run(self, problem_count: int, alpha: float = 1.0, beta: float = 0.0) -> bool:

        assert get_allocated_size(
        ) == 0, "%d byte of pool memory is not released in previous run" % get_allocated_size()

        # initialize
        np.random.seed(self.seed)

        # generate the problem sizes
        problem_sizes = []
        tensor_As = []
        tensor_Bs = []
        tensor_Cs = []
        tensor_Ds = []
        tensor_D_refs = []

        for i in range(problem_count):
            if self.dtype_A == np.int8:
                if i == 0:
                    problem_size = cutlass.gemm.GemmCoord(48, 16, 32)
                else:
                    problem_size = cutlass.gemm.GemmCoord(
                        16 * np.random.randint(0, 64) + 48,
                        16 * np.random.randint(0, 64) + 48,
                        16 * np.random.randint(0, 64) + 48
                    )
            else:
                if i == 0:
                    problem_size = cutlass.gemm.GemmCoord(48, 16, 8)
                else:
                    problem_size = cutlass.gemm.GemmCoord(
                        8 * np.random.randint(0, 64) + 24,
                        8 * np.random.randint(0, 64) + 24,
                        8 * np.random.randint(0, 64) + 24
                    )

            tensor_As.append(
                self.uniform_init(
                    size=(problem_size.m() * problem_size.k(),),
                    dtype=self.dtype_A)
            )
            tensor_Bs.append(
                self.uniform_init(
                    size=(problem_size.n() * problem_size.k(),),
                    dtype=self.dtype_B)
            )
            tensor_Cs.append(
                self.uniform_init(
                    size=(problem_size.m() * problem_size.n(),),
                    dtype=self.dtype_C)
            )

            tensor_Ds.append(
                np.zeros(
                    shape=(problem_size.m() * problem_size.n(),),
                    dtype=self.dtype_D
                )
            )

            tensor_D_refs.append(
                np.ones(
                    shape=(problem_size.m() * problem_size.n(),),
                    dtype=self.dtype_D
                )
            )

            problem_sizes.append(problem_size)

        arguments = GemmGroupedArguments(
            operation=self.operation, problem_sizes=problem_sizes,
            A=tensor_As, B=tensor_Bs, C=tensor_Cs, D=tensor_Ds,
            output_op=self.operation.epilogue_type(alpha, beta)
        )

        self.operation.run(arguments)

        arguments.sync()

        #
        # Reference check
        #
        alpha = self.compute_type(alpha).value()
        beta = self.compute_type(beta).value()
        init_acc = self.accumulator_type(0).value()

        for idx, problem_size in enumerate(problem_sizes):
            if self.operation.switched:
                tensor_ref_A = getTensorRef(
                    tensor_As[idx], problem_size, "a", transpose(self.operation.B.layout))
                tensor_ref_B = getTensorRef(
                    tensor_Bs[idx], problem_size, "b", transpose(self.operation.A.layout))
                tensor_ref_C = getTensorRef(
                    tensor_Cs[idx], problem_size, "c", transpose(self.operation.C.layout))
                tensor_ref_D_ref = getTensorRef(
                    tensor_D_refs[idx], problem_size, "d", transpose(self.operation.C.layout))
            else:
                tensor_ref_A = getTensorRef(
                    tensor_As[idx], problem_size, "a", self.operation.A.layout)
                tensor_ref_B = getTensorRef(
                    tensor_Bs[idx], problem_size, "b", self.operation.B.layout)
                tensor_ref_C = getTensorRef(
                    tensor_Cs[idx], problem_size, "c", self.operation.C.layout)
                tensor_ref_D_ref = getTensorRef(
                    tensor_D_refs[idx], problem_size, "d", self.operation.C.layout)

            tensor_view_D_ref = getTensorView(
                tensor_D_refs[idx], problem_size, "d", self.operation.C.layout)

            cutlass.test.gemm.host.gemm(problem_size, alpha, tensor_ref_A,
                                        tensor_ref_B, beta, tensor_ref_C, tensor_ref_D_ref, init_acc)

            tensor_view_D = getTensorView(
                tensor_Ds[idx], problem_size, "d", self.operation.C.layout)

            passed = cutlass.test.gemm.host.equals(
                tensor_view_D, tensor_view_D_ref)

            try:
                assert passed
            except AssertionError:
                self.print_problem_size(problem_size)

        del arguments

        assert get_allocated_size(
        ) == 0, "%d byte of pool memory is not released after current run" % get_allocated_size()

        return passed
