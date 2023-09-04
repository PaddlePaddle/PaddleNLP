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

"""
Test cases for frontends
"""

import pycutlass
import unittest
from pycutlass import *
from pycutlass.utils.device import device_cc


class Test_Frontend(unittest.TestCase):
    def setUp(self) -> None:
        #
        # define the cutlass operator
        #
        cc = device_cc()
        math_inst = MathInstruction(
            [1, 1, 1], cutlass.float32, cutlass.float32, cutlass.float32,
            cutlass.OpClass.Simt, MathOperation.multiply_add
        )

        stages = 2
        tile_description = TileDescription(
            [128, 128, 8], stages, [2, 4, 1],
            math_inst
        )

        A = TensorDescription(
            cutlass.float32, cutlass.RowMajor, 1
        )

        B = TensorDescription(
            cutlass.float32, cutlass.RowMajor, 1
        )

        C = TensorDescription(
            cutlass.float32, cutlass.RowMajor, 1
        )

        epilogue_functor = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, cutlass.float32)

        self.operation = GemmOperationUniversal(
            arch=cc, tile_description=tile_description,
            A=A, B=B, C=C, 
            epilogue_functor=epilogue_functor, 
            swizzling_functor=cutlass.IdentitySwizzle1
        )

        pycutlass.compiler.add_module([self.operation,])


    def test_torch_frontend(self):
        try:
            import torch
        except:
            self.assertTrue(False, "Unable to import torch")

        problem_size = cutlass.gemm.GemmCoord(512, 256, 128)

        tensor_A = torch.ceil(torch.empty(size=(problem_size.m(), problem_size.k()), dtype=torch.float32, device="cuda").uniform_(-8.5, 7.5))
        tensor_B = torch.ceil(torch.empty(size=(problem_size.k(), problem_size.n()), dtype=torch.float32, device="cuda").uniform_(-8.5, 7.5))
        tensor_C = torch.ceil(torch.empty(size=(problem_size.m(), problem_size.n()), dtype=torch.float32, device="cuda").uniform_(-8.5, 7.5))
        tensor_D = torch.empty_like(tensor_C)
        

        alpha = 1.0
        beta = 0.0

        arguments = GemmArguments(
            operation=self.operation, problem_size=problem_size,
            A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_D,
            output_op=self.operation.epilogue_type(alpha, beta),
            gemm_mode=cutlass.gemm.Mode.Gemm, split_k_splices=1
        )

        self.operation.run(arguments)

        arguments.sync()

        tensor_D_ref = alpha * tensor_A @ tensor_B + beta * tensor_C

        self.assertTrue(torch.equal(tensor_D, tensor_D_ref))
    
    def test_cupy_frontend(self):
        try:
            import cupy as cp
        except:
            self.assertTrue(False, "Unable to import cupy")

        cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

        problem_size = cutlass.gemm.GemmCoord(512, 256, 128)

        tensor_A = cp.ceil(cp.random.uniform(low=-8.5, high=7.5, size=(problem_size.m(), problem_size.k()), dtype=cp.float32))
        tensor_B = cp.ceil(cp.random.uniform(low=-8.5, high=7.5, size=(problem_size.k(), problem_size.n()), dtype=cp.float32))
        tensor_C = cp.ceil(cp.random.uniform(low=-8.5, high=7.5, size=(problem_size.m(), problem_size.n()), dtype=cp.float32))
        tensor_D = cp.ones_like(tensor_C)

        alpha = 1.0
        beta = 1.0

        tensor_D_ref = alpha * tensor_A @ tensor_B + beta * tensor_C

        arguments = GemmArguments(
            operation=self.operation, problem_size=problem_size,
            A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_D,
            output_op=self.operation.epilogue_type(alpha, beta),
            gemm_mode=cutlass.gemm.Mode.Gemm, split_k_splices=1
        )

        self.operation.run(arguments)

        arguments.sync()

        self.assertTrue(cp.array_equal(tensor_D, tensor_D_ref))


if __name__ == '__main__':
    pycutlass.get_memory_pool(2**32, 2**32)
    unittest.main()
