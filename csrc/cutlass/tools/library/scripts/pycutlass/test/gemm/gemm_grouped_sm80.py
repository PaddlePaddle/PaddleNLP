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
import unittest

from pycutlass.test.gemm_grouped_testbed import TestbedGrouped
from pycutlass.utils.device import device_cc


@unittest.skipIf(device_cc() < 80, "Device compute capability is insufficient for SM80 tests.")
class GemmGroupedSm80(unittest.TestCase):
    def test_SM80_Device_GemmGrouped_f16n_f16t_f32n_tensor_op_f32_128x128x32_64x64x32(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 16], element_a=cutlass.float16,
            element_b=cutlass.float16, element_accumulator=cutlass.float32,
            opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=[128, 128, 32],
            stages=3, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.float16, layout=cutlass.ColumnMajor,
            alignment=8
        )

        B = TensorDescription(
            element=cutlass.float16, layout=cutlass.ColumnMajor,
            alignment=8
        )

        C = TensorDescription(
            element=cutlass.float32, layout=cutlass.ColumnMajor,
            alignment=4
        )

        element_epilogue = cutlass.float32
        epilogue_functor = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, element_epilogue)
        swizzling_functor = cutlass.BatchedIdentitySwizzle

        for precompute_mode in [SchedulerMode.Device, SchedulerMode.Host]:
            operation = GemmOperationGrouped(
                80,
                tile_description, A, B, C,
                epilogue_functor, swizzling_functor,
                precompute_mode=precompute_mode
            )

            testbed = TestbedGrouped(operation=operation)

            self.assertTrue(testbed.run(24))
    
    def test_SM80_Device_GemmGrouped_f64t_f64t_f64n_tensor_op_f64_64x64x16_32x32x16(self):
        math_inst = MathInstruction(
            instruction_shape=[8, 8, 4], element_a=cutlass.float64,
            element_b=cutlass.float64, element_accumulator=cutlass.float64,
            opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=[64, 64, 16],
            stages=4, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.float64, layout=cutlass.RowMajor,
            alignment=1
        )

        B = TensorDescription(
            element=cutlass.float64, layout=cutlass.RowMajor,
            alignment=1
        )

        C = TensorDescription(
            element=cutlass.float64, layout=cutlass.ColumnMajor,
            alignment=1
        )

        element_epilogue = cutlass.float64
        epilogue_functor = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, element_epilogue)
        swizzling_functor = cutlass.BatchedIdentitySwizzle

        for precompute_mode in [SchedulerMode.Device, SchedulerMode.Host]:
            operation = GemmOperationGrouped(
                80,
                tile_description, A, B, C,
                epilogue_functor, swizzling_functor,
                precompute_mode=precompute_mode
            )

            testbed = TestbedGrouped(operation=operation)

            self.assertTrue(testbed.run(24))
    
    def test_SM80_Device_GemmGrouped_f32t_f32t_f32t_simt_f32_128x64x8_64x32x1(self):
        math_inst = MathInstruction(
            instruction_shape=[1, 1, 1], element_a=cutlass.float32,
            element_b=cutlass.float32, element_accumulator=cutlass.float32,
            opcode_class=cutlass.OpClass.Simt,
            math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=[128, 64, 8],
            stages=4, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.float32, layout=cutlass.RowMajor,
            alignment=1
        )

        B = TensorDescription(
            element=cutlass.float32, layout=cutlass.RowMajor,
            alignment=1
        )

        C = TensorDescription(
            element=cutlass.float32, layout=cutlass.RowMajor,
            alignment=1
        )

        element_epilogue = cutlass.float32
        epilogue_functor = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, element_epilogue)
        swizzling_functor = cutlass.BatchedIdentitySwizzle

        for precompute_mode in [SchedulerMode.Device, SchedulerMode.Host]:
            operation = GemmOperationGrouped(
                80,
                tile_description, A, B, C,
                epilogue_functor, swizzling_functor,
                precompute_mode=precompute_mode
            )

            testbed = TestbedGrouped(operation=operation)

            self.assertTrue(testbed.run(27))
    
    def test_SM80_Device_GemmGrouped_f16n_f16t_f32n_tensor_op_f32_128x128x32_64x64x32_cache(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 16], element_a=cutlass.float16,
            element_b=cutlass.float16, element_accumulator=cutlass.float32,
            opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=[128, 128, 32],
            stages=3, warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        A = TensorDescription(
            element=cutlass.float16, layout=cutlass.ColumnMajor,
            alignment=8
        )

        B = TensorDescription(
            element=cutlass.float16, layout=cutlass.ColumnMajor,
            alignment=8
        )

        C = TensorDescription(
            element=cutlass.float32, layout=cutlass.ColumnMajor,
            alignment=4
        )

        element_epilogue = cutlass.float32
        epilogue_functor = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, element_epilogue)
        swizzling_functor = cutlass.BatchedIdentitySwizzle

        for precompute_mode in [SchedulerMode.Device, SchedulerMode.Host]:
            operation = GemmOperationGrouped(
                80,
                tile_description, A, B, C,
                epilogue_functor, swizzling_functor,
                precompute_mode=precompute_mode
            )

            testbed = TestbedGrouped(operation=operation)

            self.assertTrue(testbed.run(5))


if __name__ == '__main__':
    pycutlass.get_memory_pool(2**30, 2**30)
    unittest.main()
