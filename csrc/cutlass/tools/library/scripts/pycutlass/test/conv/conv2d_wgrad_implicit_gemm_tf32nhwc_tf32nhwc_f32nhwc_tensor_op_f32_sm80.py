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

# test/unit/conv/device/conv2d_wgrad_implicit_gemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_sm80.cu
import pycutlass
from pycutlass import *
from pycutlass.test import *
from pycutlass.utils.device import device_cc
import unittest


@unittest.skipIf(device_cc() < 80, "Device compute capability is insufficient for SM80 tests.")
class Conv2dWgradImplicitGemmTF32nhwcTF32nhwcTF32nhwcTensorOpF32SM80(unittest.TestCase):
    def test_SM80_Device_Conv2d_Wgrad_Optimized_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 8],
            element_a=cutlass.float32, element_b=cutlass.float32,
            element_accumulator=cutlass.float32, opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        A = TensorDescription(
            element=math_inst.element_a, 
            layout=cutlass.TensorNHWC,
            alignment=4)
        B = TensorDescription(
            element=math_inst.element_b, 
            layout=cutlass.TensorNHWC, 
            alignment=4)
        C = TensorDescription(
            element=cutlass.float32,
            layout=cutlass.TensorNHWC, 
            alignment=8)

        tile_description = TileDescription(
            threadblock_shape=[128, 128, 16], stages=3, 
            warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        epilogue_functor = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, cutlass.float32)

        operation = Conv2dOperation(
            conv_kind=cutlass.conv.Operator.wgrad, iterator_algorithm=cutlass.conv.IteratorAlgorithm.optimized,
            arch=80, tile_description=tile_description, A=A, B=B, C=C, 
            stride_support=StrideSupport.Strided,
            epilogue_functor=epilogue_functor,
            swizzling_functor=cutlass.IdentitySwizzle1
        )
        
        self.assertTrue(test_all_conv2d(operation))
    
    def test_SM80_Device_Conv2d_Wgrad_Optimized_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_align1(self):
        math_inst = MathInstruction(
            instruction_shape=[16, 8, 8],
            element_a=cutlass.float32, element_b=cutlass.float32,
            element_accumulator=cutlass.float32, opcode_class=cutlass.OpClass.TensorOp,
            math_operation=MathOperation.multiply_add
        )

        A = TensorDescription(
            element=math_inst.element_a, 
            layout=cutlass.TensorNHWC,
            alignment=1)
        B = TensorDescription(
            element=math_inst.element_b, 
            layout=cutlass.TensorNHWC, 
            alignment=1)
        C = TensorDescription(
            element=cutlass.float32,
            layout=cutlass.TensorNHWC, 
            alignment=4)

        tile_description = TileDescription(
            threadblock_shape=[128, 128, 32], stages=3, 
            warp_count=[2, 2, 1],
            math_instruction=math_inst
        )

        epilogue_functor = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, cutlass.float32)

        operation = Conv2dOperation(
            conv_kind=cutlass.conv.Operator.wgrad, iterator_algorithm=cutlass.conv.IteratorAlgorithm.optimized,
            arch=80, tile_description=tile_description, A=A, B=B, C=C, 
            stride_support=StrideSupport.Strided,
            epilogue_functor=epilogue_functor,
            swizzling_functor=cutlass.IdentitySwizzle1
        )
        
        problem_sizes = [
            cutlass.conv.Conv2dProblemSize(
                cutlass.Tensor4DCoord(1, 8, 8, 1),
                cutlass.Tensor4DCoord(1, 3, 3, 1),
                cutlass.Tensor4DCoord(1, 1, 1, 1),
                cutlass.MatrixCoord(1, 1),
                cutlass.MatrixCoord(1, 1),
                cutlass.conv.Mode.cross_correlation,
                1, 1
            ),
        ]
        
        self.assertTrue(test_all_conv2d(operation, problem_sizes))
    
if __name__ == '__main__':
    pycutlass.get_memory_pool(2**26, 2**26)
    unittest.main()
