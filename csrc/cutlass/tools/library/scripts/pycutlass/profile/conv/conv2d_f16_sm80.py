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

from pycutlass import *
import pycutlass
from pycutlass.epilogue import LinearCombination
from pycutlass.test.conv2d_testbed import Conv2dLauncher


if __name__ == "__main__":
    pycutlass.get_memory_pool(2**33, 2**33)
    pycutlass.compiler.nvcc()

    math_inst = MathInstruction(
        instruction_shape=[16, 8, 16],
        element_a=cutlass.float16, element_b=cutlass.float16,
        element_accumulator=cutlass.float32, opcode_class=cutlass.OpClass.TensorOp,
        math_operation=MathOperation.multiply_add
    )

    A = TensorDescription(
        element=math_inst.element_a, 
        layout=cutlass.TensorNHWC,
        alignment=8)
    B = TensorDescription(
        element=math_inst.element_b, 
        layout=cutlass.TensorNHWC, 
        alignment=8)
    C = TensorDescription(
        element=cutlass.float32,
        layout=cutlass.TensorNHWC, 
        alignment=8)

    tile_description = TileDescription(
        threadblock_shape=[128, 128, 64], stages=4, 
        warp_count=[2, 2, 1],
        math_instruction=math_inst
    )

    epilogue_functor = LinearCombination(cutlass.float32, 4, cutlass.float32, cutlass.float32)

    operation = Conv2dOperation(
        conv_kind=cutlass.conv.Operator.fprop, iterator_algorithm=cutlass.conv.IteratorAlgorithm.optimized,
        arch=80, tile_description=tile_description, A=A, B=B, C=C, 
        element_epilogue=cutlass.float32, stride_support=StrideSupport.Strided,
        epilogue_functor=epilogue_functor,
        swizzling_functor=cutlass.IdentitySwizzle1
    )

    profiler = Conv2dLauncher(operation, verification=False, profiling=True)

    python_runtime = profiler.run(
        problem_size = cutlass.conv.Conv2dProblemSize(
            cutlass.Tensor4DCoord(32, 224, 224, 128),
            cutlass.Tensor4DCoord(128, 3, 3, 128),
            cutlass.Tensor4DCoord(1, 1, 1, 1),
            cutlass.MatrixCoord(1, 1),
            cutlass.MatrixCoord(1, 1),
            cutlass.conv.Mode.cross_correlation,
            1, 1
        ), split_k_mode=cutlass.conv.SplitKMode.Serial
    )


    cpp_runtime = profiler.run_cutlass_profiler(
        problem_size = cutlass.conv.Conv2dProblemSize(
            cutlass.Tensor4DCoord(32, 224, 224, 128),
            cutlass.Tensor4DCoord(128, 3, 3, 128),
            cutlass.Tensor4DCoord(1, 1, 1, 1),
            cutlass.MatrixCoord(1, 1),
            cutlass.MatrixCoord(1, 1),
            cutlass.conv.Mode.cross_correlation,
            1, 1
        ), split_k_mode=cutlass.conv.SplitKMode.Serial
    )

    print(cpp_runtime / python_runtime)
