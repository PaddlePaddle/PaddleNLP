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

from functools import partial
import pycutlass
from pycutlass import *
from pycutlass import library
from pycutlass.test import *
import unittest

from pycutlass.test.utils import LayoutCombination, get_name
from pycutlass.test.gemm_testbed import test_all_gemm
from pycutlass.utils.device import device_cc


name_fn = partial(get_name, element_a=cutlass.float16, element_b=cutlass.float16, arch=90)

def add_test(cls, layouts, alignments, element_output, element_accumulator, element_epilogue,
             cluster_shape, threadblock_shape, stages, opclass, persistent=False):
    """
    Create a test-running function with the given specification and set it as a method of `cls`.

    :param cls: class to which the generated method will be added
    :type cls: type
    :param layouts: indexable container of layouts of A, B, and C operands
    :param alignments: indexable container of alingments of A, B, and C operands
    :param element_output: data type of the output element
    :param element_accumulator: data type used in accumulation
    :param element_epilogue: data type used in computing the epilogue
    :param cluster_shape: indexable container of dimensions of threadblock cluster to be launched
    :param threadblock_shape: indexable container of dimensions of threadblock tiles
    :param stages: number of pipeline stages to use in the kernel
    :type stages: int
    :param opclass: class of operation being performed (e.g., SIMT, Tensor Core)
    :type opclass: cutlass.OpClass
    :param persistent: whether this is a persistent warp-specialized kernel
    :type persistent: bool
    """

    def run(self):
        """
        Dynamically-generated function that constructs a GEMM operation and verifies it against
        multiple test cases.
        """
        element_A = cutlass.int8
        element_B = cutlass.int8
        inst_shape = [1, 1, 1] if opclass == cutlass.OpClass.Simt else None
        warp_count = [2, 2, 1] if opclass == cutlass.OpClass.Simt else None
        math_inst = MathInstruction(
            instruction_shape=inst_shape,
            element_a=element_A, element_b=element_B, element_accumulator=element_accumulator,
            opcode_class=opclass, math_operation=MathOperation.multiply_add
        )

        tile_description = TileDescription(
            threadblock_shape=threadblock_shape,
            cluster_shape=cluster_shape,
            stages=stages, warp_count=warp_count,
            math_instruction=math_inst,
            persistent=persistent
        )

        A = TensorDescription(element=element_A, layout=layouts[0], alignment=alignments[0])
        B = TensorDescription(element=element_B, layout=layouts[1], alignment=alignments[1])
        C = TensorDescription(element=element_output, layout=layouts[2], alignment=alignments[2])

        if opclass == cutlass.OpClass.Simt:
            epilogue_functor_cls = LinearCombinationClamp
        else:
            epilogue_functor_cls = LinearCombination
        epilogue_functor = epilogue_functor_cls(C.element, C.alignment, math_inst.element_accumulator, element_epilogue)

        swizzling_functor = cutlass.IdentitySwizzle1

        operation = GemmOperationUniversal(
            arch=90, tile_description=tile_description, A=A, B=B, C=C,
            epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor)

        self.assertTrue(test_all_gemm(operation, "universal"))

    if persistent:
        suffix = "_persistent"
    else:
        suffix = ""

    name = name_fn(layouts, alignments, element_output, element_accumulator,
                  element_epilogue, cluster_shape, threadblock_shape, stages, opclass=opclass, suffix=suffix)
    setattr(cls, name, run)

    return run


@unittest.skipIf(device_cc() < 90, "Device compute capability is insufficient for SM90 tests.")
class GemmS8Sm90(unittest.TestCase):
    """
    Wrapper class to which tests will be added dynamically in __main__
    """
    pass


add_test_tensorop = partial(add_test, opclass=cutlass.OpClass.TensorOp)
add_test_simt = partial(add_test, opclass=cutlass.OpClass.Simt)

# Tests with 1x1x1 clusters
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNN, [16, 16, 16], cutlass.int8, cutlass.int32, cutlass.int32, [1, 1, 1], [128, 128, 128], 3)
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [16, 16, 16], cutlass.int8, cutlass.int32, cutlass.int32, [1, 1, 1], [128, 128, 128], None)
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [16, 16, 8],  cutlass.int8, cutlass.int32, cutlass.int32, [1, 1, 1], [128, 128, 128], None)
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [16, 16, 16], cutlass.int8, cutlass.int32, cutlass.int32, [1, 1, 1], [64, 128, 128], None)
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [16, 16, 16], cutlass.int8, cutlass.int32, cutlass.int32, [1, 1, 1], [128, 64, 32], None)
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [4, 4, 16], cutlass.int8, cutlass.int32, cutlass.int32, [1, 1, 1], [128, 128, 128], None)

# Tests with different cluster shapes
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [16, 16, 16], cutlass.int8, cutlass.int32, cutlass.int32, [2, 2, 1], [128, 128, 128], None)
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [16, 16, 16], cutlass.int8, cutlass.int32, cutlass.int32, [1, 4, 1], [128, 128, 128], None)
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [16, 16, 16], cutlass.int8, cutlass.int32, cutlass.int32, [4, 4, 1], [128, 128, 128], None)

# Tests with persistent warp-specialized threadblocks
add_test_tensorop(GemmS8Sm90, LayoutCombination.TNT, [16, 16, 16], cutlass.int8, cutlass.int32, cutlass.int32, [2, 1, 1], [128, 128, 128], None, persistent=True)

# Tests for SIMT
add_test_simt(GemmS8Sm90, LayoutCombination.TNN, [1, 1, 1], cutlass.int8, cutlass.int32, cutlass.int32, [1, 1, 1], [64, 32, 8], 2)

if __name__ == '__main__':
    pycutlass.get_memory_pool(2**30, 2**30)
    unittest.main()
