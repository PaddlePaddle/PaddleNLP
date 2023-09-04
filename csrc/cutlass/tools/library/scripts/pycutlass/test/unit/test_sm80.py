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

## Test case generator for SM80

import pycutlass
from pycutlass import *
from pycutlass.test import *
from pycutlass.utils.device import device_cc
import unittest
import xmlrunner
import argparse

#
# Create GEMM operation
#
@unittest.skipIf(device_cc() < 80, "Device compute capability is insufficient for SM80 tests.")
def TestGemmOperator(gemm_kind, math_inst, layout, alignment, tiling, arch, mixed=False,
    epilogue_functor=None, swizzling_functor=cutlass.IdentitySwizzle1, **kwargs):
    """
    Test GEMM Operation based on configuration
    """

    if "data_type" in kwargs.keys():
        data_type = kwargs["data_type"]
    else:
        if mixed or math_inst.element_a == cutlass.bfloat16:
            data_type = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_accumulator,
                math_inst.element_accumulator
            ]
        else:
            data_type = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_a,
                math_inst.element_accumulator
            ]
    
    tile_description = TileDescription(
        tiling[0], tiling[1], tiling[2],
        math_inst
    )

    A = TensorDescription(
        data_type[0], layout[0], alignment[0]
    )

    B = TensorDescription(
        data_type[1], layout[1], alignment[1]
    )

    C = TensorDescription(
        data_type[2], layout[2], alignment[2]
    )

    element_epilogue = data_type[3]
    if epilogue_functor is None:
        epilogue_functor = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, element_epilogue)

    if gemm_kind == GemmKind.Universal:
        operation = GemmOperationUniversal(
            arch=arch, tile_description=tile_description,
            A=A, B=B, C=C,
            epilogue_functor=epilogue_functor, swizzling_functor=swizzling_functor
        )
        if A.layout in [cutlass.ColumnMajorInterleaved32, cutlass.RowMajorInterleaved32]:
            return test_all_gemm(operation, "interleaved")
        else:
            return test_all_gemm(operation, "universal")
        
    elif gemm_kind == GemmKind.Grouped:
        operation = GemmOperationGrouped(
            arch, tile_description, A, B, C,
            epilogue_functor, swizzling_functor,
            precompute_mode=kwargs["precompute_mode"]
        )
        testbed = TestbedGrouped(operation=operation)
        return testbed.run(24)
    else:
        raise NotImplementedError("the gemm kind is not implemented")


def TestConv2dOperator(math_inst, alignment, tiling, arch, 
    stride_supports=[StrideSupport.Strided, StrideSupport.Strided, StrideSupport.Strided],
    epilogue_functor=None, 
    swizzling_functor=cutlass.IdentitySwizzle1, interleaved=False, **kwargs):
    """
    Test Conv2d Operation based on configurations
    """

    mixeds = [False, True, False]
    conv_kinds = [cutlass.conv.Operator.fprop, cutlass.conv.Operator.dgrad, cutlass.conv.Operator.wgrad]

    results = []

    default_swizzling_functor = swizzling_functor

    if "layout" in kwargs.keys():
        layout = kwargs["layout"]
    else:
        layout = (cutlass.TensorNHWC, cutlass.TensorNHWC, cutlass.TensorNHWC)

    for mixed, conv_kind, stride_support in zip(mixeds, conv_kinds, stride_supports):

        if "data_type" in kwargs.keys():
            data_type = kwargs["data_type"]
        else:
            if mixed or math_inst.element_a == cutlass.bfloat16:
                data_type = [
                    math_inst.element_a,
                    math_inst.element_b,
                    math_inst.element_accumulator,
                    math_inst.element_accumulator
                ]
            else:
                data_type = [
                    math_inst.element_a,
                    math_inst.element_b,
                    math_inst.element_a,
                    math_inst.element_accumulator
                ]
        # skip Int8 Conv Backward
        if data_type[0] == cutlass.int8 and conv_kind in [cutlass.conv.Operator.dgrad, cutlass.conv.Operator.wgrad]:
            continue

        A = TensorDescription(
            element=data_type[0],
            layout=layout[0],
            alignment=alignment[0])
        B = TensorDescription(
            element=data_type[1],
            layout=layout[1], 
            alignment=alignment[1])
        C = TensorDescription(
            element=data_type[2],
            layout=layout[2], 
            alignment=alignment[2])
        
        tile_description = TileDescription(
            threadblock_shape=tiling[0], stages=tiling[1], 
            warp_count=tiling[2],
            math_instruction=math_inst
        )

        if conv_kind == cutlass.conv.Operator.dgrad and stride_support == StrideSupport.Strided:
            swizzling_functor = cutlass.StridedDgradIdentitySwizzle1
        else:
            swizzling_functor = default_swizzling_functor
        
        if epilogue_functor is None:
            epilogue_functor_ = LinearCombination(
            C.element, C.alignment, 
            math_inst.element_accumulator, data_type[3])

        operation = Conv2dOperation(
            conv_kind=conv_kind, iterator_algorithm=cutlass.conv.IteratorAlgorithm.optimized,
            arch=arch, tile_description=tile_description, A=A, B=B, C=C, 
            stride_support=stride_support,
            epilogue_functor=epilogue_functor_,
            swizzling_functor=swizzling_functor
        )
        
        results.append(test_all_conv2d(operation, interleaved=interleaved))
    
    return results



class Test_SM80(unittest.TestCase):
    def test_SM80_TensorOp_16816(self):
        math_instructions = [
            MathInstruction(
                [16, 8, 16], cutlass.float16, cutlass.float16, cutlass.float32,
                cutlass.OpClass.TensorOp, MathOperation.multiply_add
            ),
            MathInstruction(
                [16, 8, 16], cutlass.float16, cutlass.float16, cutlass.float16,
                cutlass.OpClass.TensorOp, MathOperation.multiply_add
            ),
            MathInstruction(
                [16, 8, 16], cutlass.bfloat16, cutlass.bfloat16, cutlass.float32,
                cutlass.OpClass.TensorOp, MathOperation.multiply_add
            )
        ]

        layouts = [
            (cutlass.RowMajor, cutlass.RowMajor, cutlass.RowMajor),
            (cutlass.ColumnMajor, cutlass.RowMajor, cutlass.RowMajor),
            (cutlass.RowMajor, cutlass.ColumnMajor, cutlass.RowMajor)
        ]

        alignments = [
            (8, 8, 8), (4, 8, 8), (8, 4, 8)
        ]

        tilings = [
            ([256, 128, 32], 3, [4, 2, 1]),
            ([64, 256, 32], 4, [1, 4, 1]),
            ([128, 64, 64], 3, [2, 2, 1])
        ]

        for math_inst, layout, alignment, tiling in zip(math_instructions, layouts, alignments, tilings):
            self.assertTrue(TestGemmOperator(GemmKind.Universal, math_inst, layout, alignment, tiling, 80, False))
            self.assertTrue(TestGemmOperator(GemmKind.Grouped, math_inst, layout, alignment, tiling, 80, True, precompute_mode=SchedulerMode.Host))
            stride_supports = [StrideSupport.Strided, StrideSupport.Strided, StrideSupport.Strided]
            results = TestConv2dOperator(math_inst, alignment, tiling, 80, stride_supports=stride_supports)
            for res in results:
                self.assertTrue(res)

    def test_SM80_TensorOp_1688(self):
        # tf32 is not supported by most of python environment. Skip the test
        self.assertTrue(True)
    
    def test_SM80_TensorOp_1688_fast_math(self):
        math_instructions = [
            MathInstruction(
                [16, 8, 8], cutlass.tfloat32, cutlass.tfloat32, cutlass.float32,
                cutlass.OpClass.TensorOp, MathOperation.multiply_add
            ),
            MathInstruction(
                [16, 8, 8], cutlass.float16, cutlass.float16, cutlass.float32,
                cutlass.OpClass.TensorOp, MathOperation.multiply_add_fast_f16
            ),
            MathInstruction(
                [16, 8, 8], cutlass.bfloat16, cutlass.bfloat16, cutlass.float32,
                cutlass.OpClass.TensorOp, MathOperation.multiply_add_fast_bf16
            ),
            MathInstruction(
                [16, 8, 8], cutlass.float32, cutlass.float32, cutlass.float32,
                cutlass.OpClass.TensorOp, MathOperation.multiply_add_fast_f32
            )
        ]

        layouts = [
            (cutlass.RowMajor, cutlass.RowMajor, cutlass.ColumnMajor),
            (cutlass.RowMajor, cutlass.ColumnMajor, cutlass.ColumnMajor),
            (cutlass.ColumnMajor, cutlass.RowMajor, cutlass.ColumnMajor),
            (cutlass.ColumnMajor, cutlass.ColumnMajor, cutlass.RowMajor)
        ]
        alignments = [
            (4, 4, 4), (4, 2, 4), (2, 4, 4), (2, 2, 4)
        ]
        tilings = [
            ([128, 256, 16], 3, [4, 2, 1]),
            ([64, 256, 16], 4, [1, 4, 1]),
            ([128, 64, 32], 3, [2, 2, 1]),
            ([256, 64, 32], 3, [4, 2, 1])
        ]
        data_type = [
            cutlass.float32, cutlass.float32, cutlass.float32, cutlass.float32
        ]
        for math_inst, layout, alignment, tiling in zip(math_instructions, layouts, alignments, tilings):
            self.assertTrue(
                TestGemmOperator(
                    GemmKind.Universal, math_inst, layout, 
                    alignment, tiling, 80, False, data_type=data_type))
            self.assertTrue(
                TestGemmOperator(
                    GemmKind.Grouped, math_inst, layout, alignment, tiling, 80, 
                    True, precompute_mode=SchedulerMode.Device, data_type=data_type))
            stride_supports = [StrideSupport.Unity, StrideSupport.Strided, StrideSupport.Unity]
            results = TestConv2dOperator(math_inst, alignment, tiling, 80, stride_supports=stride_supports, data_type=data_type)
            for res in results:
                self.assertTrue(res)

    def test_SM80_TensorOp_884(self):
        math_inst = MathInstruction(
            [8, 8, 4], cutlass.float64, cutlass.float64, cutlass.float64,
            cutlass.OpClass.TensorOp, MathOperation.multiply_add
        )
        layout = (cutlass.ColumnMajor, cutlass.ColumnMajor, cutlass.ColumnMajor)
        alignment = (1, 1, 1)

        tiling = ([64, 256, 16], 3, [2, 4, 1])
        data_type = [cutlass.float64, cutlass.float64, cutlass.float64, cutlass.float64]
        self.assertTrue(TestGemmOperator(GemmKind.Universal, math_inst, layout, alignment, tiling, 80, False, data_type=data_type))
        self.assertTrue(TestGemmOperator(GemmKind.Grouped, math_inst, layout, alignment, tiling, 80, True, precompute_mode=SchedulerMode.Device, data_type=data_type))
        stride_supports = [StrideSupport.Unity, StrideSupport.Strided, StrideSupport.Unity]
        results = TestConv2dOperator(math_inst, alignment, tiling, 80, stride_supports=stride_supports, data_type=data_type)
        for res in results:
            self.assertTrue(res)
    
    def test_SM80_TensorOp_16832_TN(self):
        math_inst = MathInstruction(
            [16, 8, 32], cutlass.int8, cutlass.int8, cutlass.int32,
            cutlass.OpClass.TensorOp, MathOperation.multiply_add_saturate
        )
        layout = (cutlass.RowMajor, cutlass.ColumnMajor, cutlass.ColumnMajor)
        alignment = (16, 16, 4)
        alignment_mixed = (16, 16, 16)
        tiling = ([128, 256, 64], 3, [2, 4, 1])

        data_type = [cutlass.int8, cutlass.int8, cutlass.int32, cutlass.int32]
        data_type_mixed = [cutlass.int8, cutlass.int8, cutlass.int8, cutlass.float32]

        self.assertTrue(TestGemmOperator(GemmKind.Universal, math_inst, layout, alignment, tiling, 80, False, data_type=data_type))
        self.assertTrue(TestGemmOperator(GemmKind.Grouped, math_inst, layout, alignment_mixed, tiling, 80, True, precompute_mode=SchedulerMode.Device, data_type=data_type_mixed))
        stride_supports = [StrideSupport.Strided, StrideSupport.Strided, StrideSupport.Strided]
        results = TestConv2dOperator(math_inst, alignment, tiling, 80, stride_supports=stride_supports, data_type=data_type)
        for res in results:
            self.assertTrue(res)
    
    def test_SM80_Simt_f32(self):
        math_inst = MathInstruction(
            [1, 1, 1], cutlass.float32, cutlass.float32, cutlass.float32,
            cutlass.OpClass.Simt, MathOperation.multiply_add
        )
        layout = (cutlass.RowMajor, cutlass.RowMajor, cutlass.RowMajor)
        alignment = (1, 1, 1)

        tiling = ([128, 256, 8], 4, [2, 4, 1])
        data_type = [cutlass.float32, cutlass.float32, cutlass.float32, cutlass.float32]
        self.assertTrue(TestGemmOperator(GemmKind.Universal, math_inst, layout, alignment, tiling, 80, False, data_type=data_type))
        self.assertTrue(TestGemmOperator(GemmKind.Grouped, math_inst, layout, alignment, tiling, 80, True, precompute_mode=SchedulerMode.Host, data_type=data_type))
        stride_supports = [StrideSupport.Strided, StrideSupport.Strided, StrideSupport.Strided]
        results = TestConv2dOperator(math_inst, alignment, tiling, 80, stride_supports=stride_supports, data_type=data_type)
        for res in results:
            self.assertTrue(res)

    def test_SM80_Simt_f64(self):
        math_inst = MathInstruction(
            [1, 1, 1], cutlass.float64, cutlass.float64, cutlass.float64,
            cutlass.OpClass.Simt, MathOperation.multiply_add
        )
        layout = (cutlass.RowMajor, cutlass.RowMajor, cutlass.ColumnMajor)
        alignment = (1, 1, 1)

        tiling = ([64, 128, 8], 5, [2, 2, 1])
        data_type = [cutlass.float64, cutlass.float64, cutlass.float64, cutlass.float64]
        self.assertTrue(TestGemmOperator(GemmKind.Universal, math_inst, layout, alignment, tiling, 80, False, data_type=data_type))
        self.assertTrue(TestGemmOperator(GemmKind.Grouped, math_inst, layout, alignment, tiling, 80, True, precompute_mode=SchedulerMode.Device, data_type=data_type))
        stride_supports = [StrideSupport.Strided, StrideSupport.Strided, StrideSupport.Strided]
        results = TestConv2dOperator(math_inst, alignment, tiling, 80, stride_supports=stride_supports, data_type=data_type)
        for res in results:
            self.assertTrue(res)

    def test_SM80_TensorOp_16832_Interleaved(self):
        math_inst = MathInstruction(
            [16, 8, 32], cutlass.int8, cutlass.int8, cutlass.int32,
            cutlass.OpClass.TensorOp, MathOperation.multiply_add_saturate
        )

        layout = (cutlass.ColumnMajorInterleaved32, cutlass.RowMajorInterleaved32, cutlass.ColumnMajorInterleaved32)
        alignment_mixed = (16, 16, 8)
        tiling = ([256, 64, 64], 4, [4, 1, 1])
        data_type_mixed = [cutlass.int8, cutlass.int8, cutlass.int8, cutlass.float32]

        epilogue_functor = FastLinearCombinationClamp(
            data_type_mixed[2], alignment_mixed[2]
        )

        self.assertTrue(TestGemmOperator(GemmKind.Universal, math_inst, layout, alignment_mixed, tiling, 80, False, data_type=data_type_mixed, epilogue_functor=epilogue_functor))
        stride_supports = [StrideSupport.Strided, StrideSupport.Strided, StrideSupport.Strided]
        layout = [cutlass.TensorNC32HW32, cutlass.TensorC32RSK32, cutlass.TensorNC32HW32]
        results = TestConv2dOperator(math_inst, alignment_mixed, tiling, 80, stride_supports=stride_supports, data_type=data_type_mixed, layout=layout, interleaved=True)
        for res in results:
            self.assertTrue(res)

    def SM80_SparseTensorOp_16832(self):
        pass
    def SM80_PlanarComplexTensorOp_16816(self):
        pass
    def SM80_SparseTensorOp_16816_fast_math(self):
        pass
    def SM80_TensorOp_1688_complex(self):
        pass
    def SM80_TensorOp_1688_fast_fp32_math_complex(self):
        pass
    def SM80_TensorOp_1688_rank_k(self):
        pass
    def SM80_TensorOp_1688_rank_k_complex(self):
        pass
    def SM80_TensorOp_1688_trmm(self):
        pass
    def SM80_TensorOp_1688_trmm_complex(self):
        pass
    def SM80_TensorOp_1688_symm(self):
        pass
    def SM80_TensorOp_1688_symm_complex(self):
        pass
    def SM80_TensorOp_884_complex(self):
        pass
    def SM80_TensorOp_884_complex_gaussian(self):
        pass
    def SM80_TensorOp_884_rank_k(self):
        pass
    def SM80_TensorOp_884_rank_k_complex(self):
        pass
    def SM80_TensorOp_884_rank_k_complex_gaussian(self):
        pass
    def SM80_TensorOp_884_trmm(self):
        pass
    def SM80_TensorOp_884_trmm_complex(self):
        pass
    def SM80_TensorOp_884_trmm_complex_gaussian(self):
        pass
    def SM80_TensorOp_884_symm(self):
        pass
    def SM80_TensorOp_884_symm_complex(self):
        pass
    def SM80_TensorOp_884_symm_complex_gaussian(self):
        pass
    def SM80_SparseTensorOp_16864_TN(self):
        pass
    def SM80_TensorOp_16864_TN(self):
        pass
    def SM80_SparseTensorOp_168128_TN(self):
        pass
    def SM80_TensorOp_16864_Interleaved(self):
        pass
    def SM80_TensorOp_168256(self):
        pass
    def SM80_Simt_complex(self):
        pass


def argumentParser():
    parser = argparse.ArgumentParser(description="Entrypoint for PyCutlass testing on Ampere architecture.")
    parser.add_argument("-j", "--junit_path", help="The absolute path to the directory for generating a junit xml report", default="")
    return parser.parse_args()


if __name__ == '__main__':
    pycutlass.get_memory_pool(2**20, 2**34)
    pycutlass.compiler.nvcc()
    args = argumentParser()
    if args.junit_path:
        unittest.main(argv=[''], testRunner=xmlrunner.XMLTestRunner(output=args.junit_path))
    else:
        unittest.main(argv=[''])
