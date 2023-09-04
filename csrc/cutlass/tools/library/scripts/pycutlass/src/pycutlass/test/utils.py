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

import cutlass
from pycutlass import library, SubstituteTemplate


class Layout:
    """
    Utility class to map transpose and non-transpose terminology to row- and column-major terminology
    """
    T = cutlass.RowMajor
    N = cutlass.ColumnMajor


class LayoutCombination:
    """
    Utility class defining all combinations of row- and column-major layouts for operands to a GEMMs
    """
    NNN = (Layout.N, Layout.N, Layout.N)
    NNT = (Layout.N, Layout.N, Layout.T)
    NTN = (Layout.N, Layout.T, Layout.N)
    NTT = (Layout.N, Layout.T, Layout.T)
    TNN = (Layout.T, Layout.N, Layout.N)
    TNT = (Layout.T, Layout.N, Layout.T)
    TTN = (Layout.T, Layout.T, Layout.N)
    TTT = (Layout.T, Layout.T, Layout.T)


def get_name(layouts, alignments, element_output,
             element_accumulator, element_epilogue, cluster_shape,
             threadblock_shape, stages, element_a, element_b, arch, opclass, suffix=""):
    """
    Generates a procedural name for a test case.

    :param layouts: indexable container of layouts of A, B, and C operands
    :param alignments: indexable container of alingments of A, B, and C operands
    :param element_output: data type of the output element
    :param element_accumulator: data type used in accumulation
    :param element_epilogue: data type used in computing the epilogue
    :param cluster_shape: indexable container of dimensions of threadblock cluster to be launched
    :param threadblock_shape: indexable container of dimensions of threadblock tiles
    :param stages: number of pipeline stages to use in the kernel
    :type stages: int
    :param element_a: data type of operand A
    :param element_b: data type of operand B
    :param arch: compute capability of kernel being generated
    :type arch: int
    :param opclass: class of operation being performed (e.g., SIMT, Tensor Core)
    :type opclass: cutlass.OpClass
    :param suffix: additional string to add to the suffix of the name
    :type suffix: str

    :return: str
    """
    name_format = 'test_SM${arch}_Device_Gemm_${eA}${lA}_${eB}${lB}_${eC}${lC}_${opclass}_${acc}_${tbM}x${tbN}x${tbK}_${cM}x${cN}x${cK}_${stages}_align${aA}-${aB}-${aC}${suffix}'
    return SubstituteTemplate(name_format,
        {
            'arch': str(arch),
            'eA': library.DataTypeNames[element_a],
            'eB': library.DataTypeNames[element_b],
            'eC': library.DataTypeNames[element_output],
            'lA': library.ShortLayoutTypeNames[layouts[0]],
            'lB': library.ShortLayoutTypeNames[layouts[1]],
            'lC': library.ShortLayoutTypeNames[layouts[2]],
            'opclass': library.OpcodeClassNames[opclass],
            'acc': library.DataTypeNames[element_accumulator],
            'cM': str(cluster_shape[0]),
            'cN': str(cluster_shape[1]),
            'cK': str(cluster_shape[2]),
            'tbM': str(threadblock_shape[0]),
            'tbN': str(threadblock_shape[1]),
            'tbK': str(threadblock_shape[2]),
            'stages': str(stages) if stages is not None else 'auto',
            'aA' : str(alignments[0]),
            'aB' : str(alignments[1]),
            'aC' : str(alignments[2]),
            'suffix': '' if suffix is None else suffix
        }
    )
