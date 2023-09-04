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
Utility functions for converting between frontend datatypes and CUTLASS datatypes
"""

from typing import Union, Tuple

import cutlass

import pycutlass.library as library


try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False

def numpy_to_cutlass(inp):
    if numpy_available:
        if inp == np.float16:
            return cutlass.float16
        elif inp == np.float32:
            return cutlass.float32
        elif inp == np.float64:
            return cutlass.float64
        elif inp == np.int8:
            return cutlass.int8
        elif inp == np.int32:
            return cutlass.int32
    return None

try:
    import cupy as cp
    cupy_available = True
    cupy_to_cutlass_dict = {
        cp.float16: cutlass.float16,
        cp.float32: cutlass.float32,
        cp.float64: cutlass.float64
    }
except ImportError:
    cupy_available = False

def cupy_to_cutlass(inp):
    if cupy_available:
        if inp == cp.float16:
            return cutlass.float16
        elif inp == cp.float32:
            return cutlass.float32
        elif inp == cp.float64:
            return cutlass.float64
    return None

try:
    import torch
    torch_available = True
    torch_to_cutlass_dict = {
        torch.half:    cutlass.float16,
        torch.float16: cutlass.float16,
        torch.float:   cutlass.float32,
        torch.float32: cutlass.float32,
        torch.double:  cutlass.float64,
        torch.float64: cutlass.float64
    }
except ImportError:
    torch_available = False

def torch_to_cutlass(inp):
    if torch_available:
        return torch_to_cutlass_dict.get(inp, None)

try:
    import bfloat16
    bfloat16_available = True
except ImportError:
    bfloat16_available = False

def bfloat16_to_cutlass(inp):
    if bfloat16_available:
        if inp == bfloat16.bfloat16:
            return cutlass.bfloat16


def to_cutlass(inp):
    for cvt_fn in [bfloat16_to_cutlass, cupy_to_cutlass, numpy_to_cutlass, torch_to_cutlass]:
        out = cvt_fn(inp)
        if out is not None:
            return out

    raise Exception('No available conversion from type {} to a CUTLASS type.'.format(inp))
