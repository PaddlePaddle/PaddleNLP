################################################################################
#
# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved
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
################################################################################

import numpy as np
from cuda import cuda
from pycutlass.memory_manager import *
from typing import TYPE_CHECKING
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    if TYPE_CHECKING:
        import torch

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False
    if TYPE_CHECKING:
        import cupy as cp


class NumpyFrontend:
    """
    Frontend node for numpy
    """

    @staticmethod
    def argument(np_tensor: 'np.ndarray', is_output: 'bool') -> cuda.CUdeviceptr:
        """Convert the input numpy tensor to CUDA device pointer

        :param np_tensor: input numpy nd array
        :param is_output: whether the tensor is output

        :return: CUDA device pointer
        """
        # copy the data to device
        if is_output:
            return device_mem_alloc(np_tensor.size * np_tensor.itemsize)
        else:
            return todevice(np_tensor)


class TorchFrontend:
    """
    Frontend node for torch
    """

    @staticmethod
    def argument(torch_tensor: 'torch.Tensor') -> cuda.CUdeviceptr:
        """Convert the input torch tensor to CUDA device pointer

        :param torch_tensor: input torch tensor
        :param is_output: whether the tensor is output

        :return: CUDA device pointer
        """

        # check the device of torch_tensor
        if not torch_tensor.is_cuda:
            torch_tensor = torch_tensor.to("cuda")

        return cuda.CUdeviceptr(torch_tensor.data_ptr())


class CupyFrontend:
    """
    Frontend node for cupy
    """

    @staticmethod
    def argument(cupy_ndarray: 'cp.ndarray'):
        return cuda.CUdeviceptr(int(cupy_ndarray.data.ptr))
