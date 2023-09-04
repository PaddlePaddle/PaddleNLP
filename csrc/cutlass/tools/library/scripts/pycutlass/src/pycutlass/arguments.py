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
from .frontend import CupyFrontend
from typeguard import typechecked
from pycutlass.frontend import *
from typing import Union
import numpy as np
from cuda import cuda
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
from cuda import cudart
try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False


# @typechecked
class ArgumentBase:
    """
    Base class for operation arguments
    """

    def __init__(self,
                 A: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor, cp.ndarray]',
                 B: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor, cp.ndarray]',
                 C: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor, cp.ndarray]',
                 D: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor, cp.ndarray]',
                 **kwargs) -> None:
        
        # tensor_C can be interpreted as the bias with bias=True in keyword args
        if "bias" in kwargs.keys():
            self.bias = kwargs["bias"]
        else:
            # by default, tensor_C is not bias
            self.bias = False

        # preprocessing input tensors
        if isinstance(A, np.ndarray):
            self.host_D = D
            self.buffer_A = NumpyFrontend.argument(A, False)
            self.buffer_B = NumpyFrontend.argument(B, False)
            self.buffer_C = NumpyFrontend.argument(C, False)
            self.buffer_D = NumpyFrontend.argument(D, True)
            self.ptr_A = self.buffer_A.ptr
            self.ptr_B = self.buffer_B.ptr
            self.ptr_C = self.buffer_C.ptr
            self.ptr_D = self.buffer_D.ptr
            # number of elements in C
            self.tensor_c_numel = C.size
        elif torch_available and isinstance(A, torch.Tensor):
            self.ptr_A = TorchFrontend.argument(A)
            self.ptr_B = TorchFrontend.argument(B)
            self.ptr_C = TorchFrontend.argument(C)
            self.ptr_D = TorchFrontend.argument(D)
            # number of elements in C
            self.tensor_c_numel = C.numel()
        elif isinstance(A, cuda.CUdeviceptr):
            self.ptr_A = A
            self.ptr_B = B
            self.ptr_C = C
            self.ptr_D = D
            
        elif cupy_available and isinstance(A, cp.ndarray):
            self.ptr_A = CupyFrontend.argument(A)
            self.ptr_B = CupyFrontend.argument(B)
            self.ptr_C = CupyFrontend.argument(C)
            self.ptr_D = CupyFrontend.argument(D)
            # number of elements in C
            self.tensor_c_numel = C.size
        else:
            raise TypeError(
                "Unsupported Frontend. Only support numpy and torch")

    def sync(self, stream_sync=True):
        if stream_sync:
            err, = cudart.cudaDeviceSynchronize()
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("CUDA Error %s" % str(err))

        if hasattr(self, "host_D"):
            err, = cuda.cuMemcpyDtoH(
                self.host_D, self.ptr_D, self.host_D.size * self.host_D.itemsize)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("CUDA Error %s" % str(err))
