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

import numpy as np
import cutlass
from pycutlass.library import TensorDescription
from typing import Union
from bfloat16 import bfloat16
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

class ReferenceModule:
    def __init__(self, A: TensorDescription, B: TensorDescription, C: TensorDescription) -> None:
        self.layout_A = A.layout
        self.layout_B = B.layout
        self.layout_C = C.layout
    
    def run(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, problem_size: cutlass.gemm.GemmCoord, alpha: float=1.0, beta: float=0.0, bias=False, batch=1):
        """
        Compute the reference result on CPU
        Args:
            A: dense operator with shape (M, K) in row-major and (K, M) in column-major
            B: dense operator with shape (K, N) in row-major and (N, K) in column-major
            C: dense operator with shape (M, N) in row-major and (N, M) in column-major
        """
        M, N, K = problem_size.m(), problem_size.n(), problem_size.k()
        if isinstance(A, np.ndarray):
            if self.layout_A == cutlass.RowMajor:
                A_row = np.reshape(A, newshape=(batch, M, K))
            else:
                A_col = np.reshape(A, newshape=(batch, K, M))
                A_row = np.transpose(A_col, axes=(0, 2, 1))
            
            if self.layout_B == cutlass.RowMajor:
                B_row = np.reshape(B, newshape=(batch, K, N))
            else:
                B_col = np.reshape(B, newshape=(batch, N, K))
                B_row = np.transpose(B_col, axes=(0, 2, 1))

            if self.layout_C == cutlass.RowMajor:
                if bias:
                    C_row = np.reshape(C, newshape=(batch, 1, N))
                else:
                    C_row = np.reshape(C, newshape=(batch, M, N))
            else:
                if bias:
                    C_row = np.reshape(C, newshape=(batch, M, 1))
                else:
                    C_col = np.reshape(C, newshape=(batch, N, M))
                    C_row = np.transpose(C_col, axes=(0, 2, 1))
            
            if A_row.dtype == bfloat16:
                # numpy's einsum doesn't support bfloat16
                out_row = np.einsum("bik,bkj->bij", A_row.astype(np.float32), B_row.astype(np.float32)) * alpha + C_row * beta
                out_row = out_row.astype(C_row.dtype)
            else:
                out_row = np.einsum("bik,bkj->bij", A_row, B_row) * alpha + C_row * beta

            if self.layout_C == cutlass.ColumnMajor:
                out = np.transpose(out_row, axes=(0, 2, 1))
            else:
                out = out_row
            
            return out.ravel()

        elif isinstance(A, torch.Tensor):
            if self.layout_A == cutlass.RowMajor:
                A_row = A.view((M, K))
            else:
                A_col = A.view((K, M))
                A_row = torch.permute(A_col, (1, 0))
            
            if self.layout_B == cutlass.RowMajor:
                B_row = B.view((K, N))
            else:
                B_col = B.view((N, K))
                B_row = torch.permute(B_col, (1, 0))

            if self.layout_C == cutlass.RowMajor:
                C_row = C.view((M, N))
            else:
                C_col = C.view((N, M))
                C_row = torch.permute(C_col, (1, 0))
            
            out_row = torch.matmul(A_row, B_row) * alpha + C_row * beta

            if self.layout_C == cutlass.ColumnMajor:
                out = torch.permute(out_row, (1, 0))
            else:
                out = out_row
            
            return torch.flatten(out)



#####################################################################################################
# Conv2d
#####################################################################################################

if torch_available:
    class Conv2dReferenceModule:
        def __init__(self, A: TensorDescription, B: TensorDescription, C: TensorDescription, kind: cutlass.conv.Operator.fprop) -> None:
            self.layout_A = A.layout
            self.layout_B = B.layout
            self.layout_C = C.layout
            self.kind = kind
        
        def run(self, 
            A: Union[np.ndarray, torch.Tensor],
            B: Union[np.ndarray, torch.Tensor],
            C: Union[np.ndarray, torch.Tensor], problem_size, alpha=1.0, beta=0.0, bias=False) -> np.ndarray:
            """
            Compute the reference result on CPU
            """
            n = problem_size.N
            h = problem_size.H
            w = problem_size.W
            c = problem_size.C

            k = problem_size.K
            r = problem_size.R
            s = problem_size.S

            p = problem_size.P
            q = problem_size.Q

            stride_h = problem_size.stride_h
            stride_w = problem_size.stride_w

            pad_h = problem_size.pad_h
            pad_w = problem_size.pad_w

            dilation_h = problem_size.dilation_h
            dilation_w = problem_size.dilation_w

            groups = problem_size.groups

            if isinstance(A, np.ndarray):
                # the pytorch activation layout is NCHW
                #             weight layout is Cout Cin Kh Kw (also NCHW)
                if self.layout_A == cutlass.TensorNHWC:
                    A_nhwc = np.reshape(A, newshape=(n, h, w, c))
                    A_torch_nhwc = torch.from_numpy(A_nhwc).to("cuda")
                    A_torch_nchw = torch.permute(A_torch_nhwc, (0, 3, 1, 2))
                
                if self.layout_B == cutlass.TensorNHWC:
                    B_nhwc = np.reshape(B, newshape=(k, r, s, c))
                    B_torch_nhwc = torch.from_numpy(B_nhwc).to("cuda")
                    B_torch_nchw = torch.permute(B_torch_nhwc, (0, 3, 1, 2))
                
                if self.layout_C == cutlass.TensorNHWC:
                    C_nhwc = np.reshape(C, newshape=(n, p, q, k))
                    C_torch_nhwc = torch.from_numpy(C_nhwc).to("cuda")
                    C_torch_nchw = torch.permute(C_torch_nhwc, (0, 3, 1, 2))
            
            elif isinstance(A, torch.Tensor):
                if self.kind == cutlass.conv.Operator.wgrad:
                    if self.layout_A == cutlass.TensorNHWC:
                        A_nhwc = A.view((n, p, q, k))
                        A_torch_nchw = torch.permute(A_nhwc, (0, 3, 1, 2))
                
                    if self.layout_B == cutlass.TensorNHWC:
                        B_nhwc = B.view((n, h, w, c))
                        B_torch_nchw = torch.permute(B_nhwc, (0, 3, 1, 2))
                    
                    if self.layout_C == cutlass.TensorNHWC:
                        if bias:
                            C_nhwc = C.view((1, 1, 1, c))
                        else:
                            C_nhwc = C.view((k, r, s, c))
                        C_torch_nchw = torch.permute(C_nhwc, (0, 3, 1, 2))
                elif self.kind == cutlass.conv.Operator.dgrad:
                    if self.layout_A == cutlass.TensorNHWC:
                        A_nhwc = A.view((n, p, q, k))
                        A_torch_nchw = torch.permute(A_nhwc, (0, 3, 1, 2))
                    
                    if self.layout_B == cutlass.TensorNHWC:
                        B_nhwc = B.view((k, r, s, c))
                        B_torch_nchw = torch.permute(B_nhwc, (0, 3, 1, 2))
                    
                    if self.layout_C == cutlass.TensorNHWC:
                        if bias:
                            C_nhwc = C.view((1, 1, 1, c))
                        else:
                            C_nhwc = C.view((n, h, w, c))
                        C_torch_nchw = torch.permute(C_nhwc, (0, 3, 1, 2))
                else:
                    if self.layout_A == cutlass.TensorNHWC:
                        A_nhwc = A.view((n, h, w, c))
                        A_torch_nchw = torch.permute(A_nhwc, (0, 3, 1, 2))
                    
                    if self.layout_B == cutlass.TensorNHWC:
                        B_nhwc = B.view((k, r, s, c))
                        B_torch_nchw = torch.permute(B_nhwc, (0, 3, 1, 2))
                    
                    if self.layout_C == cutlass.TensorNHWC:
                        if bias:
                            C_nhwc = C.view((1, 1, 1, k))
                        else:
                            C_nhwc = C.view((n, p, q, k))
                        C_torch_nchw = torch.permute(C_nhwc, (0, 3, 1, 2))

            if self.kind == cutlass.conv.Operator.fprop:
                D_torch_nchw = alpha * torch.nn.functional.conv2d(
                    A_torch_nchw, B_torch_nchw, stride=(stride_h, stride_w),
                    padding=(pad_h, pad_w), dilation=(dilation_h, dilation_w), groups=groups) + beta * C_torch_nchw
            elif self.kind == cutlass.conv.Operator.dgrad:
                D_torch_nchw = alpha * torch.nn.grad.conv2d_input(
                    (n, c, h, w), B_torch_nchw, A_torch_nchw, padding=(pad_h, pad_w), stride=(stride_h, stride_w)
                ).to(torch.float32) + beta * C_torch_nchw
            elif self.kind == cutlass.conv.Operator.wgrad:
                D_torch_nchw = alpha * torch.nn.grad.conv2d_weight(
                    B_torch_nchw, (k, c, r, s), A_torch_nchw, padding=(pad_h, pad_w), stride=(stride_h, stride_w)
                ).to(torch.float32) + beta * C_torch_nchw


            if self.layout_C == cutlass.TensorNHWC:
                if isinstance(A, np.ndarray):
                    D_torch_out = torch.permute(D_torch_nchw, (0, 2, 3, 1)).detach().cpu().numpy()
                elif isinstance(A, torch.Tensor):
                    D_torch_out = torch.permute(D_torch_nchw, (0, 2, 3, 1))
            
            return D_torch_out.flatten()
