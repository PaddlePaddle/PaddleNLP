# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct
from paddle import Tensor
from paddle._typing import ParamAttrLike

_enable_pcc = True
_ap_path = "/work/abstract_pass/Athena/tests/ap"


def is_enabled():
    return _enable_pcc


def convert_dtype_to_str(dtype):
    dtype2str_dict = {
        paddle.float32: "float32",
        paddle.float16: "float16",
        paddle.bfloat16: "bfloat16",
    }
    return dtype2str_dict.get(dtype, None)


class PccLinearAdd(paddle.nn.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_attr: ParamAttrLike | None = None,
        bias_attr: ParamAttrLike | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        weight_shape = [in_features, out_features]
        dtype = self._helper.get_default_dtype()
        self.weight = self.create_parameter(shape=weight_shape, attr=weight_attr, dtype=dtype, is_bias=False)
        self.bias = self.create_parameter(shape=[out_features], attr=bias_attr, dtype=dtype, is_bias=True)
        self.name = name
        self.fused_matmul = None
        print(f"-- PccLinearAdd: weight.shape={self.weight.shape}, weight.dtype={self.weight.dtype}")
        if self.bias is not None:
            print(f"-- PccLinearAdd: bias.shape={self.bias.shape}, bias.dtype={self.bias.dtype}")

    def forward(
        self,
        input: Tensor,
        residual1: Tensor,
        residual2: Tensor | None = None,
    ) -> Tensor:
        print(f"-- input.shape: {input.shape}, residual1.shape: {residual1.shape}")
        if self.fused_matmul is None:
            B = pct.DimVar(input.shape[0])
            M = pct.DimVar(input.shape[1])
            N = pct.DimVar(self.weight.shape[1])
            K = pct.DimVar(input.shape[2])
            DType = pct.DTypeVar("T", convert_dtype_to_str(self.weight.dtype))

            def matmul_add_residual(
                x: pct.Tensor([B, M, K], DType),
                y: pct.Tensor([K, N], DType),
                z: pct.Tensor([B, M, N], DType),
            ):
                out = paddle.matmul(x, y)
                return out + z

            def matmul_add_dual_residual(
                x: pct.Tensor([B, M, K], DType),
                y: pct.Tensor([K, N], DType),
                z1: pct.Tensor([B, M, N], DType),
                z2: pct.Tensor([B, M, N], DType),
            ):
                out = paddle.matmul(x, y)
                return out + z1 + z2

            if residual2 is None:
                self.fused_matmul = pcc.compile(matmul_add_residual, ap_path=_ap_path, train=True)
            else:
                self.fused_matmul = pcc.compile(matmul_add_dual_residual, ap_path=_ap_path, train=True)

        if residual2 is None:
            out = self.fused_matmul(input, self.weight, residual1)
        else:
            out = self.fused_matmul(input, self.weight, residual1, residual2)
        return out
