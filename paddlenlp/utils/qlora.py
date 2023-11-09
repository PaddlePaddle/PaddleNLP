# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddleslim.lc.quantizers.quant_func import (
    dequantize_8bit,
    dequantize_fp4,
    dequantize_nf4,
    quantize_8bit,
    quantize_fp4,
    quantize_nf4,
)


def weight_quantize(x, algo, double_quant=False, block_size=64, double_quant_block_size=256):
    if algo == "nf4":
        out, quant_scale = quantize_nf4(x, block_size)
    else:
        out, quant_scale = quantize_fp4(x, block_size)

    if double_quant:
        quant_sacle_offset = quant_scale.mean()
        quant_scale -= quant_sacle_offset
        qquant_scale, double_quant_scale = quantize_8bit(
            quant_scale, None, double_quant_block_size, quant_type="dynamic_fp8"
        )
        return out, (qquant_scale, double_quant_scale, quant_sacle_offset)
    else:
        return out, quant_scale


def weight_dequantize(x, algo, state, double_quant=False, block_size=64, double_quant_block_size=256):
    if double_quant:
        qquant_scale, double_quant_scale, quant_sacle_offset = state
        quant_scale = dequantize_8bit(
            qquant_scale, None, double_quant_scale, double_quant_block_size, quant_type="dynamic_fp8"
        )
        quant_scale += quant_sacle_offset
    else:
        quant_scale = state
    if algo == "nf4":
        out = dequantize_nf4(x, quant_scale, blocksize=block_size)
    else:
        out = dequantize_fp4(x, quant_scale, blocksize=block_size)
    return out


def weight_linear(x, weight, algo, state, double_quant=False, block_size=64, double_quant_block_size=256, bias=None):
    weight = (
        weight_dequantize(weight, algo, state, double_quant, block_size, double_quant_block_size)
        .cast(x.dtype)
        .reshape((x.shape[-1], -1))
    )
    out = paddle.nn.functional.linear(x, weight, bias)
    return out
