# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.autograd import PyLayer

from paddlenlp.utils import infohub

from .hadamard_utils import random_hadamard_matrix


def quantize_tensorwise(x, quantization_config=None, bit_length=8, state=0, training=False, act_scale=None):
    qmax = (1 << (bit_length - 1)) - 1
    qmin = -1 * qmax - 1
    if quantization_config.apply_hadamard:
        target_x = x @ infohub.hadamard[x.shape[-1]][0]
    else:
        target_x = x

    if act_scale is not None:
        if training:
            scale = paddle.max(paddle.abs(target_x)) / qmax
            act_scale.set_value((state * act_scale + scale) / (state + 1))
            if state > quantization_config.skip_first_act_scale_step:
                scale = act_scale
        else:
            scale = act_scale
    else:
        scale = paddle.max(paddle.abs(target_x)) / qmax

    x_int8 = paddle.clip((target_x / scale).round(), qmin, qmax).astype("int8")
    return x_int8, scale


def dequantize_tensorwise(x_int8, scale, apply_hadamard=False):
    x = x_int8.astype(scale.dtype) * scale
    if apply_hadamard:
        x = x @ infohub.hadamard[x.shape[-1]][0].T
    return x


def quantize_channelwise(w, apply_hadamard=False, bit_length=8):
    qmax = (1 << (bit_length - 1)) - 1
    qmin = -1 * qmax - 1
    if apply_hadamard:
        if getattr(infohub, "hadamard") is None:
            setattr(infohub, "hadamard", {})
        if w.shape[0] in infohub.hadamard:
            hadamard_matrix, block_size = infohub.hadamard[w.shape[0]]
        else:
            hadamard_matrix, block_size = random_hadamard_matrix(w.shape[0], w.dtype, is_block=True)
            infohub.hadamard[w.shape[0]] = (hadamard_matrix, block_size)
        w = hadamard_matrix.T @ w
    else:
        block_size = 1
    scale = paddle.max(paddle.abs(w), axis=0, keepdim=True) / qmax
    w_int8 = paddle.clip((w / scale).round(), qmin, qmax).astype("int8")
    scale.stop_gradient = True
    return w_int8.T, scale.squeeze(0) / block_size


def dequantize_channelwise(w_int8, scale, apply_hadamard=False):
    w = w_int8.T.astype(scale.dtype) * scale
    if apply_hadamard:
        w = infohub.hadamard[w_int8.shape[1]][0] @ w
    return w


def a8w8_linear(
    x, w_int8, w_scale=None, bias=None, dtype=None, quantization_config=None, state=0, training=False, act_scale=None
):
    x_int8, x_scale = quantize_tensorwise(
        x, quantization_config, bit_length=8, state=state, training=training, act_scale=act_scale
    )
    out = paddle.matmul(x_int8, w_int8.T).astype(dtype) * (x_scale * w_scale.unsqueeze(0))
    if bias is not None:
        out += bias
    return out


class QATFunc(PyLayer):
    @staticmethod
    def forward(
        ctx,
        x,
        quant_weight,
        bias,
        quant_scale,
        quantization_config,
        dtype,
        state,
        training,
        act_scale,
    ):
        output = a8w8_linear(
            x,
            quant_weight,
            w_scale=quant_scale,
            bias=bias,
            dtype=dtype,
            quantization_config=quantization_config,
            state=state,
            training=training,
            act_scale=act_scale,
        )
        ctx.quantization_config = quantization_config
        ctx.dtype = dtype
        ctx.save_for_backward(x, quant_weight, bias, quant_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, quant_weight, bias, quant_scale = ctx.saved_tensor()

        if not x.stop_gradient:
            if ctx.quantization_config.quant_input_grad:
                x_int8, x_scale = quantize_tensorwise(grad_output * quant_scale)
                input_grad = (
                    paddle.matmul(x_int8, quant_weight).astype(ctx.dtype)
                    @ infohub.hadamard[quant_weight.shape[-1]][0].T
                    * x_scale
                )
            else:
                qdq_weight = dequantize_channelwise(
                    quant_weight, quant_scale, apply_hadamard=ctx.quantization_config.apply_hadamard
                )
                input_grad = paddle.matmul(grad_output, qdq_weight.T)
        else:
            input_grad = None

        if not quant_weight.stop_gradient:
            if len(x.shape) == 2:
                weight_grad = paddle.matmul(x.transpose([1, 0]), grad_output)
            else:
                weight_grad = paddle.matmul(
                    x.reshape([-1, x.shape[-1]]).transpose([1, 0]), grad_output.reshape([-1, grad_output.shape[-1]])
                )
        else:
            weight_grad = None

        if bias is not None and not bias.stop_gradient:
            bias_grad = grad_output.sum(axis=[0, 1])
        else:
            bias_grad = None

        return input_grad, weight_grad, bias_grad
