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


def random_hadamard(n, dtype):
    A = paddle.randint(low=0, high=2, shape=[n, n]).astype("float32") * 2 - 1
    Q, _ = paddle.linalg.qr(A)
    return Q.astype(dtype)


def quantize_tensorwise(x, apply_hadamard=False, bit_length=8):
    qmax = (1 << (bit_length - 1)) - 1
    qmin = -1 * qmax - 1
    if apply_hadamard:
        target_x = x @ infohub.hadamard[x.shape[-1]]
    else:
        target_x = x.clone()
    scale = paddle.max(paddle.abs(target_x)) / qmax
    x_int8 = paddle.clip((target_x / scale).round(), qmin, qmax).astype("int8")
    return x_int8, scale


def dequantize_tensorwise(x_int8, scale, apply_hadamard=False):
    x = x_int8.astype(scale.dtype) * scale
    if apply_hadamard:
        x = x @ infohub.hadamard[x.shape[-1]].T
    return x


def quantize_channelwise(w, apply_hadamard=False, bit_length=8):
    qmax = (1 << (bit_length - 1)) - 1
    qmin = -1 * qmax - 1
    if apply_hadamard:
        if getattr(infohub, "hadamard") is None:
            setattr(infohub, "hadamard", {})
        if w.shape[0] in infohub.hadamard:
            hadamard_matrix = infohub.hadamard[w.shape[0]]
        else:
            hadamard_matrix = random_hadamard(w.shape[0], w.dtype)
            infohub.hadamard[w.shape[0]] = hadamard_matrix
        w = hadamard_matrix.T @ w
    scale = paddle.max(paddle.abs(w), axis=0, keepdim=True) / qmax
    w_int8 = paddle.clip((w / scale).round(), qmin, qmax).astype("int8")
    return w_int8.T, scale.squeeze(0)


def dequantize_channelwise(w_int8, scale, apply_hadamard=False):
    w = w_int8.T.astype(scale.dtype) * scale
    if apply_hadamard:
        w = infohub.hadamard[w_int8.shape[1]] @ w
    return w


def a8w8_linear(x, w_int8, w_scale=None, bias=None, dtype=None, apply_hadamard=False):
    x_int8, x_scale = quantize_tensorwise(x, apply_hadamard, bit_length=8)
    out = paddle.matmul(x_int8, w_int8.T).astype(dtype) * x_scale * w_scale.unsqueeze(0)
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
    ):

        output = a8w8_linear(
            x,
            quant_weight,
            w_scale=quant_scale,
            bias=bias,
            dtype=dtype,
            apply_hadamard=quantization_config.apply_hadamard,
        )
        ctx.quantization_config = quantization_config
        ctx.dtype = dtype
        ctx.save_for_backward(x, quant_weight, bias, quant_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, quant_weight, bias, quant_scale = ctx.saved_tensor()

        if not x.stop_gradient:
            print("1")
            if ctx.quantization_config.quant_input_grad:
                print("grad_output, quant_scale", grad_output.shape, quant_scale.shape)
                print("grad_output*quant_scale", grad_output * quant_scale)
                x_int8, x_scale = quantize_tensorwise(
                    grad_output * quant_scale, ctx.quantization_config.apply_hadamard, bit_length=8
                )
                print("x_int8, x_scale", x_int8.shape, x_scale.shape, x_int8.dtype, x_scale.dtype)
                input_grad = paddle.matmul(x_int8, quant_weight).astype(ctx.dtype) * x_scale
                print(input_grad.dtype, input_grad.shape)
            else:
                qdq_weight = dequantize_channelwise(
                    quant_weight, quant_scale, apply_hadamard=ctx.quantization_config.apply_hadamard
                )
                input_grad = paddle.matmul(grad_output, qdq_weight.T)
        else:
            input_grad = None

        if not quant_weight.stop_gradient:
            print("2")
            weight_grad = paddle.einsum("bsh,bsd->hd", x, grad_output)
        else:
            weight_grad = None

        if bias is not None and not bias.stop_gradient:
            print("3")
            bias_grad = grad_output.sum(axis=[0, 1])
        else:
            bias_grad = None
        print("4")

        return input_grad, weight_grad, bias_grad
