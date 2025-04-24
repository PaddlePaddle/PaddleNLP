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

from copy import deepcopy
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


def fp8_quantize_tensorwise(
    x, tensor_type, quantization_config=None, state=0, training=False, act_scale=None
):
    assert tensor_type in ["weight", "activation", "grad_output"], "Only support weight, activation and grad_output"
    fp8_format = quantization_config.fp8_format[tensor_type]
    qmin, qmax = (-448, 448) if fp8_format == "float8_e4m3fn" else (-57344, 57344)  
    tensor_type_to_shape_index = {"weight": 0, "activation": -1, "grad_output": -2}

    if quantization_config is not None and quantization_config.apply_hadamard:
        if getattr(infohub, "hadamard") is None:
            setattr(infohub, "hadamard", {})

        hadamard_matrix_shape = x.shape[tensor_type_to_shape_index[tensor_type]]
        if hadamard_matrix_shape in infohub.hadamard:
            hadamard_matrix, block_size = infohub.hadamard[hadamard_matrix_shape]
        else:
            hadamard_matrix, block_size = random_hadamard_matrix(hadamard_matrix_shape, x.dtype, is_block=True)
            infohub.hadamard[hadamard_matrix_shape] = (hadamard_matrix, block_size)
        target_x = hadamard_matrix.T @ x if tensor_type in ["weight", "grad_output"] else x @ hadamard_matrix
    else:
        target_x = x
        block_size = 1
    
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

    x_fp8 = target_x / scale
    x_fp8 = x_fp8.astype(fp8_format).view("int8")
    x_fp8 = x_fp8.T if tensor_type == "weight" else x_fp8
    scale.stop_gradient = True
    scale = scale / block_size if tensor_type in ["weight", "grad_output"] else scale
    return x_fp8, scale


def fp8_dequantize_tensorwise(x_fp8, scale, tensor_type, quantization_config=None):
    x_fp8 = x_fp8.view(quantization_config.fp8_format[tensor_type])
    x_fp8 = x_fp8.T if tensor_type == "weight" else x_fp8
    x = x_fp8.astype(scale.dtype) * scale
    if quantization_config.apply_hadamard:
        hadamard_matrix_shape = x.shape[0] if tensor_type == "weight" else x.shape[-1]
        hadamard_matrix, _ = infohub.hadamard[hadamard_matrix_shape]
        x = hadamard_matrix @ x if tensor_type == "weight" else x @ hadamard_matrix.T
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


def a8w8_forward(
    x, w_int8, w_scale=None, bias=None, dtype=None, quantization_config=None, state=0, training=False, act_scale=None
):
    x_int8, x_scale = quantize_tensorwise(
        x, quantization_config, bit_length=8, state=state, training=training, act_scale=act_scale
    )
    out = paddle.matmul(x_int8, w_int8.T).astype(dtype) * (x_scale * w_scale.unsqueeze(0))
    if bias is not None:
        out += bias
    return out, x_int8, x_scale


def a8w8_backward(ctx, x, grad_output, quant_weight, quant_scale, quant_x, x_scale):
    if not ctx.x_stop_gradient:
        if ctx.quantization_config.quant_input_grad:
            grad_output_int8, grad_output_scale = quantize_tensorwise(grad_output * quant_scale)
            input_grad = (
                paddle.matmul(grad_output_int8, quant_weight).astype(ctx.dtype)
                * grad_output_scale
            )
            if ctx.quantization_config.apply_hadamard:
                input_grad = input_grad @ infohub.hadamard[quant_weight.shape[-1]][0].T
        else:
            qdq_weight = dequantize_channelwise(
                quant_weight, quant_scale, apply_hadamard=ctx.quantization_config.apply_hadamard
            )
            input_grad = paddle.matmul(grad_output, qdq_weight.T)
    else:
        input_grad = None

    if not ctx.w_stop_gradient:
        if len(x.shape) == 2:
            weight_grad = paddle.matmul(x.transpose([1, 0]), grad_output)
        else:
            weight_grad = paddle.matmul(
                x.reshape([-1, x.shape[-1]]).transpose([1, 0]), grad_output.reshape([-1, grad_output.shape[-1]])
            )
    else:
        weight_grad = None

    return input_grad, weight_grad


def fp8_forward(
    x, w_fp8, w_scale=None, bias=None, dtype=None, quantization_config=None, state=0, training=False, act_scale=None
):
    x_fp8, x_scale = fp8_quantize_tensorwise(
        x, tensor_type="activation", quantization_config=quantization_config, state=state, training=training, act_scale=act_scale
    )
    # TODO(wanderHZ): support real fp8 gemm
    x_fp8 = x_fp8.view(quantization_config.fp8_format["activation"])
    w_fp8 = w_fp8.view(quantization_config.fp8_format["weight"])
    x = x_fp8.astype(dtype) * x_scale
    w = w_fp8.astype(dtype) * w_scale
    out = paddle.matmul(x, w.T).astype(dtype)
    if bias is not None:
        out += bias
    return out, x_fp8, x_scale


def fp8_backward(ctx, x, grad_output, quant_weight, quant_scale, quant_x, x_scale):
    if not ctx.x_stop_gradient:
        if ctx.quantization_config.quant_input_grad:
            grad_output_fp8, grad_output_scale = fp8_quantize_tensorwise(
                grad_output,
                tensor_type="grad_output",
                quantization_config=ctx.quantization_config,
            )
            # TODO(wanderHZ): support real fp8 gemm
            grad_output_fp8 = grad_output_fp8.view(ctx.quantization_config.fp8_format["grad_output"])
            quant_weight = quant_weight.view(ctx.quantization_config.fp8_format["weight"])
            grad_output_ = grad_output_fp8.astype(ctx.dtype) * grad_output_scale
            weight_ = quant_weight.astype(ctx.dtype) * quant_scale
            input_grad = paddle.matmul(grad_output_, weight_).astype(ctx.dtype)
            if ctx.quantization_config.apply_hadamard:
                input_grad = infohub.hadamard[grad_output.shape[-2]][0] @ input_grad
                input_grad = input_grad @ infohub.hadamard[quant_weight.shape[-1]][0].T
        else:
            qdq_weight = fp8_dequantize_tensorwise(
                quant_weight, quant_scale, tensor_type="weight", quantization_config=ctx.quantization_config
            )
            input_grad = paddle.matmul(grad_output, qdq_weight.T)
    else:
        input_grad = None

    if not ctx.w_stop_gradient:
        if ctx.quantization_config.quant_weight_grad:
            quantization_config_ = deepcopy(ctx.quantization_config)
            quantization_config_.apply_hadamard = False
            grad_output_fp8, grad_output_scale = fp8_quantize_tensorwise(
                grad_output,
                tensor_type="grad_output",
                quantization_config=quantization_config_,
            )
            # TODO(wanderHZ): support real fp8 gemm
            grad_output_fp8 = grad_output_fp8.view(ctx.quantization_config.fp8_format["grad_output"])
            quant_x = quant_x.view(ctx.quantization_config.fp8_format["activation"])
            grad_output_ = grad_output_fp8.astype(ctx.dtype) * grad_output_scale
            x_ = quant_x.astype(ctx.dtype) * x_scale
            if len(x_.shape) == 2:
                weight_grad = paddle.matmul(x_.transpose([1, 0]), grad_output_).astype(ctx.dtype)
            else:
                weight_grad = paddle.matmul(
                    x_.reshape([-1, x_.shape[-1]]).transpose([1, 0]), grad_output_.reshape([-1, grad_output_.shape[-1]])
                ).astype(ctx.dtype)
            if ctx.quantization_config.apply_hadamard:
                hadamard_matrix, block_size = infohub.hadamard[quant_x.shape[-1]]
                weight_grad = weight_grad / block_size
                weight_grad = hadamard_matrix @ weight_grad
        else:
            if len(x.shape) == 2:
                weight_grad = paddle.matmul(x.transpose([1, 0]), grad_output)
            else:
                weight_grad = paddle.matmul(
                    x.reshape([-1, x.shape[-1]]).transpose([1, 0]), grad_output.reshape([-1, grad_output.shape[-1]])
                )
    else:
        weight_grad = None
    
    return input_grad, weight_grad


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
        quant_x, x_scale = None, None
        if quantization_config.weight_quantize_algo in ["fp8linear"]:
            output, quant_x, x_scale = fp8_forward(
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
        else:
            output, quant_x, x_scale = a8w8_forward(
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
        ctx.x_stop_gradient = x.stop_gradient
        ctx.w_stop_gradient = quant_weight.stop_gradient
        ctx.b_stop_gradient = bias.stop_gradient if bias is not None else True
        ctx.save_for_backward(
            x if not quantization_config.quant_weight_grad else None,
            quant_weight,
            bias,
            quant_scale,
            quant_x if quantization_config.quant_weight_grad else None,
            x_scale if quantization_config.quant_weight_grad else None,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, quant_weight, bias, quant_scale, quant_x, x_scale = ctx.saved_tensor()

        if ctx.quantization_config.weight_quantize_algo in ["fp8linear"]:
            input_grad, weight_grad = fp8_backward(ctx, x, grad_output, quant_weight, quant_scale, quant_x, x_scale)
        else:
            input_grad, weight_grad = a8w8_backward(ctx, x, grad_output, quant_weight, quant_scale, quant_x, x_scale)

        if not ctx.b_stop_gradient:
            bias_grad = grad_output.sum(axis=[0, 1])
        else:
            bias_grad = None

        return input_grad, weight_grad, bias_grad
