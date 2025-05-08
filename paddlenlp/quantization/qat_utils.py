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

from .hadamard_utils import apply_hadamard_matmul

QMAX_QMIN_MAPPING = {
    "a8w8linear_activation": (-127, 128),
    "a8w4linear_activation": (-127, 128),
    "a8w8linear_weight": (-127, 128),
    "a8w4linear_weight": (-8, 7),
}


def quantize(
    x,
    weight_quantize_algo,
    tensor_type,
    quantization_config,
    apply_hadamard=False,
    side="right",
    act_scale=None,
    state=0,
    training=False,
    group=None,
):
    if apply_hadamard:
        target_x, block_size = apply_hadamard_matmul(x, side, quantization_config)
    else:
        target_x = x
        block_size = 1
    qmin, qmax = QMAX_QMIN_MAPPING[weight_quantize_algo + "_" + tensor_type]
    if tensor_type == "activation":
        if act_scale is not None:
            if training:
                scale = paddle.max(paddle.abs(target_x)) / qmax
                act_scale.set_value((state * act_scale + scale) / (state + 1))
                if state > quantization_config.apply_online_actscale_step:
                    scale = act_scale
            else:
                scale = act_scale
        else:
            scale = paddle.max(paddle.abs(target_x)) / qmax
        if weight_quantize_algo in ["a8w8linear", "a8w4linear"]:
            quant_x = paddle.clip((target_x / scale).round(), qmin, qmax).astype("int8")
        else:
            raise NotImplementedError(f"Unknown {weight_quantize_algo}.")
    elif tensor_type == "weight":
        if weight_quantize_algo in ["a8w8linear", "a8w4linear"]:
            # channelwise
            scale = paddle.max(paddle.abs(target_x), axis=0, keepdim=True) / qmax
            if group is not None:
                paddle.distributed.all_reduce(scale, op=paddle.distributed.ReduceOp.MAX, group=group, sync_op=True)
            quant_x = paddle.clip((target_x / scale).round(), qmin, qmax).astype("int8").T
            scale.stop_gradient = True
            scale = scale.squeeze(0) / block_size
        else:
            raise NotImplementedError(f"Unknown {weight_quantize_algo}.")
    else:
        raise NotImplementedError(f"Unknown {tensor_type}.")
    return quant_x, scale


def dequantize(quant_x, scale, tensor_type, weight_quantize_algo, apply_hadamard=False, side="right"):
    if tensor_type == "weight":
        if weight_quantize_algo in ["a8w8linear", "a8w4linear"]:
            x = quant_x.T.astype(scale.dtype)
            if apply_hadamard:
                x, block_size = apply_hadamard_matmul(x, side, dequant=True)
                x *= scale / block_size
            else:
                x *= scale
    else:
        raise NotImplementedError(f"Unknown {tensor_type}.")
    return x


def int8_forward(
    x,
    quant_w,
    scale_w,
    weight_quantize_algo,
    bias=None,
    quantization_config=None,
    state=0,
    training=False,
    act_scale=None,
):
    quant_x, scale_x = quantize(
        x=x,
        weight_quantize_algo=weight_quantize_algo,
        tensor_type="activation",
        quantization_config=quantization_config,
        apply_hadamard=quantization_config.apply_hadamard,
        side="right",
        act_scale=act_scale,
        state=state,
        training=training,
    )

    out = paddle.matmul(quant_x, quant_w.T).astype(scale_w.dtype) * (scale_x * scale_w)
    if bias is not None:
        out += bias
    return out


def int8_backward(ctx, grad_output):
    x, quant_weight, bias, quant_scale = ctx.saved_tensor()

    if not x.stop_gradient:
        if ctx.quantization_config.quant_input_grad:
            raise NotImplementedError("Not yet support quant_input_grad")
        else:
            qdq_weight = dequantize(
                quant_weight, quant_scale, "weight", ctx.quantization_config.apply_hadamard, side="left"
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


class QATFunc(PyLayer):
    @staticmethod
    def forward(
        ctx, x, quant_weight, bias, quant_scale, quantization_config, state, training, act_scale, weight_quantize_algo
    ):
        if weight_quantize_algo in ["a8w8linear", "a8w4linear"]:
            output = int8_forward(
                x=x,
                quant_w=quant_weight,
                scale_w=quant_scale,
                weight_quantize_algo=weight_quantize_algo,
                bias=bias,
                quantization_config=quantization_config,
                state=state,
                training=training,
                act_scale=act_scale,
            )
            ctx.quantization_config = quantization_config
            ctx.save_for_backward(x, quant_weight, bias, quant_scale)
        ctx.weight_quantize_algo = weight_quantize_algo
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.weight_quantize_algo in ["a8w8linear", "a8w4linear"]:
            input_grad, weight_grad, bias_grad = int8_backward(ctx, grad_output)

        return input_grad, weight_grad, bias_grad
