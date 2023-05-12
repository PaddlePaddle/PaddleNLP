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
import paddle.nn as nn
from paddle.autograd import PyLayer

paddle.set_default_dtype("float64")


def print_t(x, name="x"):
    print(f"{name}: {x.shape}, max: {x.abs().max().item()} mean: {x.abs().mean().item()}")


class rms_norm_py(PyLayer):
    @staticmethod
    def forward(ctx, x_in, w, eps):
        # hidden_states_in = hidden_states
        with paddle.amp.auto_cast(False):
            # variance = x_in.astype("float32").pow(2).mean(-1, keepdim=True) + eps
            variance = x_in.pow(2).mean(-1, keepdim=True) + eps
            var_rsqrt = paddle.rsqrt(variance)
            x_out = var_rsqrt.astype(x_in.dtype) * x_in

        x_out = x_out * w
        ctx.save_for_backward(var_rsqrt, w, x_in)

        return x_out

    @staticmethod
    def backward(ctx, dy):
        # https://www.derivative-calculator.net/
        # 1/sqrt( (x^2 + x2^2 + x3^2)/3 + eps) * x * w
        # var_rsqrt - x^2 /n * var_rsqrt^3

        # [.., x] [.., x, h] [.., x, h]

        # raise ValueError()
        (
            var_rsqrt,
            w,
            x_in,
        ) = ctx.saved_tensor()
        #
        print_t(var_rsqrt, "var_rsqrt")
        d_x_in = var_rsqrt - var_rsqrt.pow(3) / w.shape[-1] * x_in.pow(2)
        d_x_in = d_x_in.astype(dy.dtype) * w * dy

        d_w = (var_rsqrt.astype(dy.dtype) * x_in * dy).sum(axis=(0, 1))

        print("dw max:", d_w.abs().max().item())
        print("dw min:", d_w.abs().min().item())

        print("d_x_in max:", d_x_in.abs().max().item())
        print("d_x_in min:", d_x_in.abs().min().item())

        return d_x_in, d_w


def rms_norm(x_in, w, eps):
    var = x_in.pow(2).mean(-1, keepdim=True)
    return paddle.rsqrt(var + eps) * x_in * w


class LlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        # return paddle.nn.functional.layer_norm(hidden_states, self.hidden_size, weight=self.weight, bias=None, epsilon=self.variance_epsilon)
        # return rms_norm_py.apply(hidden_states, self.weight, self.variance_epsilon)

        if self.config.fp16_opt_level is not None:
            with paddle.amp.auto_cast(False):
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        else:
            with paddle.amp.auto_cast(False):
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = (
                    paddle.rsqrt(variance + self.variance_epsilon).astype(hidden_states.dtype) * hidden_states
                )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


import unittest

import numpy as np


class TestRmsNorm(unittest.TestCase):
    def setUp(self):
        paddle.seed(1023)
        width = 4096
        self.eps = 1e-6
        self.x_in = paddle.randn([1, 1024, width])
        # self.w = paddle.randn([1, 1024, width])
        self.w = paddle.normal(mean=0.0, std=0.02, shape=[width])

    def create(self):
        paddle.seed(1023)
        width = 4096
        eps = 1e-6
        x_in = paddle.randn([1, 1024, width]) + 0.02
        # self.w = paddle.randn([1, 1024, width])
        w = paddle.normal(mean=0.2, std=0.02, shape=[width])
        # w = paddle.ones(shape=[width]) * 3

        x_in.stop_gradient = False
        w.stop_gradient = False
        return x_in, w, eps

    def test_forward(self):
        ret1 = rms_norm(self.x_in, self.w, self.eps)
        ret2 = rms_norm_py.apply(self.x_in, self.w, self.eps)
        print(ret1)
        np.testing.assert_equal(ret1.numpy(), ret2.numpy())

    def test_backward(self):
        x_in, w, eps = self.create()
        ret1 = rms_norm(x_in, w, eps)
        ret1.sum().backward()
        x_in_g_1 = x_in.grad.numpy()
        w_g_1 = w.grad.numpy()

        x_in, w, eps = self.create()
        ret2 = rms_norm_py.apply(x_in, w, eps)
        ret2.sum().backward()
        x_in_g_2 = x_in.grad.numpy()
        w_g_2 = w.grad.numpy()

        np.testing.assert_equal(ret1.numpy(), ret2.numpy())
        np.testing.assert_equal(w_g_1, w_g_2)
        np.testing.assert_equal(x_in_g_1, x_in_g_2)
        np.testing.assert_equal(np.abs(x_in_g_1 * 1000).sum(), np.abs(x_in_g_2 * 1000).sum())
