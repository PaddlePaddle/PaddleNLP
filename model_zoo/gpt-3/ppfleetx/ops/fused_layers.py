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

import distutils.util
import importlib
import os

import paddle
from paddle import _C_ops

OriginLayerNorm = paddle.nn.LayerNorm
origin_linear = paddle.incubate.nn.functional.fused_linear


def try_import(module_name, func_name=None):
    if func_name is None:
        func_name = module_name
    try:
        m = importlib.import_module(module_name)
        return getattr(m, func_name)
    except ImportError:
        return None


fast_ln = try_import("fast_ln")
fused_ln = try_import("fused_ln")


def check_normalized_shape(normalized_shape):
    if isinstance(normalized_shape, (list, tuple)):
        assert len(normalized_shape) == 1


class FusedLayerNorm(OriginLayerNorm):
    def __init__(self, normalized_shape, epsilon=1e-05, weight_attr=None, bias_attr=None, name=None):
        super().__init__(
            normalized_shape=normalized_shape, epsilon=epsilon, weight_attr=weight_attr, bias_attr=bias_attr
        )
        check_normalized_shape(self._normalized_shape)

    def forward(self, input):
        return fused_ln(input, self.weight, self.bias, self._epsilon)[0]


class FastLayerNorm(OriginLayerNorm):
    def __init__(self, normalized_shape, epsilon=1e-05, weight_attr=None, bias_attr=None, name=None):
        super().__init__(
            normalized_shape=normalized_shape, epsilon=epsilon, weight_attr=weight_attr, bias_attr=bias_attr
        )
        check_normalized_shape(self._normalized_shape)

    def forward(self, input):
        return fast_ln(input, self.weight, self.bias, self._epsilon)[0]


class FusedLinearWithGradAdd(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        ctx.need_bias = bias is not None and not bias.stop_gradient
        y = origin_linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        x, weight, bias = ctx.saved_tensor()
        x_grad = paddle.matmul(y_grad, weight, transpose_y=True)
        if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
            weight.main_grad, bias.main_grad = _C_ops.fused_linear_param_grad_add(
                x, y_grad, weight.main_grad, bias.main_grad, True
            )
            return x_grad, None, None
        else:
            weight_grad, bias_grad = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False)
            return x_grad, weight_grad, bias_grad


def strtobool(s):
    return True if distutils.util.strtobool(s) else False


def get_env(env_name, default_value=False):
    return strtobool(os.getenv(env_name, str(default_value)))


def mock_layers():
    if get_env("USE_FAST_LN"):
        paddle.nn.LayerNorm = FastLayerNorm
    elif get_env("USE_FUSED_LN"):
        paddle.nn.LayerNorm = FusedLayerNorm

    if get_env("USE_LINEAR_WITH_GRAD_ADD"):
        paddle.nn.functional.linear = FusedLinearWithGradAdd.apply
        paddle.incubate.nn.functional.fused_linear = FusedLinearWithGradAdd.apply
