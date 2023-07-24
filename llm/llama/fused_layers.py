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
import os

import paddle
from paddle import _C_ops

origin_linear = paddle.incubate.nn.functional.fused_linear


class FusedLinearWithGradAdd(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        y = origin_linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        x, weight, bias = ctx.saved_tensor()
        x_grad = paddle.matmul(y_grad, weight, transpose_y=True)

        if bias is None:
            if hasattr(weight, "main_grad"):
                weight.main_grad, _ = _C_ops.fused_linear_param_grad_add(x, y_grad, weight.main_grad, None, True)
                return x_grad, None
            else:
                if weight.grad is not None:
                    weight_grad, _ = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False)
                    return x_grad, None
                else:
                    weight_grad, _ = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False)
                    return x_grad, weight_grad

        if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
            weight.main_grad, bias.main_grad = _C_ops.fused_linear_param_grad_add(
                x, y_grad, weight.main_grad, bias.main_grad, True
            )
            return x_grad, None, None
        else:
            if weight.grad is not None:
                assert bias.grad is not None
                weight_grad, bias_grad = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False)
                return x_grad, None, None
            else:
                weight_grad, bias_grad = _C_ops.fused_linear_param_grad_add(x, y_grad, None, None, False)
                return x_grad, weight_grad, bias_grad


def strtobool(s):
    return True if distutils.util.strtobool(s) else False


def get_env(env_name, default_value=False):
    return strtobool(os.getenv(env_name, str(default_value)))


def mock_layers():
    if get_env("USE_LINEAR_WITH_GRAD_ADD"):
        paddle.nn.functional.linear = FusedLinearWithGradAdd.apply
        paddle.incubate.nn.functional.fused_linear = FusedLinearWithGradAdd.apply
