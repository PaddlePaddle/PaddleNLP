# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
from typing import Callable

import paddle
from packaging.version import Version


def custom_fwd(func):
    def wrapper(*args, **kwargs):
        ctx = args[0]
        if len(args) == 1:
            all_args = tuple(kwargs.values())
        else:
            all_args = args[1:] + tuple(kwargs.values())

        if not hasattr(ctx, "needs_input_grad"):
            ctx.needs_input_grad = [False] * len(all_args)
        for i, arg in enumerate(all_args):
            if isinstance(arg, paddle.Tensor):
                if not arg.stop_gradient:
                    ctx.needs_input_grad[i] = True
            else:
                ctx.needs_input_grad[i] = "not_tensor"
        return func(*args, **kwargs)

    return wrapper


def custom_bwd(func):
    def wrapper(*args, **kwargs):
        ctx = args[0]
        output = func(*args, **kwargs)
        result = []
        for each, need_input_grad in zip(output, ctx.needs_input_grad):
            if isinstance(need_input_grad, str) and need_input_grad == "not_tensor":
                continue
            if need_input_grad:
                result.append(each)
            else:
                result.append(None)
        while result and result[-1] is None:
            result.pop()
        return tuple(result)

    return wrapper


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def is_autocast_enabled():
    tracer = paddle.framework._dygraph_tracer()
    return False if tracer._amp_level == paddle.core.AmpLevel.O0 else True


def get_autocast_gpu_dtype():
    from paddle.amp.auto_cast import amp_global_state

    return amp_global_state().amp_dtype
