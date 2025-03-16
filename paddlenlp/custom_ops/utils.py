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


import importlib

import paddle


def is_gpu():
    return paddle.device.is_compiled_with_cuda()


def is_cpu():
    # To be determined
    return False


def is_npu():
    try:
        importlib.import_module("paddle_custom_device.npu.ops")
        return True
    except:
        return False


def is_sdaa():
    try:
        importlib.import_module("paddle_sdaa.sdaa_ext")
        return True
    except:
        return False


def is_xpu():
    return paddle.device.is_compiled_with_xpu()


def custom_dispatch(func):
    def wrapper(*args, **kwargs):
        try:
            if is_cpu():
                module = importlib.import_module("paddlenlp.custom_ops.cpu")
                dispatch_func = getattr(module, func.__name__)
                return dispatch_func(*args, **kwargs)
            elif is_npu():
                module = importlib.import_module("paddlenlp.custom_ops.npu")
                dispatch_func = getattr(module, func.__name__)
                return dispatch_func(*args, **kwargs)
            elif is_sdaa():
                module = importlib.import_module("paddlenlp.custom_ops.sdaa")
                dispatch_func = getattr(module, func.__name__)
                return dispatch_func(*args, **kwargs)
            elif is_xpu():
                module = importlib.import_module("paddlenlp.custom_ops.xpu")
                dispatch_func = getattr(module, func.__name__)
                return dispatch_func(*args, **kwargs)
            elif is_gpu():
                module = importlib.import_module("paddlenlp.custom_ops.gpu")
                dispatch_func = getattr(module, func.__name__)
                return dispatch_func(*args, **kwargs)
            else:
                raise RuntimeError("No support this device")
        except AttributeError:
            raise RuntimeError(f"Device {paddle.get_device()} does not implement {func.__name__} op")

    return wrapper
