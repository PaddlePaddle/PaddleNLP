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


def rand_int_tensor(low, high, shape):
    return paddle.randint(
        low,
        high,
        shape=shape,
        dtype=paddle.int64,
    )


def clone_tensor(x):
    y = x.clone()
    return y


def clone_input(x):
    def paddle_clone(x):
        y = paddle.clone(x)
        if x.is_leaf:
            y.stop_gradient = x.stop_gradient
        if x.is_leaf and x.grad is not None:
            y.grad = clone_input(x.grad)
        return y

    with paddle.no_grad():
        result = paddle.empty(x.shape, dtype=x.dtype)
        result.copy_(x.clone(), True)
        if x.is_leaf:
            result.stop_gradient = x.stop_gradient
        if x.is_leaf and x.grad is not None:
            result.grad = clone_input(x.grad)
        return result


def clone_inputs(example_inputs):
    if isinstance(example_inputs, dict):
        res = dict(example_inputs)
        for key, value in res.items():
            assert isinstance(value, paddle.Tensor)
            res[key] = clone_input(value)
        return res

    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], paddle.Tensor):
            res[i] = clone_input(res[i])
    return res
