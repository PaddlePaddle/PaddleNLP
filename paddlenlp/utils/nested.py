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

import collections
import copy

import paddle

TensorHolder = collections.namedtuple("TensorHolder", ["shape", "dtype", "name"])


def nested_reduce_tensor(tensor):
    if isinstance(tensor, dict):
        # copy tensor since it will be inplace modified dict
        tensor = copy.copy(tensor)
        for key in list(tensor.keys()):
            tensor[key] = nested_reduce_tensor(tensor[key])
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(nested_reduce_tensor(t) for t in tensor)

    if isinstance(tensor, paddle.Tensor):
        return TensorHolder(tensor.shape, tensor.dtype, tensor.name)

    return tensor


def nested_empty_tensor(tensor):
    if isinstance(tensor, dict):
        for key in list(tensor.keys()):
            tensor[key] = nested_empty_tensor(tensor[key])
    if isinstance(tensor, list):
        return type(tensor)(nested_empty_tensor(t) for t in tensor)

    # TensorHolder is tuple
    if isinstance(tensor, TensorHolder):
        t = paddle.empty(tensor.shape, dtype=tensor.dtype, name=tensor.name)
        t.name = tensor.name
        return t

    return tensor


def nested_broadcast_tensor(tensor, src=0, group=None):
    if isinstance(tensor, dict):
        for key in list(tensor.keys()):
            tensor[key] = nested_broadcast_tensor(tensor[key], src=src, group=group)
    if isinstance(tensor, list):
        return type(tensor)(nested_broadcast_tensor(t, src=src, group=group) for t in tensor)

    if isinstance(tensor, paddle.Tensor):
        paddle.distributed.broadcast(tensor, src=src, group=group, sync_op=True)
    return tensor


def nested_copy(inputs):
    if isinstance(inputs, dict):
        outputs = {}
        for key in list(inputs.keys()):
            outputs[key] = nested_copy(inputs[key])
        return outputs
    return inputs


def nested_copy_place(inputs, place=None, blocking=False):
    if isinstance(inputs, dict):
        outputs = {}
        for key in list(inputs.keys()):
            outputs[key] = nested_copy_place(inputs[key], place, blocking)
        return outputs
    if isinstance(inputs, paddle.Tensor):
        inputs = inputs if inputs.place == place else inputs._copy_to(place, blocking)
    return inputs
