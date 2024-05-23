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

import numpy as np


def split_qkv_gate_up_tensor_parallel_weight(
    weight, tensor_parallel_degree, tensor_parallel_rank, hidden_size, intermediate_size, num_heads
):
    """
    [QKV, G, U]  -> [QKV1, G1, U1], [QKV2, G2, U2]

    Only support split Column dim.

    """
    assert weight.shape[-1] == 3 * hidden_size + 2 * intermediate_size, "input weight size dismatch!"

    if "PySafeSlice" in str(type(weight)):
        QKV = weight[:, 0 : 3 * hidden_size]
        G = weight[:, 3 * hidden_size : 3 * hidden_size + intermediate_size]
        U = weight[:, 3 * hidden_size + intermediate_size :]

        # Split QKV
        block_size = 3 * hidden_size // tensor_parallel_degree
        start = tensor_parallel_rank * block_size
        stop = (tensor_parallel_rank + 1) * block_size
        assert (
            3 * hidden_size % tensor_parallel_degree == 0
        ), f"The choosen size {hidden_size} is not compatible with sharding on {tensor_parallel_degree} shards"
        qkv = QKV[:, start:stop]

        # Split G, U
        block_size = intermediate_size // tensor_parallel_degree
        start = tensor_parallel_rank * block_size
        stop = (tensor_parallel_rank + 1) * block_size
        assert (
            intermediate_size % tensor_parallel_degree == 0
        ), f"The choosen size {intermediate_size} is not compatible with sharding on {tensor_parallel_degree} shards"
        g = G[:, start:stop]
        u = U[:, start:stop]

        tensor = np.concatenate([qkv, g, u], axis=-1)
        return tensor

    QKV, G, U = np.split(weight, [hidden_size * 3, hidden_size * 3 + intermediate_size], axis=-1)
    assert (
        weight.shape[-1] % tensor_parallel_degree == 0
    ), f"The choosen size {weight.shape[-1]} is not compatible with sharding on {tensor_parallel_degree} shards, for tensor shape {weight.shape}"
    sQKV, sG, sU = [np.split(item, tensor_parallel_degree, axis=-1) for item in [QKV, G, U]]
    qkv, g, u = [item[tensor_parallel_rank] for item in [sQKV, sG, sU]]
    tensor = np.concatenate([qkv, g, u], axis=-1)
    return tensor


def merge_qkv_gate_up_tensor_parallel_weight(weight_list, tensor_parallel_degree, hidden_size, intermediate_size):
    """
    [QKV1, G1, U1], [QKV2, G2, U2] -> [Q, K, V, G, U]

    Only support split Column dim.

    """
    bhs = hidden_size // tensor_parallel_degree
    bis = intermediate_size // tensor_parallel_degree

    qkv_l, g_l, u_l = [], [], []
    for weight in weight_list:
        qkv, g, u = np.split(weight, [bhs * 3, bhs * 3 + bis], axis=-1)
        qkv_l.append(qkv)
        g_l.append(g)
        u_l.append(u)
    QKV, G, U = [np.concatenate(item, axis=-1) for item in [qkv_l, g_l, u_l]]
    tensor = np.concatenate([QKV, G, U], axis=-1)
    return tensor


def qkv_gate_up_proj_split_fn(tensor_parallel_degree, tensor_parallel_rank, hidden_size, intermediate_size, num_heads):
    def fn(x):
        if x is None:
            return None
        x = split_qkv_gate_up_tensor_parallel_weight(
            x,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
        )
        return x

    return fn


def qkv_gate_up_proj_merge_fn(tensor_parallel_degree, tensor_parallel_rank, hidden_size, intermediate_size, num_heads):
    def fn(x):
        if x is None:
            return None
        x = merge_qkv_gate_up_tensor_parallel_weight(
            x,
            tensor_parallel_degree=tensor_parallel_degree,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        return x

    return fn


def split_o_tensor_parallel_weight(
    weight, tensor_parallel_degree, tensor_parallel_rank, hidden_size, intermediate_size
):
    """
    Only support split Row dim.
    """
    assert weight.shape[0] == intermediate_size + hidden_size, "input weight size dismatch!"
    if "PySafeSlice" in str(type(weight)):
        A = weight[:intermediate_size]
        block_size = intermediate_size // tensor_parallel_degree
        start = tensor_parallel_rank * block_size
        stop = (tensor_parallel_rank + 1) * block_size
        assert (
            intermediate_size % tensor_parallel_degree == 0
        ), f"The choosen size {intermediate_size} is not compatible with sharding on {tensor_parallel_degree} shards"
        a = A[start:stop]

        B = weight[intermediate_size:]
        block_size = hidden_size // tensor_parallel_degree
        start = tensor_parallel_rank * block_size
        stop = (tensor_parallel_rank + 1) * block_size
        assert (
            hidden_size % tensor_parallel_degree == 0
        ), f"The choosen size {hidden_size} is not compatible with sharding on {tensor_parallel_degree} shards"
        b = B[start:stop]
        tensor = np.concatenate([a, b], axis=0)
        return tensor

    A, B = np.split(weight, [intermediate_size], axis=0)
    assert (
        weight.shape[0] % tensor_parallel_degree == 0
    ), f"The choosen size {weight.shape[-1]} is not compatible with sharding on {tensor_parallel_degree} shards, for tensor shape {weight.shape}"
    sA = np.split(A, tensor_parallel_degree, axis=0)
    sB = np.split(B, tensor_parallel_degree, axis=0)
    a, b = [item[tensor_parallel_rank] for item in [sA, sB]]
    tensor = np.concatenate([a, b], axis=0)
    return tensor


def merge_o_tensor_parallel_weight(weight_list, tensor_parallel_degree, hidden_size, intermediate_size):
    bis = intermediate_size // tensor_parallel_degree
    a_l, b_l = [], []
    for weight in weight_list:
        a, b = np.split(weight, [bis], axis=0)
        a_l.append(a)
        b_l.append(b)
    A, B = [np.concatenate(item, axis=0) for item in [a_l, b_l]]
    tensor = np.concatenate([A, B], axis=0)
    return tensor


def o_proj_split_fn(tensor_parallel_degree, tensor_parallel_rank, hidden_size, intermediate_size):
    def fn(x):
        if x is None:
            return None
        x = split_o_tensor_parallel_weight(
            x,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        return x

    return fn


def o_proj_merge_fn(tensor_parallel_degree, tensor_parallel_rank, hidden_size, intermediate_size):
    def fn(x):
        if x is None:
            return None
        x = merge_o_tensor_parallel_weight(
            x,
            tensor_parallel_degree=tensor_parallel_degree,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        return x

    return fn
