# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.framework import core
import numpy as np
from collections import OrderedDict

from paddle.framework import in_dygraph_mode, _in_legacy_dygraph

if in_dygraph_mode():
    from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_storage import ParamStorage, GradStorage
elif _in_legacy_dygraph():
    from paddle.distributed.fleet.utils.internal_storage import ParamStorage, GradStorage

from paddle.distributed.fleet.meta_parallel.sharding.sharding_utils import Type

alignment = {
    "gpu": 256,
}
align = {
    Type.fp16.value: 2,
    Type.fp32.value: 4,
}


def assign_group_by_size(parameters, group_size=256 * 1024 * 1024):
    is_sparse_gradient = [False] * len(parameters)

    if in_dygraph_mode():
        group_indices = core.eager_assign_group_by_size(
            parameters, is_sparse_gradient, [group_size, group_size])
    elif _in_legacy_dygraph():
        group_indices = core.assign_group_by_size(parameters,
                                                  is_sparse_gradient,
                                                  [group_size, group_size])

    var_groups = OrderedDict()
    for group_idx, indices in enumerate(group_indices):
        for index in indices:
            var_groups.setdefault(group_idx, []).append(parameters[index])
    return var_groups


def flatten_dense_tensors(parameters):
    _buffer_size = 0
    _param2align = {}
    dtype = parameters[0].dtype

    for param in parameters:
        assert param.trainable, "param must be trainable..."
        size = np.prod(param.shape) * align[dtype]
        remaining = size % alignment["gpu"]
        ali = 0 if remaining == 0 else alignment["gpu"] - remaining
        align_ = ali // align[dtype]
        _buffer_size += np.prod(param.shape) + align_
        _param2align[param.name] = align_

    param_storage = ParamStorage(size=_buffer_size, dtype=dtype, device="gpu")

    param_storage.add_rank_params(parameters, _param2align)

    # process gradient
    grad_storage = GradStorage(size=_buffer_size,
                               dtype=dtype,
                               device="gpu",
                               destination="0",
                               parm2align=_param2align)

    for param in parameters:
        grad_storage.add_grad(param, _param2align[param.name])

    # param_storage --> grad_storage
    param_storage.buffer._copy_gradient_from(grad_storage.buffer)
    param_storage.buffer.stop_gradient = False
    return param_storage, grad_storage


def obtain_storage(parameters):
    if len(parameters) < 1:
        return []

    var_groups = assign_group_by_size(parameters)
    storage = []
    for group_idx, parameters in var_groups.items():
        param_storage, grad_storage = flatten_dense_tensors(parameters)
        storage.append(param_storage.buffer)
    return storage
