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
import paddle
from paddle.distributed.fleet.model import PipelineParallel

from ....transformers.model_utils import unwrap_optimizer

try:
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizerV2,
    )
except:
    DygraphShardingOptimizerV2 = None


def shard(node_model_state, optimizer, model, hcg):
    assert DygraphShardingOptimizerV2 is not None
    group = hcg.get_sharding_parallel_group()
    cur_rank = group.rank
    split_infos = collect_split_info(optimizer, model)

    def split_func(k, v):
        param_name = k[0]
        assert param_name in split_infos
        index, padded_size, buffer_size = split_infos[param_name]
        v = pad_tensor(v, padded_size)
        assert buffer_size % group.nranks == 0, f"buffer_size {buffer_size} group.nranks {group.nranks}"
        buffer_slice = buffer_size // group.nranks
        offset = buffer_size - index % buffer_slice
        tensors = []
        tensors.append((0, slice_tensor(v, 0, offset)))
        left_size = padded_size - offset
        for _ in range((left_size + buffer_slice - 1) // buffer_slice):
            end = min(offset + buffer_slice, padded_size)
            tensors.append((0, slice_tensor(v, offset, end)))
            offset = end

        return tensors

    node_model_state.split_items(split_func).flatten_key()

    def filter_func(k):
        names, rank = k
        return rank == cur_rank

    # reshard
    node_model_state.reshard(group, filter_func)
    return node_model_state


def restore(node_model_state, optimizer, model, hcg):
    group = hcg.get_sharding_parallel_group()
    # evenly distribute param
    node_model_state.even_distribute(group)
    param_shapes = {k: v.shape for (k, v) in model.state_dict().items()}

    def merge_func(k, v):
        structure_name = k[0]
        assert structure_name in param_shapes, structure_name
        tensor_list = [e[1] for e in v]
        shape = param_shapes[structure_name]
        return merge_tensors(tensor_list, shape)

    node_model_state.collapse_key().merge_items(merge_func)
    return node_model_state


def merge_tensors(tensor_list, shape):
    assert len(tensor_list) > 0
    if len(tensor_list) == 1:
        t = tensor_list[0]
    else:
        assert len(tensor_list[0].shape) == 1
        t = paddle.concat(x=tensor_list, axis=0)
    tensor_size = np.prod(shape)
    padded_size = t._numel()
    assert padded_size >= tensor_size
    t = t._slice(0, tensor_size)
    t.get_tensor._set_dims(shape)
    return t


def pad_tensor(tensor, padded_size):
    tensor_shape = tensor.shape
    tensor_size = np.prod(tensor_shape)
    assert tensor_size < padded_size
    t = paddle.zeros([padded_size], dtype=tensor.dtype)
    tensor.flatten_()
    t[0:tensor_size] = tensor
    tensor.get_tensor()._set_dims(tensor_shape)
    return t


def slice_tensor(tensor, begin, end):
    return tensor[begin:end]


def collect_split_info(optimizer, model):
    split_infos = {}

    def gather_infos(comm_buffer):
        for (k, v) in comm_buffer._sharding_param_grad_view.items():
            index = v._index
            padded_size = v._padded_size
            buffer_size = v._grad_buffer._numel()
            split_infos[k] = (index, padded_size, buffer_size)

    if isinstance(model, PipelineParallel) and len(model._chunk_2_comm_buffers) > 0:
        for (k, v) in model._chunk_2_comm_buffers.items():
            for comm_buffer in v:
                gather_infos(comm_buffer)
    else:
        optimizer = unwrap_optimizer(optimizer, DygraphShardingOptimizerV2)
        for comm_buffer in optimizer._comm_buffer_list:
            gather_infos(comm_buffer)
    assert len(split_infos)
    return split_infos
