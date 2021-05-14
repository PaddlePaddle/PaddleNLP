# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import in_dygraph_mode
from paddle.distributed.fleet import fleet

__all__ = [
    'guard',
    'ParallelEmbedding',
    'ParallelLinear',
    'ColumnParallelLiner',
    'RowParallelLiner',
]


def guard(device):
    def decorator(Layer):
        class WrapperClass(Layer):
            def __init__(self, *args, **kw):
                with paddle.static.device_guard(device):
                    print("Init {} on {}".format(Layer.__name__, device))
                    return super().__init__(*args, **kw)

            def forward(self, *args, **kw):
                with paddle.static.device_guard(device):
                    print("Forward {} on {}".format(Layer.__name__, device))
                    return super().forward(*args, **kw)

        return WrapperClass

    return decorator


class ParallelEmbedding(nn.Layer):
    """
    Parallel Embedding
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 num_partitions,
                 padding_idx=None,
                 weight_attr=None,
                 name=None):
        super().__init__()
        size = (num_embeddings, embedding_dim)
        if in_dygraph_mode():
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
        else:
            assert fleet._role_maker, ("To use paddle.distributed.split, "
                                       "you must call fleet.init() firstly.")
            rank = fleet.worker_index()
            nranks = fleet.worker_num()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.inner_rank = inner_rank
        self.num_partitions = num_partitions

        per_part_size = (size[0] + num_partitions - 1) // num_partitions
        last_part_size = size[0] - per_part_size * (num_partitions - 1)
        if inner_rank == num_partitions - 1: per_part_size = last_part_size
        per_part_size += 1  # make the last row as the padding index

        self.origin_size = size
        if not name:
            self.name = "emb_rank_%d" % inner_rank
        else:
            self.name = name + "_rank_%d" % inner_rank
        self.per_part_embeddings = per_part_size
        self.origin_num_embeddings = self.origin_size[0]
        self.weight_attr = weight_attr

        self.embedding = paddle.nn.Embedding(
            self.per_part_embeddings,
            self.origin_size[1],
            padding_idx=self.per_part_embeddings - 1,
            sparse=False,
            weight_attr=self.weight_attr,
            name=self.name)

        self.embedding.weight.is_distributed = True
        # Alias for nn.Embedding
        self.weight = self.embedding.weight
        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[self.embedding.weight.name].is_distributed = True
        main_block.vars[self.embedding.weight.name].is_distributed = True

    def forward(self, x):
        origin_input_shape = x.shape
        if len(origin_input_shape) == 2:
            x = paddle.unsqueeze(x, axis=-1)
        else:
            assert origin_input_shape[-1] == 1, (
                "The last dimension size of x must be 1.")

        x_shard = paddle.shard_index(x, self.origin_num_embeddings,
                                     self.num_partitions, self.inner_rank,
                                     self.per_part_embeddings - 1)
        if len(origin_input_shape) == 2:
            x_shard = paddle.squeeze(x_shard, axis=-1)

        emb_out = self.embedding(x_shard)

        paddle.distributed.all_reduce(emb_out, group=None)
        return emb_out


class ParallelLinear(nn.Layer):
    """
    Parallel Linear
    """

    def __init__(self,
                 size,
                 axis,
                 num_partitions=1,
                 gather_out=True,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__()

        if in_dygraph_mode():
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
        else:
            assert fleet._role_maker, ("To use paddle.distributed.split, "
                                       "you must call fleet.init() firstly.")
            rank = fleet.worker_index()
            nranks = fleet.worker_num()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.axis = axis
        if axis == 0:
            assert size[0] % num_partitions == 0, (
                "Number of rows of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[0],
                                                           num_partitions))
            self.per_part_size = size[0] // num_partitions
            linear_size = (self.per_part_size, size[1])

        elif axis == 1:
            assert size[1] % num_partitions == 0, (
                "Number of column of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[1],
                                                           num_partitions))
            self.per_part_size = size[1] // num_partitions
            linear_size = (size[0], self.per_part_size)
        else:
            raise ValueError("The value of axis must be 0 or 1, but the value "
                             "given is {}.".format(axis))

        num_rows, num_cols = linear_size

        self.gather_out = gather_out
        self.axis = axis
        if not name:
            name = "fc_by_row_rank_%d" % inner_rank if axis == 0 else "fc_by_col_rank_%d" % inner_rank
        else:
            name = name + "_by_row_rank_%d" % inner_rank if axis == 0 else name + "_by_col_rank_%d" % inner_rank
        self.linear = paddle.nn.Linear(
            num_rows,
            num_cols,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)

        weight = self.linear.weight
        weight.is_distributed = True
        self.weight = self.linear.weight

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[weight.name].is_distributed = True
        main_block.vars[weight.name].is_distributed = True

    def forward(self, x):
        if self.axis == 0:
            assert x.shape[-1] == self.per_part_size, (
                "The width ({}) of the input "
                "x must be equal to the height ({}) of the weight. Maybe you "
                "should split the input x using paddle.split.".format(
                    x.shape[-1], self.per_part_size))

        linear_out = self.linear(x)
        if self.gather_out:
            if self.axis == 0:
                paddle.distributed.all_reduce(linear_out)
            else:
                output = []
                paddle.distributed.all_gather(output, linear_out)
                linear_out = paddle.concat(
                    output, axis=len(linear_out.shape) - 1)
        return linear_out


class ColumnParallelLiner(ParallelLinear):
    def __init__(self,
                 size,
                 num_partitions,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__(
            size,
            axis=1,
            num_partitions=num_partitions,
            gather_out=False,
            param_attr=param_attr,
            bias_attr=bias_attr)


class RowParallelLiner(ParallelLinear):
    def __init__(self,
                 size,
                 num_partitions,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__(
            size,
            axis=0,
            num_partitions=num_partitions,
            gather_out=True,
            param_attr=param_attr,
            bias_attr=bias_attr)
