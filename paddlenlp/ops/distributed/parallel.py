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
    'ColumnParallelLiner',
    'RowParallelLiner',
]


def guard(device):
    def decorator(Layer):
        class WrapperClass(Layer):
            def __init__(self, *args, **kw):
                with paddle.static.device_guard(device):
                    print("Init {} on {}".format(Layer.__name__, device))
                    super().__init__(*args, **kw)

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
                 topo,
                 weight_attr=None,
                 name=None):
        super(ParallelEmbedding, self).__init__()
        self.rank = topo.mp_info.rank
        self.world_size = topo.mp_info.size
        self.num_embeddings = num_embeddings
        self.is_mp = (self.world_size > 1)

        assert num_embeddings % self.world_size == 0, \
            "The length of the vocabulary must be divisible by the parallelism degree of MP"

        per_part_size = num_embeddings // self.world_size

        self.vocab_start_index = self.rank * per_part_size
        self._dtype = self._helper.get_default_dtype()
        self._size = [per_part_size, embedding_dim]
        self._weight_attr = weight_attr
        self._name = name

        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False)
        self.weight.is_distributed = True

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[self.weight.name].is_distributed = True
        main_block.vars[self.weight.name].is_distributed = True
        if self.is_mp:
            self.model_group = \
                fleet.get_hybrid_communicate_group().get_model_parallel_group()

    def forward(self, x):
        if self.is_mp:
            output_parallel = paddle.distributed.collective._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                name=self._name)
            output = paddle.distributed.collective._mp_allreduce(
                output_parallel,
                group=self.model_group,  # None is ok in static
                use_calc_stream=True,
                use_model_parallel=True)
        else:
            output = paddle.nn.functional.embedding(
                x,
                weight=self.weight,
                padding_idx=None,
                sparse=False,
                name=self._name)
        return output


def _get_rank():
    if in_dygraph_mode():
        rank = paddle.distributed.get_rank()
        nranks = paddle.distributed.get_world_size()
    else:
        assert fleet._role_maker, ("To use paddle.distributed.split, "
                                   "you must call fleet.init() firstly.")
        rank = fleet.worker_index()
        nranks = fleet.worker_num()
    return rank, nranks


def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True
    startup_block = paddle.static.default_startup_program().global_block()
    main_block = paddle.static.default_main_program().global_block()
    startup_block.vars[var.name].is_distributed = True
    main_block.vars[var.name].is_distributed = True


class ColumnParallelLiner(nn.Layer):
    """
    Parallel Linear, axis=1
    """

    def __init__(self,
                 size,
                 num_partitions=1,
                 gather_out=True,
                 param_attr=None,
                 bias_attr=None,
                 skip_bias_add=False,
                 name=None):
        super(ColumnParallelLiner, self).__init__()

        self.gather_out = gather_out
        self.skip_bias_add = skip_bias_add

        rank, nranks = _get_rank()
        # rank within a model parallel group
        inner_rank = rank % num_partitions

        assert size[1] % num_partitions == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by num_partitions ({})".format(size[1],
                                                       num_partitions))
        self.per_part_size = size[1] // num_partitions
        linear_size = (size[0], self.per_part_size)

        num_rows, num_cols = linear_size

        if not name:
            name = "fc_by_col_rank_%d" % inner_rank
        else:
            name = name + "_by_col_rank_%d" % inner_rank

        self.linear = paddle.nn.Linear(
            num_rows,
            num_cols,
            weight_attr=param_attr,
            bias_attr=False,
            name=name)

        if bias_attr is not False:
            self.bias = self.create_parameter(
                shape=[num_cols],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True
            )
        else:
            self.bias = None

        # alias for weight tensor
        self.weight = self.linear.weight

        _set_var_distributed(self.weight)
        # if a linear layer is splited by col, the bias would also be split into each rank as its weight
        _set_var_distributed(self.bias)

        self.model_group = \
            fleet.get_hybrid_communicate_group().get_model_parallel_group()

    def forward(self, x):
        x = paddle.distributed.collective._c_identity(
            x, group=self.model_group)  # None is ok in static

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = paddle.nn.functional.linear(x, self.weight, bias)

        if self.gather_out:
            # must be model_group when hybrid because rank
            # and nranks not given
            output = paddle.distributed.collective._concat(
                output_parallel, group=self.model_group)
        else:
            output = output_parallel

        if self.skip_bias_add:
            return output, self.bias
        return output


class RowParallelLiner(nn.Layer):
    """
    Parallel Linear, axis=0
    """

    def __init__(self,
                 size,
                 num_partitions=1,
                 input_is_parallel=False,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super(RowParallelLiner, self).__init__()

        # TODO(wangxi): use model_group
        rank, nranks = _get_rank()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.input_is_parallel = input_is_parallel

        assert size[0] % num_partitions == 0, (
            "Number of rows of the weight for linear ({}) must be"
            " divisible by num_partitions ({})".format(size[0],
                                                       num_partitions))
        self.per_part_size = size[0] // num_partitions
        linear_size = (self.per_part_size, size[1])

        num_rows, num_cols = linear_size

        if not name:
            name = "fc_by_row_rank_%d" % inner_rank
        else:
            name = name + "_by_row_rank_%d" % inner_rank
        self.linear = paddle.nn.Linear(
            num_rows,
            num_cols,
            weight_attr=param_attr,
            # NOTE(wangxi): row split, bias need add after allreduce
            bias_attr=False,
            name=name)

        # alias for weight tensor
        self.weight = self.linear.weight
        _set_var_distributed(self.weight)

        if bias_attr is not False:
            self.bias = self.create_parameter(
                shape=[num_cols],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True
            )
        else:
            self.bias = None

        self.model_group = \
            fleet.get_hybrid_communicate_group().get_model_parallel_group()

    def forward(self, x):
        if self.input_is_parallel:
            assert x.shape[-1] == self.per_part_size, (
                "The width ({}) of the input "
                "x must be equal to the height ({}) of the weight. Maybe you "
                "should split the input x using paddle.split.".format(
                    x.shape[-1], self.per_part_size))
        else:
            # split last dim
            # must be model_group when hybrid because rank
            # and nranks not given
            x = paddle.distributed.collective._c_split(x, group=self.model_group)
        output_parallel = self.linear(x)
        output = paddle.distributed.collective._mp_allreduce(
            output_parallel,
            group=self.model_group,  # None is ok in static
            use_calc_stream=True,
            use_model_parallel=True)
        output = output + self.bias if self.bias is not None else output
        return output
