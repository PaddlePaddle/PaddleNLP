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
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode, in_dygraph_mode, default_main_program
from paddle.nn.initializer import Constant
from paddle import _C_ops
import paddle.incubate.nn.functional as incubate_f

try:
    from paddle.distributed.fleet import fleet
except Exception as e:
    import warnings
    warnings.warn("paddle.distributed is not contains in you paddle!")

__all__ = [
    'guard',
    'ParallelEmbedding',
    'ColumnParallelLiner',
    'RowParallelLiner',
    'ParallelFusedFeedForward',
    'ParallelFusedMultiHeadAttention',
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
    Parallel Embedding.

    Args:
        num_embeddings (int):
            The size of embedding dictionary which dictates the maximum value of the input id.
        embedding_dim (int):
            The dimensions of each embedding vector.
        rank (int):
            The rank of the current part, which determines the start index of the vocab.
        world_size (int):
            The number of trainers.
        weight_attr (Tensor, optional):
            Specify the weight parameter property, including the initialization method.
            Defaults to None which means the default weight parameter property will be used.
        name (str, optional):
            Normally there is no need for user to set this property.
            Defaults to None.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 rank,
                 world_size,
                 weight_attr=None,
                 name=None):
        super(ParallelEmbedding, self).__init__()
        self.rank = rank
        self.world_size = world_size
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

    def forward(self, x):
        """
        Args:
            x (Tensor):
                A Tensor contains the id information.
                Its data type should be int32 or int64, and the value of the input id should be in [0, weight.shape[0]] .

        Returns:
            Tensor: Returns the embedding Tensor mapped by x.
        """
        if self.is_mp:
            output_parallel = paddle.distributed.collective._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                name=self._name)
            output = paddle.distributed.collective._mp_allreduce(
                output_parallel,
                group=None,
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


class ColumnParallelLiner(nn.Layer):
    """
    Parallel Linear, axis=1.

    Args:
        size (int):
            The size of embedding vector.
        num_partitions (int, optional):
            The number of parts within a model parallel group. Defaults to 1.
        gather_out (bool, optional):
            Whether to gather the output tensor. Defaults to True.
        param_attr (Tensor, optional):
            Specify the parameter property, including the initialization method.
            Defaults to None which means the default parameter property will be used.
        bias_attr (Tensor, optional):
            Specify the bias property.
            Defaults to None which means the default parameter property will be used.
        name (str, optional):
            Normally there is no need for user to set this property.
            Defaults to None.

    """

    def __init__(self,
                 size,
                 num_partitions=1,
                 gather_out=True,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__()

        if paddle.in_dynamic_mode():
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
        else:
            assert fleet._role_maker, ("To use paddle.distributed.split, "
                                       "you must call fleet.init() firstly.")
            rank = fleet.worker_index()
            nranks = fleet.worker_num()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.gather_out = gather_out

        assert size[1] % num_partitions == 0, (
            "Number of column of the weight for linear ({}) must be"
            " divisible by num_partitions ({})".format(size[1], num_partitions))
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
            bias_attr=bias_attr,
            name=name)

        weight = self.linear.weight
        weight.is_distributed = True
        # alias for weight tensor
        self.weight = self.linear.weight

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[weight.name].is_distributed = True
        main_block.vars[weight.name].is_distributed = True
        # set is_distributed for splited bias
        # if a linear layer is splited by col, the bias would also be split into each rank as its weight
        if self.linear._bias_attr != False:
            startup_block.vars[self.linear.bias.name].is_distributed = True
            main_block.vars[self.linear.bias.name].is_distributed = True
            self.bias = self.linear.bias

    def forward(self, x):
        """
        Args:
            x (Tensor):
                The input tensor. Its data type can be int or float.

        Returns:
            Tensor: Returns the embedding Tensor mapped by x.
        """
        group = None
        x = paddle.distributed.collective._c_identity(x, group=group)
        output_parallel = self.linear(x)
        if self.gather_out is False:
            return output_parallel

        return paddle.distributed.collective._c_concat(
            output_parallel, group=group)


class RowParallelLiner(nn.Layer):
    """
    Parallel Linear, axis=0.

    Args:
        size (int):
            The size of embedding vector.
        num_partitions (int, optional):
            The number of parts within a model parallel group. Defaults to 1.
        input_is_parallel (bool, optional):
            Whether the input is parallel. Defaults to `False`.
        param_attr (Tensor, optional):
            Specify the parameter property, including the initialization method.
            Defaults to None which means the default parameter property will be used.
        bias_attr (Tensor, optional):
            Specify the bias property.
            Defaults to None which means the default parameter property will be used.
        name (str, optional):
            Normally there is no need for user to set this property.
            Defaults to None.

    """

    def __init__(self,
                 size,
                 num_partitions=1,
                 input_is_parallel=False,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__()

        if paddle.in_dynamic_mode():
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
        else:
            assert fleet._role_maker, ("To use paddle.distributed.split, "
                                       "you must call fleet.init() firstly.")
            rank = fleet.worker_index()
            nranks = fleet.worker_num()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.input_is_parallel = input_is_parallel

        assert size[0] % num_partitions == 0, (
            "Number of rows of the weight for linear ({}) must be"
            " divisible by num_partitions ({})".format(size[0], num_partitions))
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

        weight = self.linear.weight
        weight.is_distributed = True
        # alias for weight tensor
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[weight.name].is_distributed = True
        main_block.vars[weight.name].is_distributed = True
        # set is_distributed for splited bias
        # if a linear layer is splited by row, each rank would hold a complete bias

        if bias_attr is not False:
            self.bias = self.create_parameter(
                shape=[num_cols],
                attr=bias_attr,
                dtype=self._dtype,
                is_bias=True)
        else:
            self.bias = None

    def forward(self, x):
        """
        Args:
            x (Tensor):
                The input tensor. Its data type can be int or float.

        Returns:
            Tensor: Returns the embedding Tensor mapped by x.
        """
        group = None
        if self.input_is_parallel:
            assert x.shape[-1] == self.per_part_size, (
                "The width ({}) of the input "
                "x must be equal to the height ({}) of the weight. Maybe you "
                "should split the input x using paddle.split.".format(
                    x.shape[-1], self.per_part_size))
        else:
            # split last dim
            x = paddle.distributed.collective._c_split(x, group=group)
        output_parallel = self.linear(x)
        output = paddle.distributed.collective._mp_allreduce(
            output_parallel,
            group=group,
            use_calc_stream=True,
            use_model_parallel=True)
        output = output + self.bias if self.bias is not None else output
        return output


def _verify_dropout_rate(dropout_rate):
    if not isinstance(dropout_rate, (float, int)):
        raise TypeError("dropout_rate argument should be a number")
    if dropout_rate < 0 or dropout_rate > 1:
        raise ValueError("dropout_rate argument should between 0 and 1")


def fused_feedforward(x,
                      linear1_weight,
                      linear2_weight,
                      linear1_bias=None,
                      linear2_bias=None,
                      ln1_scale=None,
                      ln1_bias=None,
                      ln2_scale=None,
                      ln2_bias=None,
                      dropout1_rate=0.5,
                      dropout2_rate=0.5,
                      activation="relu",
                      ln1_epsilon=1e-5,
                      ln2_epsilon=1e-5,
                      pre_layer_norm=False,
                      training=True,
                      mode='upscale_in_train',
                      ring_id=-1,
                      name=None):
    r"""
    This is a fusion operator to compute feed forward layer in transformer model architecture.
    This operator only supports running on GPU. The function of the operator is consistent with
    the following pseudo code:

    .. code-block:: python

        residual = src;
        if pre_layer_norm:
            src = layer_norm(src)
        src = linear(dropout(activation(dropout(linear(src)))))
        if not pre_layer_norm:
            src = layer_norm(out)

    Args:
        x (Tensor): the input tensor could be 3-D tensor, the input data type could be float16, float32 or float64, the shape is`[batch\_size, sequence\_length, d_model]`.
        linear1_weight (Tensor): The weight of first linear, the data type is same as `x`, the shape is `[d\_model, dim\_feedforward]`.
        linear2_weight (Tensor): The weight of second linear, the data type is same as `x`, the shape is `[dim\_feedforward, d\_model]`.
        linear1_bias (Tensor, optional): The bias of first linear, the data type is same as `x`, the shape is `[dim_feedforward]`. Default None.
        linear2_bias (Tensor, optional): The bias of second linear, the data type is same as `x`, the shape is `[d_model]`. Default None.
        ln1_scale (Tensor, optional): the weight of first layer_norm, the data type is float32 or float64, the shape is same as `x`. Default None.
        ln1_bias (Tensor, optional): The bias of first layer_norm, the data type is float32 or float64, the shape is `[d\_model]`. Default None.
        ln2_scale (Tensor, optional): The weight of second layer_norm, the data type is float32 or float64, the shape is same as `x`. Default None.
        ln2_bias (Tensor, optional): The bias of second layer_norm, the data type is float32 or float64, the shape is `[d\_model]`. Default None.
        dropout1_rate (float, optional): The first dropout probability of setting units to zero. Default 0.5.
        dropout2_rate (float, optional): The second dropout probability of setting units to zero. Default 0.5.
        activation (str, optional): The activation. Default "relu".
        ln1_epsilon (float, optional): Small float of first layer_norm added to denominator to avoid dividing by zero. Default is 1e-5.
        ln2_epsilon (float, optional): Small float of second layer_norm added to denominator to avoid dividing by zero. Default is 1e-5.
        pre_layer_norm (bool, optional): add layer_norm in the pre-processing stage or post-processing state.
        training (bool, optional): A flag indicating whether it is in train phrase or not. Default True.
        mode (str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train(default), upscale the output at training time

                                  - train: out = input * mask / ( 1.0 - p )
                                  - inference: out = input

                               2. downscale_in_infer, downscale the output at inference

                                  - train: out = input * mask
                                  - inference: out = input * (1.0 - p)
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The output Tensor, the data type and shape is same as `x`.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            import numpy as np
            x_data = np.random.random((1, 8, 8)).astype("float32")
            linear1_weight_data = np.random.random((8, 8)).astype("float32")
            linear2_weight_data = np.random.random((8, 8)).astype("float32")
            x = paddle.to_tensor(x_data)
            linear1_weight = paddle.to_tensor(linear1_weight_data)
            linear2_weight = paddle.to_tensor(linear2_weight_data)
            out = paddle.incubate.nn.functional.fused_feedforward(x, linear1_weight, linear2_weight)
            print(out.numpy().shape)
            # (1, 8, 8)
    """
    _verify_dropout_rate(dropout1_rate)
    _verify_dropout_rate(dropout2_rate)

    seed = None
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'")
    mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

    if in_dygraph_mode():
        if default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        out, _, _, _, _, _, _, _, _, _, _ = _C_ops.fused_feedforward(
            x, None, None, linear1_weight, linear1_bias, linear2_weight,
            linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias,
            'pre_layer_norm', pre_layer_norm, 'ln1_epsilon', ln1_epsilon,
            'ln2_epsilon', ln2_epsilon, 'act_method', activation,
            'dropout1_rate', dropout1_rate, 'dropout2_rate', dropout2_rate,
            "dropout1_is_test", not training, "dropout2_is_test", not training,
            "dropout1_fix_seed", seed is not None, "dropout2_fix_seed",
            seed is not None, "dropout1_seed", seed
            if seed is not None else 0, "dropout2_seed", seed
            if seed is not None else 0, 'dropout1_implementation', mode,
            'dropout2_implementation', mode, 'ring_id', ring_id)
        return out

    helper = LayerHelper("fused_feedforward")
    dtype = x.dtype
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'fused_feedforward')
    check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'],
                'fused_feedforward')

    out = helper.create_variable_for_type_inference(x.dtype)
    dropout1_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    dropout2_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    ln1_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    linear1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout2_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)

    if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
        seed = helper.main_program.random_seed

    helper.append_op(
        type='fused_feedforward',
        inputs={
            'X': x,
            'Linear1Weight': linear1_weight,
            'Linear1Bias': linear1_bias,
            'Linear2Weight': linear2_weight,
            'Linear2Bias': linear2_bias,
            'Ln1Scale': ln1_scale,
            'Ln1Bias': ln1_bias,
            'Ln2Scale': ln2_scale,
            'Ln2Bias': ln2_bias,
        },
        outputs={
            'Out': out,
            'Dropout1Mask': dropout1_mask,
            'Dropout2Mask': dropout2_mask,
            'Ln1Mean': ln1_mean,
            'Ln1Variance': ln1_variance,
            'Ln2Mean': ln2_mean,
            'Ln2Variance': ln2_variance,
            'Linear1Out': linear1_out,
            'Ln1Out': ln1_out,
            'Dropout1Out': dropout1_out,
            'Dropout2Out': dropout2_out,
        },
        attrs={
            'dropout1_rate': dropout1_rate,
            'dropout2_rate': dropout2_rate,
            'act_method': activation,
            'pre_layer_norm': pre_layer_norm,
            'ln1_epsilon': ln1_epsilon,
            'ln2_epsilon': ln2_epsilon,
            'dropout1_is_test': not training,
            'dropout2_is_test': not training,
            'dropout1_fix_seed': seed is not None,
            'dropout2_fix_seed': seed is not None,
            'dropout1_seed': seed if seed is not None else 0,
            'dropout2_seed': seed if seed is not None else 0,
            'dropout1_implementation': mode,
            'dropout2_implementation': mode,
            'ring_id': ring_id,
        })
    return out


def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    # NOTE: use current_block and find_var_recursive to support while_loop
    startup_block = paddle.static.default_startup_program().current_block()
    main_block = paddle.static.default_main_program().current_block()
    startup_block._find_var_recursive(var.name).is_distributed = True
    main_block._find_var_recursive(var.name).is_distributed = True


class ParallelFusedFeedForward(Layer):
    """
    Parameters:
        d_model (int): The expected feature size in the input and output.
        dim_feedforward (int): The hidden layer size.
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess. Default 0.1
        epsilon (float, optional): he small value added to the variance to prevent
            division by zero. Default: 1e-05.
        activation (str, optional): The activation function. Default relu.
        act_dropout_rate (float, optional): The dropout probability after activition.
            If None, use the value of `dropout_rate`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into, preprocessing or postprocessing. Default False
        weight_attr (ParamAttr, optional): The attribute for the learnable weight of this layer.
            The default value is None and the weight will be initialized to zero. For detailed
            information, please refer to paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias of thi layer.
            If it is set to False, no bias will be added to the output. If it is set to None or one
            kind of ParamAttr, a bias parameter will be created according to ParamAttr. For detailed
            information, please refer to paddle.ParamAttr. The default value is None and the bias
            will be initialized to zero.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedFeedForward

            fused_feedforward_layer = FusedFeedForward(8, 8)
            x = paddle.rand((1, 8, 8))
            out = fused_feedforward_layer(x)
            print(out.numpy().shape)
            # (1, 8, 8)
    """

    def __init__(self,
                 d_model,
                 dim_feedforward,
                 dropout_rate=0.1,
                 epsilon=1e-05,
                 activation="relu",
                 act_dropout_rate=None,
                 normalize_before=False,
                 linear1_weight_attr=None,
                 linear1_bias_attr=None,
                 linear2_weight_attr=None,
                 linear2_bias_attr=None,
                 ln1_scale_attr=None,
                 ln1_bias_attr=None,
                 ln2_scale_attr=None,
                 ln2_bias_attr=None,
                 nranks=1,
                 ring_id=-1,
                 name=None):

        super(ParallelFusedFeedForward, self).__init__()
        assert d_model > 0, (
            "Expected d_model to be greater than 0, but recieved {}".format(
                d_model))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, but recieved {}".
            format(dim_feedforward))

        self._dtype = self._helper.get_default_dtype()
        self._d_model = d_model

        assert dim_feedforward % nranks == 0
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward
        self._dropout_rate = dropout_rate
        self._act_dropout_rate = dropout_rate if act_dropout_rate is None else act_dropout_rate
        self._act_method = activation
        self._normalize_before = normalize_before
        self._epsilon = epsilon
        self._ring_id = ring_id

        self._linear1_weight = self.create_parameter(
            shape=[d_model, dim_feedforward],
            attr=linear1_weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self._linear1_bias = self.create_parameter(
            shape=[dim_feedforward],
            attr=linear1_bias_attr,
            dtype=self._dtype,
            is_bias=True)

        self._linear2_weight = self.create_parameter(
            shape=[dim_feedforward, d_model],
            attr=linear2_weight_attr,
            dtype=self._dtype,
            is_bias=False)

        self._linear2_bias = self.create_parameter(
            shape=[d_model], attr=linear2_bias_attr, dtype=self._dtype, is_bias=True)

        if nranks > 1:
            assert ring_id != -1
            # column parallel
            _set_var_distributed(self._linear1_weight)
            _set_var_distributed(self._linear1_bias)
            _set_var_distributed(self._linear2_weight)

        if normalize_before:
            self._ln1_scale = self.create_parameter(
                shape=[d_model],
                attr=ln1_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0))
            self._ln1_bias = self.create_parameter(
                shape=[d_model], attr=ln1_bias_attr, is_bias=True)
            self._ln2_scale = None
            self._ln2_bias = None
        else:
            self._ln1_bias = None
            self._ln2_bias = None
            self._ln2_scale = self.create_parameter(
                shape=[d_model],
                attr=ln2_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0))
            self._ln2_bias = self.create_parameter(
                shape=[d_model], attr=ln2_bias_attr, is_bias=True)

        self.name = name

    def forward(self, src, cache=None):
        out = fused_feedforward(
            src,
            self._linear1_weight,
            self._linear2_weight,
            self._linear1_bias,
            self._linear2_bias,
            self._ln1_scale,
            self._ln1_bias,
            self._ln2_scale,
            self._ln2_bias,
            dropout1_rate=self._act_dropout_rate,
            dropout2_rate=self._dropout_rate,
            activation=self._act_method,
            ln1_epsilon=self._epsilon,
            ln2_epsilon=self._epsilon,
            pre_layer_norm=self._normalize_before,
            training=self.training,
            ring_id=self._ring_id,
            name=self.name)
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'd_model={}, dim_feedforward={}, dropout_rate={}, epsilon={}, activation={}, act_dropout_rate={}, normalize_before={}, dtype={}{}'.format(
            self._d_model, self._dim_feedforward, self._dropout_rate,
            self._epsilon, self._act_method, self._act_dropout_rate,
            self._normalize_before, self._dtype, name_str)


class ParallelFusedMultiHeadAttention(Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        attn_dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout in attention.
            0 for no dropout. Default 0.5.
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        normalize_before (bool, optional): Indicate  whether it is pre_layer_norm
            (True) or post_layer_norm architecture (False). Default False.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Now, only False is supported. Default False.
        weight_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr`.
        bias_attr (ParamAttr|bool, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            # input: [batch_size, sequence_length, embed_dim]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.incubate.nn.FusedMultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout_rate=0.5,
                 attn_dropout_rate=0.5,
                 kdim=None,
                 vdim=None,
                 normalize_before=False,
                 need_weights=False,
                 qkv_weight_attr=None,
                 qkv_bias_attr=None,
                 linear_weight_attr=None,
                 linear_bias_attr=None,
                 pre_ln_scale_attr=None,
                 pre_ln_bias_attr=None,
                 ln_scale_attr=None,
                 ln_bias_attr=None,
                 epsilon=1e-5,
                 nranks=1,
                 ring_id=-1,
                 name=None):
        super(ParallelFusedMultiHeadAttention, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but recieved {}".format(embed_dim))
        assert num_heads > 0, ("Expected nhead to be greater than 0, "
                               "but recieved {}".format(num_heads))

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.need_weights = need_weights
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        assert need_weights == False, "Only support need_weight is False now."

        # tensor model parallel
        assert num_heads % nranks == 0
        num_heads = num_heads // nranks

        self.qkv_weight = self.create_parameter(
            shape=[3, num_heads, self.head_dim, embed_dim],
            attr=qkv_weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.qkv_bias = self.create_parameter(
            shape=[3, num_heads, self.head_dim],
            attr=qkv_bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.linear_weight = self.create_parameter(
            shape=[num_heads * self.head_dim, embed_dim],
            attr=linear_weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.linear_bias = self.create_parameter(
            shape=[embed_dim],
            attr=linear_bias_attr,
            dtype=self._dtype,
            is_bias=True)

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
            # column parallel
            _set_var_distributed(self.qkv_weight)
            _set_var_distributed(self.qkv_bias)
            # row parallel
            _set_var_distributed(self.linear_weight)

        if normalize_before:
            self.pre_ln_scale = self.create_parameter(
                attr=pre_ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0))
            self.pre_ln_bias = self.create_parameter(
                attr=pre_ln_bias_attr, shape=[embed_dim], is_bias=True)
            self.ln_scale = None
            self.ln_bias = None
        else:
            self.pre_ln_scale = None
            self.pre_ln_bias = None
            self.ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0))
            self.ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True)

        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.name = name

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        out = incubate_f.fused_multi_head_attention(
            x=query,
            qkv_weight=self.qkv_weight,
            linear_weight=self.linear_weight,
            pre_layer_norm=self.normalize_before,
            pre_ln_scale=self.pre_ln_scale,
            pre_ln_bias=self.pre_ln_bias,
            ln_scale=self.ln_scale,
            ln_bias=self.ln_bias,
            pre_ln_epsilon=self._epsilon,
            qkv_bias=self.qkv_bias,
            linear_bias=self.linear_bias,
            cache_kv=cache,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            ln_epsilon=self._epsilon,
            training=self.training,
            ring_id=self._ring_id,
            name=self.name)
        return out
