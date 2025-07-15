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

import paddle
from paddle import nn
import functools
import math
import operator
from typing import Literal, TypeAlias
import paddle.distributed as dist

from paddle import Tensor
from paddle import _C_ops, base, in_dynamic_mode
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed import collective
from paddle.tensor.manipulation import reshape
from paddle.nn.layer.layers import Layer
_ReduceMode: TypeAlias = Literal['mean', 'sum', 'none']


# TODO: this function is rewrote from paddle.nn.functional.cross_entropy,
# but better to merge into only one.
def parallel_cross_entropy(
    input: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: _ReduceMode = 'mean',
    soft_label: bool = False,
    axis: int = -1,
    use_softmax: bool = True,
    label_smoothing: float = 0.0,
    name: str | None = None,
) -> Tensor:

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in softmax_cross_entropy"
            f"should be 'sum', 'mean' or 'none', but received {reduction}, which is not allowed."
        )
    if ignore_index > 0 and soft_label:
        raise ValueError(
            "When soft_label == True, the value of 'ignore_index' in softmax_cross_entropy"
            f"should be '-100', but received {ignore_index}, which is not allowed."
        )

    input_dims = len(list(input.shape))
    if input_dims == 0:
        raise ValueError('The dimension of input should be larger than zero!')

    label_dims = len(list(label.shape))
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=axis)

    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f'Expected nput_dims - 1 = label_dims or input_dims == label_dims\
             (got nput_dims{input_dims}, label_dims{label_dims})'
        )

    if label_smoothing > 0.0:
        soft_label = True
        # converting the label to one-hot encoding
        # for 1d case, converting label's shape from [N] to [N, C]
        # for 2d case, converting label's shape from [N, d_1, ..., d_k] to [N, d_1, ..., d_k, C]
        if input_dims - 1 == label_dims:
            label = paddle.squeeze(label, axis=axis)
            label = paddle.nn.functional.one_hot(label, input.shape[-1])

        label = paddle.nn.functional.label_smooth(
            label, epsilon=label_smoothing
        )
        label = label.astype(input.dtype)
        label_dims = len(list(label.shape))

    if not soft_label:
        valid_label = (
            paddle.cast(label != ignore_index, dtype=label.dtype) * label
        )
    
    if soft_label == False and is_tensor_sharded(input):
        group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
        ring_id = group.id
        nranks = group.nranks
        global_rank = collective._get_global_env().rank
        rank = group.get_group_rank(global_rank)
        _, out = _C_ops.c_softmax_with_cross_entropy(
            input, label, ignore_index, ring_id, rank, nranks
        )
    else:
        from paddlenlp.utils.log import logger

        logger.warning(
            "Failed to replace CrossEntropyLoss with ParallelCrossEntropyLoss. Please ensure: \n"
            "1. soft_label=False is set for parallel computation (current value: {}) \n"
            "2. Input tensor is properly sharded (current sharding status: {}) \n".format(
                soft_label, 
                input.placements,
            )
        )

        _, out = _C_ops.cross_entropy_with_softmax(
            input, label, soft_label, use_softmax, True, ignore_index, axis
        )

    if weight is not None:
        # trans weight from class to sample, shape:N or [N,H,W] for 1d and 2d cases.
        if soft_label:
            # chajchaj:
            # weight's shape is C, where C is class num.
            # for 1d case: label's shape is [N,C], weight_gather's shape is N.
            # for 2d case: label's shape is [N,H,W,C], weight_gather's shape is [N,H,W].
            weight_gather = paddle.matmul(
                x=paddle.cast(label, weight.dtype),
                y=weight,
                transpose_x=False,
                transpose_y=True,
            )
            out_shape = list(out.shape)
            weight_gather_reshape = reshape(weight_gather, shape=out_shape)
            out = paddle.cast(out, weight_gather_reshape.dtype)

            out = _C_ops.multiply(out, weight_gather_reshape)
        else:
            if input.shape[axis] != weight.shape[-1]:
                raise ValueError(
                    f"input's class_dimension({input.shape[axis]}) must equal to "
                    f"weight's class_dimension({weight.shape[-1]}) "
                    "when weight is provided"
                )

            ignore_weight_mask = paddle.cast(
                (label != ignore_index), out.dtype
            )
            if (
                ignore_weight_mask.ndim > 1
                and ignore_weight_mask.shape[axis] == 1
            ):
                # TODO: Temporarily use squeeze instead of squeeze_
                ignore_weight_mask = paddle.squeeze(
                    ignore_weight_mask, axis
                )
            if axis != -1 and axis != valid_label.ndim - 1:
                temp_perm = (
                    list(range(axis % valid_label.ndim))
                    + list(
                        range(
                            (axis % valid_label.ndim + 1), valid_label.ndim
                        )
                    )
                    + [axis % valid_label.ndim]
                )
                weight_gather = _C_ops.gather_nd(
                    weight, valid_label.transpose(temp_perm)
                )
            else:
                weight_gather = _C_ops.gather_nd(weight, valid_label)
            weight_gather = _C_ops.multiply(
                weight_gather, ignore_weight_mask
            )
            input_shape = list(label.shape)
            weight_gather_reshape = reshape(
                weight_gather, shape=input_shape
            )
            out = paddle.cast(out, weight_gather_reshape.dtype)
            out = _C_ops.multiply(out, weight_gather_reshape)

    if reduction == "sum":
        #   because of base_softmax_with_cross_entropy op's inner logic,
        #   in the out tensor of this op, the loss of sample with class_index==ignore_index is 0
        #   so, reduce_sum all directly is ok
        return _C_ops.sum(out, [], None, False)
    elif reduction == "mean":
        # 1. if weight==none,
        #     numerator: reduce_sum all loss directly is ok causeof base_softmax_with_cross_entropy's inner logic
        #     denominator: count sample num with class_index!=ignore_index
        # 2. else
        #     numerator: loss's weighted sum
        #     denominator: cal the sum of weight where the sample's class_index!=ignore_index
        if ignore_index >= 0:  # ignore label
            out_sum = _C_ops.sum(out, [], None, False)
            # for each label[i],set 1 or 0, according to ignore_index
            # mask[i]=0, if label[i]==ignore_index
            # mask[i]=1, otherwise
            mask = label != ignore_index
            if weight is None:
                mask = paddle.cast(mask, dtype=out_sum.dtype)
                count = _C_ops.sum(mask, [], None, False)
                ret = out_sum / (count + (count == 0.0).astype(count.dtype))
            else:
                mask = paddle.cast(mask, weight_gather_reshape.dtype)
                weight_ignored = _C_ops.multiply(
                    mask, weight_gather_reshape
                )
                weight_sum = _C_ops.sum(weight_ignored, [], None, False)
                ret = out_sum / (
                    weight_sum
                    + (weight_sum == 0.0).astype(weight_sum.dtype)
                )
            return ret
        elif weight is not None:
            out_sum = _C_ops.sum(out, [], None, False)
            total_weight = _C_ops.sum(
                weight_gather_reshape, [], None, False
            )
            return out_sum / (
                total_weight
                + (total_weight == 0.0).astype(total_weight.dtype)
            )
        else:
            return _C_ops.mean_all(out)

    else:
        if input_dims - 1 == label_dims:
            out = paddle.squeeze(out, axis=axis)
        return out


# TODO: placement[1] may not be mp axis.
def is_tensor_sharded(tensor):
    if not tensor.is_dist():
        return False

    placement = tensor.placements
    return placement[1].is_shard()


def replace_cross_entropy():
    paddle.nn.functional.cross_entropy = parallel_cross_entropy