import paddle
from paddle import nn
import functools
import math
import operator
from typing import Literal, TypeAlias
import paddle.distributed as dist

from paddle import _C_ops, base, in_dynamic_mode
from paddle.static.nn.control_flow import Assert
from paddle.distributed.fleet.base import topology as tp
from paddle.distributed import collective
from paddle import Tensor

from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.framework import in_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle.common_ops_import import Variable
from paddle.tensor.manipulation import reshape
from paddle.nn.layer.layers import Layer
_ReduceMode: TypeAlias = Literal['mean', 'sum', 'none']


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

    if in_dynamic_mode():
        if not soft_label:
            valid_label = (
                paddle.cast(label != ignore_index, dtype=label.dtype) * label
            )
        group = tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
        ring_id = group.id
        nranks = group.nranks
        global_rank = collective._get_global_env().rank
        rank = group.get_group_rank(global_rank)
        _, out = _C_ops.c_softmax_with_cross_entropy(
            input, label, ignore_index, ring_id, rank, nranks
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

    else:
        check_variable_and_dtype(
            input,
            'input',
            ['uint16', 'float16', 'float32', 'float64'],
            'softmax_cross_entropy',
        )
        check_variable_and_dtype(
            label,
            'label',
            ['uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'],
            'softmax_cross_entropy',
        )
        if in_pir_mode():
            softmax, out = _C_ops.cross_entropy_with_softmax(
                input, label, soft_label, use_softmax, True, ignore_index, axis
            )
        else:
            attrs = {
                'soft_label': soft_label,
                'ignore_index': ignore_index,
                'numeric_stable_mode': True,
                'axis': axis,
                'use_softmax': use_softmax,
            }
            helper = LayerHelper('softmax_with_cross_entropy', **locals())
            softmax = helper.create_variable_for_type_inference(
                dtype=input.dtype
            )
            out = helper.create_variable_for_type_inference(dtype=input.dtype)

            outputs = {'Softmax': softmax, 'Loss': out}
            helper.append_op(
                type='softmax_with_cross_entropy',
                inputs={'Logits': input, 'Label': label},
                outputs=outputs,
                attrs=attrs,
            )

        if weight is not None:
            check_variable_and_dtype(
                weight,
                'weight',
                ['float32', 'float64'],
                'softmax_cross_entropy',
            )
            weight_name = name if reduction == 'none' else None
            if soft_label:
                # chajchaj:
                # trans weight from class to sample, shape:N or [N,H,W] for 1d and 2d cases.
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
            else:
                if input.shape[axis] != weight.shape[-1]:
                    raise ValueError(
                        f"input's class_dimension({input.shape[axis]}) must equal to "
                        f"weight's class_dimension({weight.shape[-1]}) "
                        "when weight is provided"
                    )

                valid_label = paddle.multiply(
                    paddle.cast(label != ignore_index, dtype=label.dtype), label
                )
                ignore_weight_mask = paddle.cast(
                    (label != ignore_index), input.dtype
                )
                if (
                    ignore_weight_mask.ndim > 1
                    and ignore_weight_mask.shape[axis] == 1
                ):
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
                    weight_gather = paddle.gather_nd(
                        weight, paddle.transpose(valid_label, temp_perm)
                    )
                else:
                    weight_gather = paddle.gather_nd(weight, valid_label)
                weight_gather = paddle.multiply(
                    weight_gather, ignore_weight_mask
                )

                input_shape = list(label.shape)
                weight_gather_reshape = reshape(
                    weight_gather, shape=input_shape
                )
            out = paddle.multiply(out, weight_gather_reshape, name=weight_name)

        if reduction == "sum":
            return paddle.sum(out, name=name)
        elif reduction == "mean":
            if ignore_index >= 0:
                out_sum = paddle.sum(out, name=name)
                # for each label[i],set 1 or 0, according to ignore_index
                # mask[i]=0, if label[i]==ignore_index
                # mask[i]=1, otherwise
                mask = label != ignore_index
                if weight is None:
                    mask = paddle.cast(mask, dtype=out_sum.dtype)
                    count = paddle.sum(mask, name=name)
                    ret = out_sum / (count + paddle.equal(count, 0.0))
                else:
                    mask = paddle.cast(mask, weight_gather_reshape.dtype)
                    weight_ignored = paddle.multiply(
                        mask, weight_gather_reshape
                    )
                    weight_sum = paddle.sum(weight_ignored, name=name)
                    ret = out_sum / (weight_sum + paddle.equal(weight_sum, 0.0))
                return ret
            elif weight is not None:
                out_sum = paddle.sum(out, name=name)
                total_weight = paddle.sum(weight_gather_reshape)
                return out_sum / (
                    total_weight + paddle.equal(total_weight, 0.0)
                )
            else:
                return paddle.mean(out, name=name)

        else:
            if input_dims - 1 == label_dims:
                out = paddle.squeeze(out, axis=axis)

            return out


class ParallelCrossEntropyLoss(Layer):
    
    weight: Tensor | None
    ignore_index: int
    reduction: _ReduceMode
    soft_label: bool
    axis: int
    use_softmax: bool
    label_smoothing: float
    name: str | None

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: _ReduceMode = 'mean',
        soft_label: bool = False,
        axis: int = -1,
        use_softmax: bool = True,
        label_smoothing: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.soft_label = soft_label
        self.axis = axis
        self.use_softmax = use_softmax
        self.label_smoothing = label_smoothing
        self.name = name

    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        hcg = dist.fleet.get_hybrid_communicate_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
        if self.soft_label == False and is_tensor_sharded(input) and tensor_parallel_degree > 1:
            cross_entropy_func = parallel_cross_entropy
        else:
            from paddlenlp.utils.log import logger

            logger.warning(
                "Failed to replace CrossEntropyLoss with ParallelCrossEntropyLoss. Please ensure: \n"
                "1. soft_label=False is set for parallel computation (current value: {}) \n"
                "2. Input tensor is properly sharded (current sharding status: {}) \n"
                "3. Not using Model Parallelisma (current mp degree: {}). ".format(
                    self.soft_label, 
                    input_placement,
                    tensor_parallel_degree,
                    hcg.get_model_parallel_world_size()
                )
            )
            cross_entropy_func = paddle.nn.functional.cross_entropy
        ret = cross_entropy_func(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            soft_label=self.soft_label,
            axis=self.axis,
            use_softmax=self.use_softmax,
            label_smoothing=self.label_smoothing,
            name=self.name,
        )

        return ret


def is_tensor_sharded(tensor):
    if not tensor.is_dist():
        return False

    placement = tensor.placements
    return placement[1].is_shard()


def replace_cross_entropy():
    paddle.nn.CrossEntropyLoss = ParallelCrossEntropyLoss