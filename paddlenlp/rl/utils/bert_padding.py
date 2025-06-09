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
# reference from Dao-AILAB flash-attn
# https://github.com/Dao-AILab/flash-attention/blob/74b0761ff7efc7b90d4e5aeb529c1b2a09a7458c/flash_attn/bert_padding.py#L38
import paddle
import paddle.nn.functional as F

try:
    from einops import rearrange, repeat
except ImportError:
    raise ImportError("`einops` is not installed. Please run `pip install einops`")


import operator
from functools import reduce
from typing import Optional


class IndexFirstAxis(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2, "input must be at least 2-dimensional"
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = reduce(operator.mul, other_shape, 1)
        return paddle.take_along_axis(
            arr=rearrange(input, "b ... -> b (...)"), axis=0, indices=repeat(indices, "z -> z d", d=second_dim)
        ).reshape([-1, *other_shape])

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensor()
        assert grad_output.ndim >= 2, "grad_output must be at least 2-dimensional"
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = paddle.zeros(shape=[ctx.first_axis_dim, tuple(grad_output.shape)[1]], dtype=grad_output.dtype)

        grad_input.put_along_axis_(
            axis=0,
            indices=repeat(indices, "z -> z d", d=tuple(grad_output.shape)[1]),
            values=grad_output,
        )
        return grad_input.reshape([ctx.first_axis_dim, *other_shape]), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1, "indices must be 1-dimensional"
        assert values.ndim >= 2, "values must be at least 2-dimensional"
        output = paddle.zeros(shape=[first_axis_dim, *tuple(values.shape)[1:]], dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensor()
        grad_values = grad_output[indices]
        return grad_values, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = paddle.sum(attention_mask, axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = paddle.max(seqlens_in_batch).item()
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0), [1, 0])

    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def sequence_parallel_pad_inputs(
    input_ids_rmpad: paddle.Tensor,
    position_ids_rmpad: Optional[paddle.Tensor] = None,
    sp_size: int = 1,
    pad_value: int = 0,
):
    """
    Pad input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.
    Note both input_ids_rmpad and position_ids_rmpad will be padded.
    The is the utility of pre-forward for sequence parallelism
    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): sequence parallelism size
    Returns:
        paddle.Tensor: padded input_ids
        paddle.Tensor: padded position_ids
        int: pad size
    """
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.shape[0] == 1
        assert input_ids_rmpad.shape[1] == position_ids_rmpad.shape[1]
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = paddle.nn.functional.pad(input_ids_rmpad, (0, 0, 0, pad_size), value=pad_value)
        if position_ids_rmpad is not None:
            pad_pos_ids = paddle.arange(pad_size).unsqueeze(0)
            position_ids_rmpad = paddle.concat((position_ids_rmpad, pad_pos_ids), axis=-1)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def prepare_flashmask_inputs(
    input_ids: paddle.Tensor, position_ids: paddle.Tensor, pad_token_id=0, sequence_parallel=False, sp_size=8
):
    attn_mask = (input_ids != pad_token_id).cast("int32")
    # input ids rmpad
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attn_mask)  # input_ids_rmpad (total_nnz, ...)
    input_ids_rmpad = input_ids_rmpad.transpose([1, 0])

    # position ids rmpad
    position_ids_rmpad = index_first_axis(
        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose([1, 0])

    input_ids_rmpad_rolled = paddle.roll(input_ids_rmpad, shifts=-1, axis=1)  # (1, total_nnz)

    # startend_row_indices
    cum_sum = attn_mask.sum(1).cumsum(0, dtype="int32").unsqueeze(1)
    valid_cum_sum = (attn_mask * cum_sum).flatten()
    attn_mask_startend_row_indices_rmpad = paddle.index_select(valid_cum_sum, indices).unsqueeze(0)

    pad_size = 0
    # For SP
    if sequence_parallel and sp_size > 1:
        input_ids_rmpad, position_ids_rmpad, pad_size = sequence_parallel_pad_inputs(
            input_ids_rmpad, position_ids_rmpad, sp_size=sp_size, pad_value=pad_token_id
        )
        input_ids_rmpad_rolled = sequence_parallel_pad_inputs(input_ids_rmpad_rolled, sp_size=sp_size, pad_value=-100)[
            0
        ]
        attn_mask_startend_row_indices_rmpad = sequence_parallel_pad_inputs(
            attn_mask_startend_row_indices_rmpad, sp_size=sp_size
        )[0]

    return {
        "input_ids": input_ids_rmpad.contiguous(),
        "position_ids": position_ids_rmpad.contiguous(),
        "input_ids_rmpad_rolled": input_ids_rmpad_rolled.contiguous(),
        "attn_mask_startend_row_indices": attn_mask_startend_row_indices_rmpad.contiguous(),
        "pad_size": pad_size,
        "indices": indices,
        "raw_input_shape": input_ids.shape,
    }
