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

import math

import paddle
import paddle.nn.functional as F

import paddlenlp
from paddlenlp.transformers.llama.modeling import get_triangle_upper_mask

ssa_group_size_ratio = 1 / 4


def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
    assert qkv.shape == [bsz, num_heads, q_len, head_dim], "qkv shape does not match expected shape"
    # Calculate the shift amount for rolling
    shift_amount = -group_size // 2
    # Roll the qkv tensor along the sequence length axis
    qkv[:, num_heads // 2 :] = qkv[:, num_heads // 2 :].roll(shift_amount, axis=2)

    # Reshape the tensor to the desired shape
    qkv = qkv.reshape([bsz * (q_len // group_size), group_size, num_heads, head_dim])
    return qkv


def ssa_scaled_dot_product_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    sequence_parallel=False,
    reshard_layer=None,
    **kwargs
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    if config.context_parallel_degree > 1:
        raise ValueError("Context parallel requires `use_flash_attention=True`")
    # [ bz, seqlen, nhead, head_dim] -> [bs, nhead, seq_len, head_dim]
    query_states = paddle.transpose(query_states, [0, 2, 1, 3])
    # merge with the next tranpose
    key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    value_states = paddle.transpose(value_states, [0, 2, 1, 3])
    assert ssa_group_size_ratio is not None, "ssa_group_size_ratio must provide"

    # Calculate the group size based on the sequence length and the group size ratio
    group_size = q_len if int(q_len * ssa_group_size_ratio) == 0 else int(q_len * ssa_group_size_ratio)
    assert q_len % group_size == 0, f"q_len {q_len} must be divisible by group size {group_size}."

    num_group = q_len // group_size

    # Apply shifting to the query, key, and value states
    query_states = shift(query_states, bsz, q_len, group_size, num_heads, head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, num_heads, head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, num_heads, head_dim)
    query_states = paddle.transpose(query_states, [0, 2, 1, 3])
    key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    value_states = paddle.transpose(value_states, [0, 2, 1, 3])
    # matmul and device by sqrt(head_dim)
    attn_weights = paddle.matmul(query_states / math.sqrt(head_dim), key_states.transpose([0, 1, 3, 2]))

    # then add alibi bias
    if alibi is not None:
        alibi = alibi.reshape([bsz, num_heads, 1, -1])
        attn_weights = attn_weights + alibi
    if paddle.in_dynamic_mode() and attn_weights.shape != [bsz * num_group, num_heads, group_size, group_size]:
        raise ValueError(
            f"Attention weights should be of shape {(bsz * num_group, num_heads, group_size, group_size)}, but is"
            f" {attn_weights.shape}"
        )

    # In sep mode, the attenion mask should be created in the runtime.
    if reshard_layer is not None:
        attention_mask = None

    if attention_mask is None:
        attention_mask = get_triangle_upper_mask(attn_weights)
    attention_mask = paddle.tile(
        paddle.cast(attention_mask[:, :, :group_size, :group_size], dtype="float32"), [num_group, 1, 1, 1]
    )

    if attention_mask.shape != [bsz * num_group, 1, group_size, group_size]:
        attention_mask = attention_mask[: bsz * num_group, :, :, :]

    attn_weights = attn_weights + attention_mask
    if not paddle.in_dynamic_mode():
        attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
    else:
        with paddle.amp.auto_cast(False):
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)

    attn_output = paddle.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose([0, 2, 1, 3])

    # shift back
    attn_output = attn_output.reshape([bsz, q_len, num_heads, head_dim])
    attn_output[:, :, num_heads // 2 :] = attn_output[:, :, num_heads // 2 :].roll(group_size // 2, axis=1)

    if reshard_layer is not None:
        attn_output = reshard_layer(
            attn_output,
            split_axis=1,
            concat_axis=2,
        )
        q_len = q_len // config.sep_parallel_degree
        num_heads = num_heads * config.sep_parallel_degree

    if sequence_parallel:
        attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
    else:
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
    return (attn_output, attn_weights) if output_attentions else attn_output


def set_group_size(group_size_ratio):
    global ssa_group_size_ratio
    ssa_group_size_ratio = group_size_ratio


def replace_llama_attn():
    paddlenlp.transformers.llama.modeling.scaled_dot_product_attention = ssa_scaled_dot_product_attention
