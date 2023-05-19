# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np
import paddle
from paddle import nn
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from .. import PretrainedModel as PreTrainedModel
from .. import register_base_model
from ..activations import ACT2FN
from .configuration import (
    FUNNEL_PRETRAINED_INIT_CONFIGURATION,
    FUNNEL_PRETRAINED_RESOURCE_FILES_MAP,
    FUNNEL_RESOURCE_FILES_NAMES,
    FunnelConfig,
)

FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge",  # B10-10-10H1024, no decoder
]

__all__ = [
    "FunnelModel",
    "FunnelForSequenceClassification",
    "FunnelForTokenClassification",
    "FunnelForQuestionAnswering",
]

INF = 1e6


def expand(self, *sizes):
    if isinstance(sizes[0], Iterable):
        sizes = sizes[0]
    # handle -1 case
    if len(sizes) > len(self.shape):
        for _ in range(len(sizes) - len(self.shape)):
            self = self.unsqueeze(axis=0)
    x = paddle.expand(self, sizes, name=None)
    return x


def repeat_interleave(x, repeats, dim=None):
    orig_shape = list(x.shape)
    if dim is None:
        dim = 1
        x = paddle.reshape(x, (-1, 1))  # x.reshape(-1,1)
        size = [1] * len(x.shape)
        size[dim] = repeats
        x = paddle.tile(x, size)
        return paddle.reshape(x, (-1))
    else:
        if len(orig_shape) == dim + 1:
            x = x.unsqueeze(-1)
        # x=x.reshape(-1,1)
        size = [1] * len(orig_shape)
        size[-1] = repeats
        x = paddle.tile(x, size)
        orig_shape[dim] = -1
        return paddle.reshape(x, orig_shape)


def gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(
                paddle.reshape(paddle.arange(x.shape[k], dtype=index.dtype), reshape_shape), index_shape
            ).flatten()
            nd_index.append(dim_index)

    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0])
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def split(x, batch_size, dim=0):
    if isinstance(batch_size, int):
        if batch_size > x.shape[dim]:
            return [x]  # do nothing
        return [y for y in paddle.split(x, x.shape[dim] // batch_size, dim)]
    else:
        return [y for y in paddle.split(x, batch_size, dim)]


def normal_(x, m=0, std=1):
    y = paddle.randn(x.shape) * std + m
    paddle.assign(y, x)
    return x


def uniform_(x, a=0, b=1.0):
    temp_value = paddle.uniform(min=a, max=b, shape=x.shape)
    x.set_value(temp_value)
    return x


def constant_(x, val):
    temp_value = paddle.full_like(x, fill_value=val)
    x.set_value(temp_value)
    return x


class FunnelEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layer_norm = LayerNorm(config.d_model, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(inputs_embeds)
        embeddings = self.dropout(embeddings)

        return embeddings


def pad(input, pad, mode="constant", value=0):
    pad2 = []
    for _ in range(len(input.shape) * 2 - len(pad)):
        pad2.append(0)
    if isinstance(pad, tuple):
        pad = list(pad)
    pad2 = pad2 + pad
    return paddle.nn.functional.pad(input, pad2, mode=mode, value=value)


class FunnelAttentionStructure(nn.Layer):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """

    cls_token_type_id: int = 2

    def __init__(self, config):
        super().__init__()
        self.config2 = config
        self.sin_dropout = nn.Dropout(config.hidden_dropout)
        self.cos_dropout = nn.Dropout(config.hidden_dropout)
        # Track where we are at in terms of pooling from the original input, e.g., by how much the sequence length was
        # divided.
        self.pooling_mult = None

    def init_attention_inputs(self, inputs_embeds, attention_mask=None, token_type_ids=None):
        """Returns the attention inputs associated to the inputs of the model."""
        # inputs_embeds has shape batch_size x seq_len x d_model
        # attention_mask and token_type_ids have shape batch_size x seq_len
        self.pooling_mult = 1
        self.seq_len = seq_len = inputs_embeds.shape[1]
        position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype)
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        cls_mask = (
            pad(
                paddle.ones([seq_len - 1, seq_len - 1], dtype=inputs_embeds.dtype), (1, 0, 1, 0)
            )  # nn.functional.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0))
            if self.config2.separate_cls
            else None
        )
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids):
        """Convert `token_type_ids` to `token_type_mat`."""
        # token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
        token_type_mat = token_type_ids.unsqueeze(2) == token_type_ids.unsqueeze(1)
        # Treat <cls> as in the same segment as both A & B
        cls_ids = token_type_ids == self.cls_token_type_id
        # cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
        cls_mat = paddle.logical_or(cls_ids.unsqueeze(2), cls_ids.unsqueeze(1))
        return paddle.logical_or(cls_mat, token_type_mat)

    def get_position_embeds(self, seq_len, dtype):
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shift attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        d_model = self.config2.d_model
        if self.config2.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula.
            # We need to create and return the matrices phi, psi, pi and omega.
            pos_seq = paddle.arange(0, seq_len, 1.0, dtype=dtype)
            freq_seq = paddle.arange(0, d_model // 2, 1.0, dtype=dtype)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            sinusoid = pos_seq.unsqueeze(1) * inv_freq.unsqueeze(0)
            sin_embed = paddle.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed)
            cos_embed = paddle.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed)
            # This is different from the formula on the paper...
            phi = paddle.concat([sin_embed_d, sin_embed_d], axis=-1)
            psi = paddle.concat([cos_embed, sin_embed], axis=-1)
            pi = paddle.concat([cos_embed_d, cos_embed_d], axis=-1)
            omega = paddle.concat([-sin_embed, cos_embed], axis=-1)
            return (phi, pi, psi, omega)
        else:
            # Notations from the paper, appending A.2.1, final formula.
            # We need to create and return all the possible vectors R for all blocks and shifts.
            freq_seq = paddle.arange(0, d_model // 2, 1, dtype=dtype)
            inv_freq = 1 / (10000 ** (freq_seq / (d_model // 2)))
            # Maximum relative positions for the first input
            rel_pos_id = paddle.arange(-seq_len * 2, seq_len * 2, 1, dtype=dtype)
            zero_offset = seq_len * 2
            sinusoid = rel_pos_id.unsqueeze(1) * inv_freq.unsqueeze(0)
            sin_embed = self.sin_dropout(paddle.sin(sinusoid))
            cos_embed = self.cos_dropout(paddle.cos(sinusoid))
            pos_embed = paddle.concat([sin_embed, cos_embed], axis=-1)

            pos = paddle.arange(0, seq_len, dtype=dtype)
            pooled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.config2.num_blocks):
                # For each block with block_index > 0, we need two types position embeddings:
                #   - Attention(pooled-q, unpooled-kv)
                #   - Attention(pooled-q, pooled-kv)
                # For block_index = 0 we only need the second one and leave the first one as None.

                # First type
                if block_index == 0:
                    position_embeds_pooling = None
                else:
                    pooled_pos = self.stride_pool_pos(pos, block_index)

                    # construct rel_pos_id
                    stride = 2 ** (block_index - 1)
                    rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                    rel_pos = rel_pos.unsqueeze(1) + zero_offset
                    rel_pos = expand(rel_pos, (rel_pos.shape[0], d_model))
                    position_embeds_pooling = gather(pos_embed, 0, rel_pos)

                # Second type
                pos = pooled_pos
                stride = 2**block_index
                rel_pos = self.relative_pos(pos, stride)

                rel_pos = rel_pos.unsqueeze(1) + zero_offset
                rel_pos = expand(rel_pos, (rel_pos.shape[0], d_model))
                position_embeds_no_pooling = gather(pos_embed, 0, rel_pos)

                position_embeds_list.append([position_embeds_no_pooling, position_embeds_pooling])
            return position_embeds_list

    def stride_pool_pos(self, pos_id, block_index):
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        if self.config2.separate_cls:
            # Under separate <cls>, we treat the <cls> as the first token in
            # the previous block of the 1st real block. Since the 1st real
            # block always has position 1, the position of the previous block
            # will be at `1 - 2 ** block_index`.
            cls_pos = paddle.to_tensor([-(2**block_index) + 1]).astype(pos_id.dtype)
            pooled_pos_id = pos_id[1:-1] if self.config2.truncate_seq else pos_id[1:]
            return paddle.concat([cls_pos, pooled_pos_id[::2]], axis=0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos, stride, pooled_pos=None, shift=1):
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        if pooled_pos is None:
            pooled_pos = pos

        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * len(pooled_pos)
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]

        return paddle.arange(max_dist, min_dist - 1, -stride, dtype=paddle.int64)

    def stride_pool(self, tensor, axis):
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None
        tensor = tensor.astype("float32")
        # Do the stride pool recursively if axis is a list or a tuple of ints.
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # Do the stride pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # Deal with negative axis
        axis %= tensor.ndim

        if self.config2.separate_cls:
            # tensor = paddle.cat([tensor[cls_slice], tensor], axis=axis)
            if axis == 1:
                tensor = paddle.concat([tensor[:, :1], tensor], axis=axis)
            if axis == 2:
                tensor = paddle.concat([tensor[:, :, :1], tensor], axis=axis)
            if axis == 0:
                tensor = paddle.concat([tensor[:1], tensor], axis=axis)
        if axis == 1:
            return tensor[:, 0:-1:2].astype("bool")
        if axis == 0:
            return tensor[0:-1:2].astype("bool")
        if axis == 2:
            return tensor[:, :, 0:-1:2].astype("bool")

    def pool_tensor(self, tensor, mode="mean", stride=2):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        # Do the pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        if self.config2.separate_cls:
            suffix = tensor[:, :-1] if self.config2.truncate_seq else tensor
            tensor = paddle.concat([tensor[:, :1], suffix], axis=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor.unsqueeze(1).unsqueeze(3)  # [:, None, :, None]
        elif ndim == 3:
            tensor = tensor.unsqueeze(1)  # [:, None, :, :]
        # Stride is applied on the second-to-last dimension.
        stride = (stride, 1)

        if mode == "mean":
            tensor = nn.functional.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = nn.functional.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -nn.functional.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        if ndim == 2:
            return tensor[:, 0, :, 0]
        elif ndim == 3:
            return tensor[:, 0]
        return tensor

    def pre_attention_pooling(self, output, attention_inputs):
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config2.pool_q_only:
            if self.config2.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.config2.pooling_type)

        else:
            self.pooling_mult *= 2
            if self.config2.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            output = self.pool_tensor(output, mode=self.config2.pooling_type)

        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs):
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config2.pool_q_only:
            self.pooling_mult *= 2
            if self.config2.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs


def _relative_shift_gather(positional_attn, context_len, shift):
    batch_size, n_head, seq_len, max_rel_len = positional_attn.shape
    # max_rel_len = 2 * context_len + shift -1 is the numbers of possible relative positions i-j

    # What's next is the same as doing the following gather, which might be clearer code but less efficient.
    # idxs = context_len + paddle.arange(0, context_len).unsqueeze(0) - paddle.arange(0, seq_len).unsqueeze(1)
    # # matrix of context_len + i-j
    # return positional_attn.gather(3, idxs.expand([batch_size, n_head, context_len, context_len]))

    positional_attn = paddle.reshape(positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    positional_attn = positional_attn[:, :, shift:, :]
    positional_attn = paddle.reshape(positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    positional_attn = positional_attn[:, :, :, :context_len]
    return positional_attn


def Parameter(shape_or_tensor, fill_value=None, requires_grad=True):
    if isinstance(shape_or_tensor, paddle.Tensor):
        X = Parameter(shape_or_tensor.shape, 0.0)
        paddle.assign(shape_or_tensor.astype("float32"), X)
    else:
        if isinstance(shape_or_tensor, int):
            shape_or_tensor = [shape_or_tensor]

        X = paddle.create_parameter(
            shape=shape_or_tensor,
            dtype="float32",
            attr=paddle.ParamAttr(name=None, initializer=paddle.nn.initializer.Constant(value=fill_value)),
            is_bias=False,
        )
    if not requires_grad:
        X.stop_gradient = True

    return X


class FunnelRelMultiheadAttention(nn.Layer):
    def __init__(self, config, block_index):
        super().__init__()
        self.config2 = config
        self.block_index = block_index
        d_model, n_head, d_head = config.d_model, config.n_head, config.d_head

        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.q_head = nn.Linear(d_model, n_head * d_head, bias_attr=False)
        self.k_head = nn.Linear(d_model, n_head * d_head)
        self.v_head = nn.Linear(d_model, n_head * d_head)

        self.r_w_bias = Parameter(paddle.zeros([n_head, d_head]))
        self.r_r_bias = Parameter(paddle.zeros([n_head, d_head]))
        self.r_kernel = Parameter(paddle.zeros([d_model, n_head, d_head]))
        self.r_s_bias = Parameter(paddle.zeros([n_head, d_head]))
        self.seg_embed = Parameter(paddle.zeros([2, n_head, d_head]))

        self.post_proj = nn.Linear(n_head * d_head, d_model)
        self.layer_norm = LayerNorm(d_model, epsilon=config.layer_norm_eps)
        self.scale = 1.0 / (d_head**0.5)

    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """Relative attention score for the positional encodings"""
        # q_head has shape batch_size x sea_len x n_head x d_head
        if self.config2.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula (https://arxiv.org/abs/2006.03236)
            # phi and pi have shape seq_len x d_model, psi and omega have shape context_len x d_model
            phi, pi, psi, omega = position_embeds
            # Shape n_head x d_head
            u = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape batch_size x sea_len x n_head x d_model
            q_r_attention = paddle.einsum("binh,dnh->bind", q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi.unsqueeze(1)  # [:, None]
            q_r_attention_2 = q_r_attention * pi.unsqueeze(1)  # [:, None]

            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = paddle.einsum("bind,jd->bnij", q_r_attention_1, psi) + paddle.einsum(
                "bind,jd->bnij", q_r_attention_2, omega
            )
        else:
            shift = 2 if q_head.shape[1] != context_len else 1
            # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
            # Grab the proper positional encoding, shape max_rel_len x d_model
            r = position_embeds[self.block_index][shift - 1]
            # Shape n_head x d_head
            v = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape max_rel_len x n_head x d_model
            r_head = paddle.einsum("td,dnh->tnh", r, w_r)
            # Shape batch_size x n_head x seq_len x max_rel_len
            positional_attn = paddle.einsum("binh,tnh->bnit", q_head + v, r_head)
            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn

    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """Relative attention score for the token_type_ids"""
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = token_type_mat.shape
        # q_head has shape batch_size x seq_len x n_head x d_head
        # Shape n_head x d_head
        r_s_bias = self.r_s_bias * self.scale

        # Shape batch_size x n_head x seq_len x 2
        token_type_bias = paddle.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)

        # Shape batch_size x n_head x seq_len x context_len
        # token_type_mat = token_type_mat[:, None].expand([batch_size, q_head.shape[2], seq_len, context_len])
        token_type_mat = expand(token_type_mat.unsqueeze(1), ([batch_size, q_head.shape[2], seq_len, context_len]))
        # Shapes batch_size x n_head x seq_len
        diff_token_type, same_token_type = split(token_type_bias, 1, dim=-1)
        # Shape batch_size x n_head x seq_len x context_len
        token_type_attn = paddle.where(
            token_type_mat,
            expand(same_token_type, (token_type_mat.shape)),
            expand(diff_token_type, (token_type_mat.shape)),
        )

        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def forward(self, query, key, value, attention_inputs, output_attentions=False):
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = self.config2.n_head, self.config2.d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = paddle.reshape(
            self.q_head(query), (batch_size, seq_len, n_head, d_head)
        )  # self.q_head(query).reshape(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = paddle.reshape(
            self.k_head(key), (batch_size, context_len, n_head, d_head)
        )  # self.k_head(key).reshape(batch_size, context_len, n_head, d_head)
        v_head = paddle.reshape(self.v_head(value), (batch_size, context_len, n_head, d_head))

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len

        content_score = paddle.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)

        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn

        # precision safe in case of mixed precision training
        dtype = attn_score.dtype
        attn_score = attn_score.astype("float32")
        # perform masking
        if attention_mask is not None:
            # attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
            attn_score = attn_score - INF * (1 - attention_mask.unsqueeze(1).unsqueeze(2).astype("float32"))
        # attention probability
        attn_prob = paddle.nn.functional.softmax(attn_score, axis=-1, dtype=dtype)
        attn_prob = self.attention_dropout(attn_prob)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = paddle.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(attn_vec.reshape((batch_size, seq_len, n_head * d_head)))
        attn_out = self.hidden_dropout(attn_out)

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)


class FunnelPositionwiseFFN(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_inner)
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.linear_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = LayerNorm(config.d_model, epsilon=config.layer_norm_eps)

    def forward(self, hidden):
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        return self.layer_norm(hidden + h)


class FunnelLayer(nn.Layer):
    def __init__(self, config, block_index):
        super().__init__()
        self.attention = FunnelRelMultiheadAttention(config, block_index)
        self.ffn = FunnelPositionwiseFFN(config)

    def forward(self, query, key, value, attention_inputs, output_attentions=False):
        attn = self.attention(query, key, value, attention_inputs, output_attentions=output_attentions)
        output = self.ffn(attn[0])
        return (output, attn[1]) if output_attentions else (output,)


class FunnelEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config2 = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.blocks = nn.LayerList(
            [
                nn.LayerList([FunnelLayer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes)
            ]
        )

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.astype(inputs_embeds.dtype)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.shape[1] > (2 if self.config2.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config2.block_repeats[block_index]):
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config2.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    # if layer_index==8 and block_index==0 and repeat_index==0 :
                    #     print(block_index,layer_index,repeat_index,layer,query.mean(), key.mean(), value.mean())
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)

                    hidden = layer_output[0]

                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)
        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


def upsample(x, stride, target_len, separate_cls=True, truncate_seq=False):
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    if stride == 1:
        return x
    if separate_cls:
        cls = x[:, :1]
        x = x[:, 1:]
    output = repeat_interleave(x, repeats=stride, dim=1)
    if separate_cls:
        if truncate_seq:
            output = pad(output, (0, 0, 0, stride - 1, 0, 0))
        output = output[:, : target_len - 1]
        output = paddle.concat([cls, output], axis=1)
    else:
        output = output[:, :target_len]
    return output


class FunnelDecoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config2 = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.layers = nn.LayerList([FunnelLayer(config, 0) for _ in range(config.num_decoder_layers)])

    def forward(
        self,
        final_hidden,
        first_block_hidden,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        upsampled_hidden = upsample(
            final_hidden,
            stride=2 ** (len(self.config2.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
            separate_cls=self.config2.separate_cls,
            truncate_seq=self.config2.truncate_seq,
        )

        hidden = upsampled_hidden + first_block_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        for layer in self.layers:
            layer_output = layer(hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions)
            hidden = layer_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)
        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelDiscriminatorPredictions(nn.Layer):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.config2 = config
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dense_prediction = nn.Linear(config.d_model, 1)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = ACT2FN[self.config2.hidden_act](hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()
        return logits


class FunnelPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    pretrained_init_configuration = FUNNEL_PRETRAINED_INIT_CONFIGURATION
    resource_files_names = FUNNEL_RESOURCE_FILES_NAMES
    pretrained_resource_files_map = FUNNEL_PRETRAINED_RESOURCE_FILES_MAP

    config_class = FunnelConfig
    base_model_prefix = "funnel"

    def _init_weights(self, module):
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            if getattr(module, "weight", None) is not None:
                if self.config.initializer_std is None:
                    fan_out, fan_in = module.weight.shape
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                constant_(module.bias, 0.0)
        elif classname == "FunnelRelMultiheadAttention":
            uniform_(module.r_w_bias, b=self.config.initializer_range)
            uniform_(module.r_r_bias, b=self.config.initializer_range)
            uniform_(module.r_kernel, b=self.config.initializer_range)
            uniform_(module.r_s_bias, b=self.config.initializer_range)
            uniform_(module.seg_embed, b=self.config.initializer_range)
        elif classname == "FunnelEmbeddings":
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            normal_(module.word_embeddings.weight, std=std)
            if module.word_embeddings._padding_idx is not None:
                module.word_embeddings.weight.data[module._padding_idx].zero_()

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)
        _init_weights = True
        if _init_weights:
            # Initialize weights
            self.apply(self._init_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            # self.tie_weights()

    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list of
                heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads
                0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config2.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)


class FunnelClassificationHead(nn.Layer):
    def __init__(self, config, n_labels):
        super().__init__()
        self.linear_hidden = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.linear_out = nn.Linear(config.d_model, n_labels)

    def forward(self, hidden):
        hidden = self.linear_hidden(hidden)
        hidden = paddle.tanh(hidden)
        hidden = self.dropout(hidden)
        return self.linear_out(hidden)


class FunnelBaseModel(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if isinstance(config, PreTrainedModel):
            config = config.config
        if isinstance(config, dict):
            config = FunnelConfig(**config)
        self.config2 = config
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config2.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config2.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


@register_base_model
class FunnelModel(FunnelPreTrainedModel):
    base_model_prefix = "model"

    def __init__(self, config: FunnelConfig):
        super().__init__(config)

        self.config2 = config
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)
        self.decoder = FunnelDecoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config2.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config2.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)
        else:
            token_type_ids = token_type_ids.astype("int64")
        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
            encoder_outputs = self.encoder(
                inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs.last_hidden_state,
            first_block_hidden=encoder_outputs.hidden_states[self.config2.block_sizes[0]],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            idx = 0
            outputs = (decoder_outputs.last_hidden_state,)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs.hidden_states + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs.attentions + decoder_outputs[idx],)
            return outputs
        return BaseModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )


class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.discriminator_predictions = FunnelDiscriminatorPredictions(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (``paddle.Tensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA-style loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:


        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.reshape(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.astype("float32"))
            else:
                loss = loss_fct(logits.reshape(-1, discriminator_sequence_output.shape[1]), labels.astype("float32"))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output
        return FunnelForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class FunnelForMaskedLM(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.lm_head = nn.Linear(config.vocab_size, config.d_model)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        prediction_logits = paddle.matmul(last_hidden_state, self.lm_head.weight, transpose_y=True) + self.lm_head.bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (prediction_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return prediction_logits


class FunnelForSequenceClassification(FunnelPreTrainedModel):
    base_model_class = FunnelModel

    def __init__(self, config, num_classes=2):
        super().__init__(config)
        self.num_classes = num_classes

        self.num_labels = config.num_labels

        self.funnel = FunnelBaseModel(config)
        self.classifier = FunnelClassificationHead(config, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return logits


class FunnelForMultipleChoice(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.funnel = FunnelBaseModel(config)
        self.classifier = FunnelClassificationHead(config, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return reshaped_logits


class FunnelForTokenClassification(FunnelPreTrainedModel):
    def __init__(self, config, num_classes=2):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.funnel = FunnelModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_classes = num_classes

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.reshape(-1, self.num_labels)
                active_labels = paddle.where(
                    active_loss, labels.reshape(-1), paddle.tensor(loss_fct.ignore_index).astype(labels.dtype)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.reshape(-1, self.num_labels), paddle.reshape(labels, -1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return logits


class FunnelForQuestionAnswering(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config2 = config
        self.num_labels = config.num_labels

        self.funnel = FunnelModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`paddle.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`paddle.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """

        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = split(logits, 1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return start_logits, end_logits


def is_tensor(x):
    """
    Tests if ``x`` is a :obj:`paddle.Tensor`,   or
    :obj:`np.ndarray`.
    """

    if isinstance(x, paddle.Tensor):
        return True

    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self):
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class FunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~hf_paddle.FunnelForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``paddle.Tensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA-style objective.
        logits (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss = None
    logits = None
    hidden_states = None
    attentions = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
