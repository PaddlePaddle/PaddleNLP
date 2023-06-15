# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..initializer import normal_, zeros_
from ..utils import deprecate, is_ppxformers_available, logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CrossAttention(nn.Layer):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.cross_attention_norm = cross_attention_norm

        self.scale = dim_head**-0.5

        self.heads = heads
        self.head_dim = dim_head
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, epsilon=1e-5)
        else:
            self.group_norm = None

        if cross_attention_norm:
            self.norm_cross = nn.LayerNorm(cross_attention_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias_attr=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias_attr=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias_attr=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.LayerList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        if processor is None:
            processor = CrossAttnProcessor()
            # processor = AttnProcessor2_5() if is_ppxformers_available() else CrossAttnProcessor()
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[str] = None
    ):
        is_lora = hasattr(self, "processor") and isinstance(
            self.processor, (LoRACrossAttnProcessor, LoRAXFormersCrossAttnProcessor)
        )

        if use_memory_efficient_attention_xformers:
            if self.added_kv_proj_dim is not None:
                # TODO(Anton, Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                raise NotImplementedError(
                    "Memory efficient attention with `xformers` is currently not supported when"
                    " `self.added_kv_proj_dim` is defined."
                )
            elif not is_ppxformers_available():
                raise NotImplementedError(
                    "requires the scaled_dot_product_attention but your PaddlePaddle donot have this. Checkout the instructions on the installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment."
                )
            else:
                try:
                    _ = F.scaled_dot_product_attention_(
                        paddle.randn((1, 1, 2, 40), dtype=paddle.float16),
                        paddle.randn((1, 1, 2, 40), dtype=paddle.float16),
                        paddle.randn((1, 1, 2, 40), dtype=paddle.float16),
                        attention_op=attention_op,
                    )
                except Exception as e:
                    raise e

            if is_lora:
                if self.head_dim > 128 and attention_op == "flash":
                    attention_op = "cutlass"
                processor = LoRAXFormersCrossAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                    attention_op=attention_op,
                )
                # we must cust dtype
                processor.to(dtype=self.dtype)
                processor.load_dict(self.processor.state_dict())
            else:
                if self.head_dim > 128 and attention_op == "flash":
                    attention_op = "cutlass"  # CrossAttnProcessor()
                processor = XFormersCrossAttnProcessor(attention_op=attention_op)
        else:
            if is_lora:
                processor = LoRACrossAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                )
                processor.to(dtype=self.dtype)
                processor.load_dict(self.processor.state_dict())
            else:
                processor = CrossAttnProcessor()

        self.set_processor(processor)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = CrossAttnAddedKVProcessor()
        else:
            processor = CrossAttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor"):
        # if current processor is in `self._sub_layers` and if passed `processor` is not, we need to
        # pop `processor` from `self._sub_layers`
        if hasattr(self, "processor") and isinstance(self.processor, nn.Layer) and not isinstance(processor, nn.Layer):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._sub_layers.pop("processor")

        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `CrossAttention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor, transpose=True):
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        tensor = tensor.reshape([0, 0, tensor.shape[2] * tensor.shape[3]])
        return tensor

    def head_to_batch_dim(self, tensor, transpose=True):
        tensor = tensor.reshape([0, 0, self.heads, self.head_dim])
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        if self.upcast_softmax or self.upcast_attention:
            dtype = query.dtype

        if self.upcast_attention:
            query = query.cast(paddle.float32)
            key = key.cast(paddle.float32)

        attention_scores = paddle.matmul(query, key, transpose_y=True) * self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.cast(paddle.float32)

        attention_probs = F.softmax(attention_scores, axis=-1)

        if self.upcast_softmax or self.upcast_attention:
            attention_probs = attention_probs.cast(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None, reshape4d=True):
        if batch_size is None:
            deprecate(
                "batch_size=None",
                "0.0.15",
                message=(
                    "Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect"
                    " attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to"
                    " `prepare_attention_mask` when preparing the attention_mask."
                ),
            )
            batch_size = 1

        num_heads = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0, data_format="NCL")

        if attention_mask.shape[0] < batch_size * num_heads:
            # TODO repeat_interleave may not support float16
            attention_mask = attention_mask.repeat_interleave(num_heads, axis=0)

        if reshape4d:
            # batch_size, num_heads, 1, seqlen
            return attention_mask.reshape([batch_size, num_heads, -1, attention_mask.shape[-1]])
        else:
            return attention_mask


class CrossAttnProcessor:
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class LoRALinearLayer(nn.Layer):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.down = nn.Linear(in_features, rank, bias_attr=False)
        self.up = nn.Linear(rank, out_features, bias_attr=False)

        normal_(self.down.weight, std=1 / rank)
        zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.cast(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.cast(orig_dtype)


class LoRACrossAttnProcessor(nn.Layer):
    def __init__(self, hidden_size, cross_attention_dim=None, rank=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CrossAttnAddedKVProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.reshape([hidden_states.shape[0], hidden_states.shape[1], -1]).transpose(
            [0, 2, 1]
        )
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.transpose([0, 2, 1])

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        key = paddle.concat([encoder_hidden_states_key_proj, key], axis=2)
        value = paddle.concat([encoder_hidden_states_value_proj, value], axis=2)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class XFormersCrossAttnProcessor:
    def __init__(self, attention_op: Optional[str] = None):
        assert attention_op in [None, "cutlass", "flash"]
        self.attention_op = attention_op

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # if transpose = False, query's shape will be [batch_size, seq_len, num_head, head_dim]
        query = attn.head_to_batch_dim(query, transpose=False)
        key = attn.head_to_batch_dim(key, transpose=False)
        value = attn.head_to_batch_dim(value, transpose=False)

        hidden_states = F.scaled_dot_product_attention_(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=attn.scale,
            dropout_p=0.0,
            training=attn.training,
            attention_op=self.attention_op,
        )

        # hidden_states = hidden_states.cast(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LoRAXFormersCrossAttnProcessor(nn.Layer):
    def __init__(self, hidden_size, cross_attention_dim, rank=4, attention_op: Optional[str] = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.attention_op = attention_op

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        query = attn.head_to_batch_dim(query, transpose=False)
        key = attn.head_to_batch_dim(key, transpose=False)
        value = attn.head_to_batch_dim(value, transpose=False)

        hidden_states = F.scaled_dot_product_attention_(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=attn.scale,
            dropout_p=0.0,
            training=attn.training,
            attention_op=self.attention_op,
        )

        hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SlicedAttnProcessor:
    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, reshape4d=False)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        batch_size_attention = query.shape[0]
        hidden_states = paddle.zeros((batch_size_attention, sequence_length, attn.head_dim), dtype=query.dtype)

        for i in range(hidden_states.shape[0] // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = paddle.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape back to [bs, num_heads, seqlen, head_dim]
        hidden_states = hidden_states.reshape([-1, attn.heads, sequence_length, attn.head_dim])

        hidden_states = attn.batch_to_head_dim(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SlicedAttnAddedKVProcessor:
    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(self, attn: "CrossAttention", hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.reshape([hidden_states.shape[0], hidden_states.shape[1], -1]).transpose(
            [0, 2, 1]
        )
        encoder_hidden_states = encoder_hidden_states.transpose([0, 2, 1])

        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, reshape4d=False)

        hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        key = paddle.concat([encoder_hidden_states_key_proj, key], axis=2)
        value = paddle.concat([encoder_hidden_states_value_proj, value], axis=2)

        # flatten
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        batch_size_attention = query.shape[0]
        hidden_states = paddle.zeros((batch_size_attention, sequence_length, attn.head_dim), dtype=query.dtype)

        for i in range(hidden_states.shape[0] // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = paddle.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape back to [bs, num_heads, seqlen, head_dim]
        hidden_states = hidden_states.reshape([-1, attn.heads, sequence_length, attn.head_dim])

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


AttnProcessor2_5 = XFormersCrossAttnProcessor

AttnProcessor = Union[
    CrossAttnProcessor,
    XFormersCrossAttnProcessor,
    SlicedAttnProcessor,
    CrossAttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    LoRACrossAttnProcessor,
    LoRAXFormersCrossAttnProcessor,
]
