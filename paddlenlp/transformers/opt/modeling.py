# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import collections
from functools import partial
from typing import Any, Dict, List

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.distributed import fleet
from paddle.nn import Layer
from paddle.nn.layer.transformer import _convert_param_attr_to_list

from paddlenlp.transformers.conversion_utils import StateDictNameMapping
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model
from paddlenlp.utils.log import logger

from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from .configuration import (
    OPT_PRETRAINED_INIT_CONFIGURATION,
    OPT_PRETRAINED_RESOURCE_FILES_MAP,
    OPTConfig,
)

__all__ = ["OPTModel", "OPTPretrainedModel", "OPTForCausalLM", "OPTForConditionalGeneration"]


def finfo(dtype):
    if dtype == "float32":
        return np.finfo(np.float32)
    if dtype == "float16":
        return np.finfo(np.float16)
    if dtype == "float64":
        return np.finfo(np.float64)


def _make_causal_mask(input_ids_shape, past_key_values_length, dtype):
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape

    mask = paddle.full((target_length, target_length), float(finfo(paddle.get_default_dtype()).min))

    mask_cond = paddle.arange(mask.shape[-1])
    mask_cond = mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1])
    mask = paddle.where(mask_cond, paddle.full(mask_cond.shape, 0), mask)

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.zeros([target_length, past_key_values_length], dtype=mask.dtype), mask], axis=-1)

    expanded_mask = mask.unsqueeze(0).expand([batch_size, 1, target_length, target_length + past_key_values_length])
    return expanded_mask


def _expand_mask(mask, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(paddle.cast(mask[:, None, None, :], "bool"))
    expanded_mask = paddle.cast(expanded_mask, dtype=paddle.float32)

    expanded_mask = expanded_mask.expand([batch_size, 1, tgt_length, src_length])
    expanded_mask = expanded_mask * float(finfo(paddle.get_default_dtype()).min)
    return expanded_mask


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(
        self,
        config: OPTConfig,
        need_weights=False,
    ):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        # get the `num_heads`
        assert self.num_heads % config.mp_degree == 0
        self.num_heads = self.num_heads // config.mp_degree

        self.dropout = config.attention_probs_dropout_prob
        self.need_weights = need_weights
        self.fuse_attention_qkv = config.fuse_attention_qkv

        assert (
            self.head_dim * self.num_heads * config.mp_degree == config.hidden_size
        ), "hidden_size must be divisible by num_heads"

        if config.mp_degree > 1:
            if self.fuse_attention_qkv:
                self.qkv_proj = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size * 3,
                    has_bias=True,
                    input_is_parallel=True,
                )
            else:
                self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                )
                self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                )
                self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                )

            self.out_proj = fleet.meta_parallel.RowParallelLinear(
                config.hidden_size, config.hidden_size, input_is_parallel=True, has_bias=True
            )
        else:
            if self.fuse_attention_qkv:
                self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
            else:
                self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
                self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
                self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def _fuse_prepare_qkv(self, query, use_cache=False, cache=None):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, self.num_heads, 3 * self.head_dim])
        mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)

        assert not isinstance(cache, self.StaticCache), "cache currently does not support the StaticCache type"

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = paddle.concat([cache.k, k], axis=2)
            v = paddle.concat([cache.v, v], axis=2)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v, cache) if use_cache else (q, k, v, None)

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        """
        q = self.q_proj(query)
        q = paddle.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = paddle.concat([cache.k, k], axis=2)
            v = paddle.concat([cache.v, v], axis=2)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v, None) if use_cache is False else (q, k, v, cache)

    def compute_kv(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.

        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = paddle.full(shape=[key.shape[0], self.num_heads, 0, self.head_dim], dtype=key.dtype, fill_value=0)
            v = paddle.full(shape=[key.shape[0], self.num_heads, 0, self.head_dim], dtype=key.dtype, fill_value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key, value, attn_mask=None, use_cache=False, cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value

        if self.fuse_attention_qkv:
            q, k, v, cache = self._fuse_prepare_qkv(query, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache, cache)

        # scale dot product attention
        product = paddle.matmul(x=q * (self.head_dim**-0.5), y=k, transpose_y=True)

        if attn_mask is not None:
            product = product + attn_mask

        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(weights, self.dropout, training=self.training, mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoderLayer(nn.Layer):
    """
    The transformer decoder layer.

    It contains multiheadattention and some linear layers.
    """

    def __init__(self, config):

        d_model = config.hidden_size
        dim_feedforward = config.intermediate_size
        dropout = config.hidden_dropout_prob
        activation = config.hidden_act
        attn_dropout = config.attention_probs_dropout_prob
        act_dropout = config.hidden_dropout_prob
        normalize_before = getattr(config, "normalize_before", True)

        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range))
        bias_attr = None

        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(config, need_weights=True)
        if config.mp_degree > 1:
            self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
                d_model,
                dim_feedforward,
                gather_output=False,
                has_bias=False,
            )
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])

        if config.mp_degree > 1:
            self.linear2 = fleet.meta_parallel.RowParallelLinear(
                dim_feedforward,
                d_model,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.linear2 = nn.Linear(dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")

        if activation == "gelu":
            self.activation = nn.GELU(approximate=True)
        else:
            self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None, output_attentions=False):
        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)

        # self.self_attn(...) --> hidden_states, weights, (cache)
        if use_cache is False:
            tgt, attn_weights = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, attn_weights, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.dropout2(self.linear2(self.activation(self.linear1(tgt))))
        tgt = residual + tgt

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if not (output_attentions or use_cache):
            return tgt

        temp_list = [tgt, attn_weights if output_attentions else None, incremental_cache if use_cache else None]

        return tuple(v for v in temp_list if v is not None)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        return incremental_cache


class TransformerDecoder(Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(self, config: OPTConfig, decoder_layers: List[Layer]):
        super(TransformerDecoder, self).__init__()

        if config.word_embed_proj_dim != config.hidden_size:
            if config.mp_degree > 1:
                self.project_out = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    config.word_embed_proj_dim,
                    gather_output=True,
                    has_bias=False,
                )
            else:
                self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias_attr=False)
        else:
            self.project_out = None

        self.num_layers = config.num_hidden_layers
        self.layers = decoder_layers

        if config.normalize_before:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None

        self.checkpoints = []

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        use_cache: bool = False,
        cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = [] if use_cache else None
        self.checkpoints = []
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, mod in enumerate(self.layers):
            outputs = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                use_cache=use_cache,
                cache=cache[i] if cache is not None else cache,
                output_attentions=output_attentions,
            )

            # outputs = hidden_states if both use_cache and output_attentions are False
            # Otherwise, outputs = (hidden_states, attention if output_attentions, cache if use_cache)
            output = outputs[0] if (use_cache or output_attentions) else outputs

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
            if use_cache:
                new_caches.append(outputs[-1])
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output,)
            self.checkpoints.append(output.name)

        if self.final_layer_norm:
            output = self.final_layer_norm(output)

        if self.project_out:
            output = self.project_out(output)

        if not return_dict:
            temp_list = [output, new_caches, all_hidden_states, all_self_attentions]

            if not (use_cache or output_attentions or output_hidden_states):
                return output

            return tuple(v for v in temp_list if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output,
            past_key_values=new_caches,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )

    def gen_cache(self, memory, do_zip=False):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
        """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


class OPTLearnedPositionEmbedding(nn.Embedding):
    """this module learns postional embeddings up to a fixed maximum size"""

    def __init__(self, num_embeddings: int, embedding_dim: int, initializer_range: float):
        """OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        and adjust num_embeddings appropriately. Other models don't have this hack.

        Args:
            num_embeddings (int): the number of embedding size
            embedding_dim (int): the dim of embedding
        """
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask, past_key_values_length: int = 0):
        """get the position embedding with attention mask

        Args:
            attention_mask: (paddle.Tensor): # create positions depending on attention_mask
            past_key_values_length (int, optional): the past key value which will . Defaults to 0.

        Returns:
            paddle.Tensor: the position embedding
        """
        # create positions depending on attention_mask
        if attention_mask.dtype not in [paddle.bool, paddle.int64]:
            attention_mask = attention_mask == 1.0

        position_ids = paddle.cumsum(paddle.cast(attention_mask, "int64"), axis=-1) - 1

        # cut positions if `past_key_values_length` is > 0
        position_ids = position_ids[:, past_key_values_length:]
        return nn.Embedding.forward(self, position_ids + self.offset)


class OPTEmbeddings(Layer):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(self, config: OPTConfig):
        super(OPTEmbeddings, self).__init__()
        if config.mp_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.word_embed_proj_dim,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range)
                ),
            )
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size,
                config.word_embed_proj_dim,
                # padding_idx=config.pad_token_id,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range)
                ),
            )

        if config.word_embed_proj_dim != config.hidden_size:
            if config.mp_degree > 1:
                self.project_in = fleet.meta_parallel.ColumnParallelLinear(
                    config.word_embed_proj_dim,
                    config.hidden_size,
                    gather_output=True,
                    has_bias=False,
                )
            else:
                self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias_attr=False)
        else:
            self.project_in = None

        self.position_embeddings = OPTLearnedPositionEmbedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            initializer_range=config.initializer_range,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, attention_mask=None, input_embeddings=None, past_key_values_length=None):
        if input_ids is not None:
            input_embeddings = self.word_embeddings(input_ids)

        if self.project_in:
            input_embeddings = self.project_in(input_embeddings)

        position_embeddings = self.position_embeddings(attention_mask, past_key_values_length)

        embeddings = input_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class OPTPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained OPT models. It provides OPT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    config_class = OPTConfig
    base_model_prefix = "opt"

    pretrained_init_configuration = OPT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = OPT_PRETRAINED_RESOURCE_FILES_MAP

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: OPTConfig, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )
        actions = {
            "word_embeddings.weight": partial(fn, is_column=False),
        }
        for layer_index in range(config.num_hidden_layers):
            actions.update(
                {
                    # Column Linear
                    f"decoder.layers.{layer_index}.self_attn.q_proj.weight": partial(fn, is_column=True),
                    f"decoder.layers.{layer_index}.self_attn.k_proj.weight": partial(fn, is_column=True),
                    f"decoder.layers.{layer_index}.self_attn.v_proj.weight": partial(fn, is_column=True),
                    f"decoder.layers.{layer_index}.linear1.weight": partial(fn, is_column=True),
                    # Row Linear
                    f"decoder.layers.{layer_index}.linear2.weight": partial(fn, is_column=False),
                    f"decoder.layers.{layer_index}.self_attn.out_proj.weight": partial(fn, is_column=False),
                }
            )

        if config.word_embed_proj_dim != config.hidden_size:
            actions.update(
                {
                    "decoder.project_out.weight": partial(fn, is_column=True),
                    "decoder.project_in.weight": partial(fn, is_column=True),
                }
            )

        if cls.__name__ != "OPTModel":
            for key in list(actions.keys()):
                actions["opt." + key] = actions.pop(key)

        return actions

    @classmethod
    def _get_name_mappings(cls, config: OPTConfig) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["decoder.embed_tokens.weight", "embeddings.word_embeddings.weight"],
            ["decoder.embed_positions.weight", "embeddings.position_embeddings.weight"],
            ["decoder.final_layer_norm.weight", "decoder.final_layer_norm.weight"],
            ["decoder.final_layer_norm.bias", "decoder.final_layer_norm.bias"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"decoder.layers.{layer_index}.self_attn.k_proj.weight",
                    f"decoder.layers.{layer_index}.self_attn.k_proj.weight",
                    "transpose",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn.k_proj.bias",
                    f"decoder.layers.{layer_index}.self_attn.k_proj.bias",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn.v_proj.weight",
                    f"decoder.layers.{layer_index}.self_attn.v_proj.weight",
                    "transpose",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn.v_proj.bias",
                    f"decoder.layers.{layer_index}.self_attn.v_proj.bias",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn.q_proj.weight",
                    f"decoder.layers.{layer_index}.self_attn.q_proj.weight",
                    "transpose",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn.q_proj.bias",
                    f"decoder.layers.{layer_index}.self_attn.q_proj.bias",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn.out_proj.weight",
                    f"decoder.layers.{layer_index}.self_attn.out_proj.weight",
                    "transpose",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn.out_proj.bias",
                    f"decoder.layers.{layer_index}.self_attn.out_proj.bias",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn_layer_norm.weight",
                    f"decoder.layers.{layer_index}.norm1.weight",
                ],
                [
                    f"decoder.layers.{layer_index}.self_attn_layer_norm.bias",
                    f"decoder.layers.{layer_index}.norm1.bias",
                ],
                [
                    f"decoder.layers.{layer_index}.fc1.weight",
                    f"decoder.layers.{layer_index}.linear1.weight",
                    "transpose",
                ],
                [f"decoder.layers.{layer_index}.fc1.bias", f"decoder.layers.{layer_index}.linear1.bias"],
                [
                    f"decoder.layers.{layer_index}.fc2.weight",
                    f"decoder.layers.{layer_index}.linear2.weight",
                    "transpose",
                ],
                [f"decoder.layers.{layer_index}.fc2.bias", f"decoder.layers.{layer_index}.linear2.bias"],
                [
                    f"decoder.layers.{layer_index}.final_layer_norm.weight",
                    f"decoder.layers.{layer_index}.norm2.weight",
                ],
                [f"decoder.layers.{layer_index}.final_layer_norm.bias", f"decoder.layers.{layer_index}.norm2.bias"],
            ]
            model_mappings.extend(layer_mappings)

        # base-model prefix "OPTModel"
        if cls.__name__ != "OPTModel":
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "opt." + mapping[1]

        # downstream mappings
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.opt.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )


@register_base_model
class OPTModel(OPTPretrainedModel):
    r"""
    The bare OPT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`OPTConfig`):
            An instance of OPTConfig used to construct OPTModel.
    """

    def __init__(self, config: OPTConfig):
        super(OPTModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embeddings = OPTEmbeddings(config)

        config.fuse_attention_qkv = False
        decoder_layers = nn.LayerList()
        for i in range(config.num_hidden_layers):
            decoder_layers.append(TransformerDecoderLayer(config))
        self.decoder = TransformerDecoder(config, decoder_layers)
        self.checkpoints = []

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length, dtype=attention_mask.dtype
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, tgt_length=input_shape[-1])
            if input_shape[-1] > 1:
                combined_attention_mask = combined_attention_mask + expanded_attn_mask
            else:
                combined_attention_mask = expanded_attn_mask

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        The OPTModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in self attention to avoid performing attention to some unwanted positions,
                usually the subsequent positions.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Its data type should be float32.
                The `masked` tokens have `-1e9` values, and the `unmasked` tokens have `0` values.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            inputs_embeds (Tensor, optional):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation
                of shape `(batch_size, sequence_length, hidden_size)`. This is useful if you want more control over
                how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
                Default to None.
            use_cache (bool, optional):
                Whether or not to use cache. Defaults to `False`. If set to `True`, key value states will be returned and
                can be used to speed up decoding.
            cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)`.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail. Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail. Defaults to `None`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` object. If `False`, the output
                will be a tuple of tensors. Defaults to `None`.


        Returns:
            Tensor: Returns tensor `encoder_output`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import OPTModel, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('facebook/opt-125m')

                model = OPTModel.from_pretrained('facebook/opt-125m')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLimage.pngP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """
        if position_ids is not None:
            logger.warning("position_ids has not required for OPTModel.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = paddle.shape(input_ids)
            input_ids = input_ids.reshape((-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = paddle.shape(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        self.checkpoints = []
        past_key_values_length = paddle.shape(cache[0].k)[2] if cache is not None else 0

        seq_length_with_past = input_shape[-1] + past_key_values_length

        if attention_mask is None:
            attention_mask = paddle.ones((input_shape[0], seq_length_with_past), dtype=paddle.bool)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_embeddings=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length)
        attention_mask.stop_gradient = True

        outputs = self.decoder.forward(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            use_cache=use_cache,
            cache=cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if output_hidden_states:
            if return_dict:
                outputs.hidden_states = (embedding_output,) + outputs.hidden_states
            else:
                # [last_hidden_state, caches, all_hidden_states, all_self_attentions]
                idx = 2 if use_cache else 1
                all_hidden_states = ((embedding_output,) + outputs[idx],)
                outputs = outputs[:idx] + all_hidden_states + outputs[idx + 1 :]

        self.checkpoints.extend(self.decoder.checkpoints)
        return outputs

    def get_input_embeddings(self):
        """get opt input word embedding
        Returns:
            nn.Embedding: the input word embedding of opt mdoel
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, embedding: nn.Embedding):
        """set opt input embedding
        Returns:
            nn.Embedding: the instance of new word embedding
        """
        self.embeddings.word_embeddings = embedding


class OPTLMHead(Layer):
    def __init__(self, config: OPTConfig, embedding_weights=None):
        super(OPTLMHead, self).__init__()
        self.config = config
        self.decoder_weight = (
            self.create_parameter(shape=[config.vocab_size, config.hidden_size], dtype=config.dtype, is_bias=True)
            if embedding_weights is None
            else embedding_weights
        )

    def forward(self, hidden_states):
        if isinstance(hidden_states, BaseModelOutputWithPastAndCrossAttentions):
            hidden_states = hidden_states["last_hidden_state"]
        logits = paddle.tensor.matmul(hidden_states, self.decoder_weight.cast(hidden_states.dtype), transpose_y=True)
        return logits


class OPTForCausalLM(OPTPretrainedModel):
    """
    The OPT Model with a `language modeling` head on top.

    Args:
        config (:class:`OPTConfig`):
            An instance of OPTConfig used to construct OPTModel.

    """

    def __init__(self, config: OPTConfig):
        super(OPTForCausalLM, self).__init__(config)
        self.opt = OPTModel(config)
        self.lm_head = OPTLMHead(
            config,
            embedding_weights=self.opt.embeddings.word_embeddings.weight,
        )

    def _get_model_inputs_spec(self, dtype: str):
        return {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`OPTModel`.
            attention_mask (Tensor, optional):
                See :class:`OPTModel`.
            inputs_embeds (Tensor, optional):
                See :class:`GPTModel`.
            use_cache (bool, optional):
                See :class:`OPTModel`.
            cache (Tensor, optional):
                See :class:`OPTModel`.
            labels (paddle.Tensor, optional):
                A Tensor of shape `(batch_size, sequence_length)`.
                Labels for language modeling. Note that the labels are shifted inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., vocab_size]`
                Defaults to None.
            output_attentions (bool, optional):
                See :class:`GPTModel`.
            output_hidden_states (bool, optional):
                See :class:`GPTModel`.
            return_dict (bool, optional):
                See :class:`GPTModel`.
        Returns:
            Tensor or tuple: Returns tensor `logits` or tuple `(logits, cached_kvs)`. If `use_cache` is True,
            tuple (`logits, cached_kvs`) will be returned. Otherwise, tensor `logits` will be returned.
            `logits` is the output of the opt model.
            `cache_kvs` is the cache output of opt model if `use_cache` is True.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import OPTForCausalLM, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('facebook/opt-125m')
                model = OPTForCausalLM.from_pretrained('facebook/opt-125m')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output_ids, score = model.generate(input_ids=inputs['input_ids'])
                print(tokenizer.batch_decode(output_ids[0]))
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.opt(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = self.lm_head(encoder_outputs)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        if not return_dict:
            if not use_cache:
                return (loss, logits) if loss is not None else logits

            outputs = (logits,) + outputs[1:]
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_fast_entry(self, kwargs: Dict[str, Any]):
        # import FasterOPT at here to avoid cycling import
        from paddlenlp.ops import FasterOPT

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decode_strategy = kwargs.get("decode_strategy")
        # decoding_lib can be passed into FasterOPT
        decoding_lib = kwargs.get("decoding_lib", None)

        if decode_strategy == "beam_search":
            raise AttributeError("'beam_search' is not supported yet in the fast version of OPT")
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.opt.config["hidden_size"] // self.opt.config["num_attention_heads"]
        if size_per_head not in [32, 64, 80, 96, 128]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the fast version of OPT" % size_per_head
            )
        if kwargs["forced_bos_token_id"] is not None:
            # not support for forced_bos_token_id yet in the fast version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the fast version")
        if kwargs["min_length"] != 0:
            # not support for min_length yet in the fast version
            raise AttributeError("'min_length != 0' is not supported yet in the fast version")
        self._fast_entry = FasterOPT(self, use_fp16_decoding=use_fp16_decoding, decoding_lib=decoding_lib).forward
        return self._fast_entry

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, cache=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if cache is not None:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "cache": cache,
                "use_cache": True,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return attention_mask

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e


OPTForConditionalGeneration = OPTForCausalLM
