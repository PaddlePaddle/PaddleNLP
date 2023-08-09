# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.distributed.fleet.utils import recompute
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from paddle.nn.layer.transformer import _convert_param_attr_to_list

from ...layers import Linear as TransposedLinear
from ...utils.converter import StateDictNameMapping
from ...utils.log import logger
from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from .configuration import (
    GPT_PRETRAINED_INIT_CONFIGURATION,
    GPT_PRETRAINED_RESOURCE_FILES_MAP,
    GPTConfig,
)

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

__all__ = [
    "GPTModel",
    "GPTPretrainedModel",
    "GPTForPretraining",
    "GPTPretrainingCriterion",
    "GPTForGreedyGeneration",
    "GPTLMHeadModel",
    "GPTForTokenClassification",
    "GPTForSequenceClassification",
    "GPTForCausalLM",
]


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
        config,
        kdim=None,
        vdim=None,
        need_weights=False,
        weight_attr=None,
        bias_attr=None,
    ):
        super(MultiHeadAttention, self).__init__()
        self.config = config

        embed_dim = config.hidden_size
        self.embed_dim = config.hidden_size
        self.kdim = kdim if kdim is not None else config.hidden_size
        self.vdim = vdim if vdim is not None else config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_probs_dropout_prob
        self.need_weights = need_weights
        self.fuse_attention_qkv = config.fuse_attention_qkv

        self.head_dim = embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self.fuse_attention_qkv:
            assert self.kdim == embed_dim
            assert self.vdim == embed_dim
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, weight_attr, bias_attr=bias_attr)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
            self.k_proj = nn.Linear(self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
            self.v_proj = nn.Linear(self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = nn.Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

    def _fuse_prepare_qkv(self, query, use_cache=False, cache=None):
        mix_layer = self.qkv_proj(query)
        # bs, seqlen, nhead, headdim
        mix_layer = paddle.reshape_(mix_layer, [0, 0, self.num_heads, 3 * self.head_dim])
        # bs, nhead, seqlen, headdim
        if not self.config.use_flash_attention:
            # falsh attn need: [ bz, seqlen, nhead, head_dim]
            mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])

        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)

        assert not isinstance(cache, self.StaticCache), "cache currently does not support the StaticCache type"

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
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
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        if not self.config.use_flash_attention:
            # falsh attn need: [ bz, seqlen, nhead, head_dim]
            q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
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

        if self.config.use_flash_attention:
            # Flash Attention now ignore attention mask
            # Current Flash Attention doesn't support attn maskt
            # Paddle Flash Attention input [ bz, seqlen, nhead, head_dim]
            # Torch Flash Attention input [ bz, nhead, seqlen, head_dim]
            bsz, q_len, num_heads, head_dim = q.shape
            # Q Shape:  [1, 16, 2048, 64]
            # bs, nhead, seqlen, head_dim
            attn_output, weights = flash_attention(
                q,
                k,
                v,
                causal=q.shape[1] != 1,
                return_softmax=self.need_weights,
            )
            out = attn_output.reshape([bsz, q_len, head_dim * num_heads])
        else:
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

        # projectt to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(self, decoder_layers, num_layers, norm=None, hidden_size=None):
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")

        self.enable_recompute = False
        self.checkpoints = []

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        past_key_value: paddle.Tensor,
        attention_mask: paddle.Tensor,
        use_cache: bool,
        cache: paddle.Tensor,
        output_attentions: paddle.Tensor,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_attentions)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            past_key_value,
            attention_mask,
            use_cache,
            cache,
            use_reentrant=False,
        )
        return hidden_states

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        use_cache=False,
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
            has_gradient = not output.stop_gradient
            # def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None, output_attentions=False):
            if self.enable_recompute and has_gradient:
                outputs = self.recompute_training(
                    mod,
                    output,
                    memory,
                    tgt_mask,
                    use_cache,
                    None,
                    output_attentions,
                )
            else:
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

        if self.norm is not None:
            output = self.norm(output)

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


class TransformerDecoderLayer(nn.Layer):
    """
    The transformer decoder layer.

    It contains multiheadattention and some linear layers.
    """

    def __init__(self, config):
        self.config = config

        d_model = config.hidden_size
        nhead = config.num_attention_heads
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

        self.self_attn = MultiHeadAttention(
            config,
            need_weights=True,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
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


class GPTEmbeddings(nn.Layer):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
    ):
        super(GPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None, inputs_embeddings=None):
        if input_ids is not None:
            input_shape = paddle.shape(input_ids)
            inputs_embeddings = self.word_embeddings(input_ids)
        else:
            input_shape = paddle.shape(inputs_embeddings)[:-1]

        if position_ids is None:
            ones = paddle.ones(input_shape, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class GPTPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained GPT models. It provides GPT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    pretrained_init_configuration = GPT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = GPT_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "gpt"
    config_class = GPTConfig

    @classmethod
    def _get_name_mappings(cls, config: GPTConfig) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["wte.weight", "embeddings.word_embeddings.weight"],
            ["wpe.weight", "embeddings.position_embeddings.weight"],
            ["ln_f.weight", "decoder.norm.weight"],
            ["ln_f.bias", "decoder.norm.bias"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [f"h.{layer_index}.ln_1.weight", f"decoder.layers.{layer_index}.norm1.weight"],
                [f"h.{layer_index}.ln_1.bias", f"decoder.layers.{layer_index}.norm1.bias"],
                [f"h.{layer_index}.ln_2.weight", f"decoder.layers.{layer_index}.norm2.weight"],
                [f"h.{layer_index}.ln_2.bias", f"decoder.layers.{layer_index}.norm2.bias"],
                [f"h.{layer_index}.mlp.c_fc.weight", f"decoder.layers.{layer_index}.linear1.weight"],
                [f"h.{layer_index}.mlp.c_fc.bias", f"decoder.layers.{layer_index}.linear1.bias"],
                [f"h.{layer_index}.mlp.c_proj.weight", f"decoder.layers.{layer_index}.linear2.weight"],
                [f"h.{layer_index}.mlp.c_proj.bias", f"decoder.layers.{layer_index}.linear2.bias"],
                [f"h.{layer_index}.attn.c_proj.weight", f"decoder.layers.{layer_index}.self_attn.out_proj.weight"],
                [f"h.{layer_index}.attn.c_proj.bias", f"decoder.layers.{layer_index}.self_attn.out_proj.bias"],
                # attention
                [
                    f"h.{layer_index}.attn.c_attn.weight",
                    f"decoder.layers.{layer_index}.self_attn.q_proj.weight",
                    "split",
                    0,
                ],
                [
                    f"h.{layer_index}.attn.c_attn.bias",
                    f"decoder.layers.{layer_index}.self_attn.q_proj.bias",
                    "split",
                    0,
                ],
                [
                    f"h.{layer_index}.attn.c_attn.weight",
                    f"decoder.layers.{layer_index}.self_attn.k_proj.weight",
                    "split",
                    1,
                ],
                [
                    f"h.{layer_index}.attn.c_attn.bias",
                    f"decoder.layers.{layer_index}.self_attn.k_proj.bias",
                    "split",
                    1,
                ],
                [
                    f"h.{layer_index}.attn.c_attn.weight",
                    f"decoder.layers.{layer_index}.self_attn.v_proj.weight",
                    "split",
                    2,
                ],
                [
                    f"h.{layer_index}.attn.c_attn.bias",
                    f"decoder.layers.{layer_index}.self_attn.v_proj.bias",
                    "split",
                    2,
                ],
            ]

            model_mappings.extend(layer_mappings)

        # downstream mappings
        if "GPT2Model" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "transformer." + mapping[0]
                mapping[1] = "gpt." + mapping[1]
        if "GPT2ForTokenClassification" in config.architectures:
            model_mappings.extend([["classifier.weight", "classifier.weight", "transpose"]])
        if "GPT2ForSequenceClassification" in config.architectures:
            model_mappings.extend([["score.weight", "score.weight", "transpose"]])
        if "GPT2LMHeadModel" in config.architectures:
            model_mappings.append(["lm_head.weight", "lm_head.decoder.weight"])

        mappings = [StateDictNameMapping(*mapping) for mapping in model_mappings]
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
                        else self.gpt.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )


@register_base_model
class GPTModel(GPTPretrainedModel):
    r"""
    The bare GPT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `GPTModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `GPTModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer and decoder layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer decoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer decoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the decoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and decoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all decoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`. Defaults to `16`.

            .. note::
                Please NOT using `type_vocab_size`, for it will be obsolete in the future..

        initializer_range (float, optional):
            The standard deviation of the normal initializer. Default to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`GPTPretrainedModel._init_weights()` for how weights are initialized in `GPTModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

    """

    def __init__(self, config: GPTConfig):
        super(GPTModel, self).__init__(config)

        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id
        self.eol_token_id = config.eol_token_id
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.bias = paddle.tril(
            paddle.ones([1, 1, config.max_position_embeddings, config.max_position_embeddings], dtype="int64")
        )

        self.embeddings = GPTEmbeddings(
            config.vocab_size,
            config.hidden_size,
            config.hidden_dropout_prob,
            config.max_position_embeddings,
            config.type_vocab_size,
            self.initializer_range,
        )

        decoder_layers = nn.LayerList()
        for i in range(config.num_hidden_layers):
            decoder_layers.append(TransformerDecoderLayer(config))

        self.decoder = TransformerDecoder(
            decoder_layers,
            config.num_hidden_layers,
            norm="LayerNorm",
            hidden_size=config.hidden_size,
        )

        self.checkpoints = []

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The GPTModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor, optional):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to None.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in self attention to avoid performing attention to some unwanted positions,
                usually the subsequent positions.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Its data type should be int64.
                The `masked` tokens have `0` values, and the `unmasked` tokens have `1` values.
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
                tensors for more detail. Defaults to `False`.
            output_hidden_states (bool, optional):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail. Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions`.

            Especially, When `return_dict=output_hidden_states=output_attentions=False`,
            returns tensor `outputs` which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import GPTModel, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
                model = GPTModel.from_pretrained('gpt2-medium-en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """

        self.checkpoints = []
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = paddle.shape(input_ids)
            input_ids = input_ids.reshape((-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = paddle.shape(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(past_length, input_shape[-1] + past_length, dtype="int64")
            position_ids = position_ids.unsqueeze(0)
            position_ids = paddle.expand(position_ids, input_shape)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeddings=inputs_embeds
        )

        # TODO, use registered buffer
        length = input_shape[-1]
        if cache is not None:
            cache_length = paddle.shape(cache[0].k)[2]
            length = length + cache_length
        else:
            cache_length = 0
        causal_mask = self.bias[:, :, cache_length:length, :length]

        if attention_mask is not None:
            if attention_mask.dtype != paddle.int64:
                attention_mask = paddle.cast(attention_mask, dtype=paddle.int64)
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - (attention_mask & causal_mask)) * -1e4
        else:
            attention_mask = (1.0 - causal_mask) * -1e4

        # The tensor returned by triu not in static graph.
        attention_mask.stop_gradient = True

        outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            use_cache=use_cache,
            cache=cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        if output_hidden_states:
            if return_dict:
                outputs.hidden_states = (embedding_output,) + outputs.hidden_states
            else:  # outputs is a tuple
                idx = 2 if use_cache else 1
                all_hidden_states = (embedding_output,) + outputs[idx]
                outputs = outputs[:idx] + (all_hidden_states) + outputs[idx + 1 :]

        self.checkpoints.extend(self.decoder.checkpoints)

        return outputs


class GPTForPretraining(GPTPretrainedModel):
    """
    GPT Model with pretraining tasks on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, config: GPTConfig):
        super(GPTForPretraining, self).__init__(config)
        self.gpt = GPTModel(config)
        self.lm_head = GPTLMHead(config)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
        use_cache=False,
        cache=None,
        labels=None,
        loss_mask=None,
    ):
        r"""

        Args:
            input_ids (Tensor, optional):
                See :class:`GPTModel`.
            position_ids (Tensor, optional):
                See :class:`GPTModel`.
            attention_mask (Tensor, optional):
                See :class:`GPTModel`.
            use_cache (bool, optional):
                See :class:`GPTModel`.
            cache (Tensor, optional):
                See :class:`GPTModel`.

        Returns:
            Tensor or tuple: Returns tensor `logits` or tuple `(logits, cached_kvs)`. If `use_cache` is True,
            tuple (`logits, cached_kvs`) will be returned. Otherwise, tensor `logits` will be returned.
            `logits` is the output of the gpt model.
            `cache_kvs` is the cache output of gpt model if `use_cache` is True.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import GPTForPretraining, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
                model = GPTForPretraining.from_pretrained('gpt2-medium-en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs,use_cache=True)

                logits = output[0]
                cached_kvs = output[1]

        """

        outputs = self.gpt(
            input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = self.lm_head(encoder_outputs)

        if labels is None:
            if use_cache:
                return logits, cached_kvs
            else:
                return logits
        else:
            loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
            masked_lm_loss = loss_func(logits, labels.unsqueeze(2))

            loss_mask = loss_mask.reshape([-1])
            masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
            loss = masked_lm_loss / loss_mask.sum()
            return loss


class GPTPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT. It calculates the final loss.
    """

    def __init__(self):
        super(GPTPretrainingCriterion, self).__init__()
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        """
        Args:
            prediction_scores(Tensor):
                The logits of masked token prediction. Its data type should be float32 and
                its shape is [batch_size, sequence_length, vocab_size].
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, the dimensionality of `masked_lm_labels`
                is equal to `prediction_scores`. Its data type should be int64 and
                its shape is [batch_size, sequence_length, 1].
            loss_mask(Tensor):
                Mask used for calculating the loss of the masked language modeling to avoid
                calculating some unwanted tokens.
                Its data type should be float32 and its shape is [batch_size, sequence_length, 1].

        Returns:
            Tensor: The pretraining loss. Its data type should be float32 and its shape is [1].

        """
        masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))

        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        loss = masked_lm_loss / loss_mask.sum()
        return loss


class GPTForGreedyGeneration(GPTPretrainedModel):
    """
    The generate model for GPT-2.
    It use the greedy strategy and generate the output sequence with highest probability.

    Args:
        gpt (:class:`GPTModel`):
            An instance of `paddlenlp.transformers.GPTModel`.
        max_predict_len(int):
            The max length of the prediction.

    """

    def __init__(self, config: GPTConfig, max_predict_len: int = 32):
        super(GPTForGreedyGeneration, self).__init__(config)
        self.gpt = GPTModel(config)
        self.max_predict_len = paddle.to_tensor(max_predict_len, dtype="int32")
        self.eol_token_id = config.eol_token_id

    def model(
        self, input_ids, position_ids=None, attention_mask=None, masked_positions=None, use_cache=False, cache=None
    ):
        r"""

        Args:
            input_ids (Tensor, optional):
                See :class:`GPTModel`.
            position_ids (Tensor, optional):
                See :class:`GPTModel`.
            attention_mask (Tensor, optional):
                See :class:`GPTModel`.
            use_cache (bool, optional):
                See :class:`GPTModel`.
            cache (Tensor, optional):
                See :class:`GPTModel`.

        Returns:
            Tensor or tuple: Returns tensor `logits` or tuple `(logits, cached_kvs)`. If `use_cache` is True,
            tuple (`logits, cached_kvs`) will be returned. Otherwise, tensor `logits` will be returned.
            `logits` is the output of the gpt model.
            `cache_kvs` is the cache output of gpt model if `use_cache` is True.

        """

        outputs = self.gpt(
            input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = paddle.matmul(encoder_outputs, self.gpt.embeddings.word_embeddings.weight, transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def forward(self, input_ids):
        """

        Args:
            input_ids(Tensor):
                See :class:`GPTModel`.

        Returns:
            Tensor: Returns tensor `src_ids`, which means the indices of output sequence tokens in the vocabulary.
            They are numerical representations of tokens that build the output sequence.
        """
        output, cached_kvs = self.model(input_ids, use_cache=True, cache=None)
        src_ids = input_ids
        nid = paddle.argmax(output[:, -1, :], axis=-1).reshape([-1, 1])
        src_ids = paddle.concat([src_ids, nid], axis=1)
        cur_len = 0
        while cur_len < self.max_predict_len:
            output, cached_kvs = self.model(nid, use_cache=True, cache=cached_kvs)

            nid = paddle.argmax(output[:, -1, :], axis=-1).reshape([-1, 1])
            src_ids = paddle.concat([src_ids, nid], axis=1)
            cur_len += 1
            if paddle.max(nid) == self.eol_token_id:
                break
        return src_ids


class GPTLMHead(nn.Layer):
    def __init__(self, config: GPTConfig):
        super(GPTLMHead, self).__init__()
        self.decoder = TransposedLinear(config.hidden_size, config.vocab_size, bias_attr=False)

    def forward(self, hidden_states):
        logits = self.decoder(hidden_states)
        return logits


class GPTLMHeadModel(GPTPretrainedModel):
    """
    The GPT Model with a `language modeling` head on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, config: GPTConfig):
        super(GPTLMHeadModel, self).__init__(config)
        self.gpt = GPTModel(config)
        self.lm_head = GPTLMHead(config)
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        cache=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""

        Args:
            input_ids (Tensor, optional):
                See :class:`GPTModel`.
            position_ids (Tensor, optional):
                See :class:`GPTModel`.
            attention_mask (Tensor, optional):
                See :class:`GPTModel`.
            inputs_embeds (Tensor, optional):
                See :class:`GPTModel`.
            use_cache (bool, optional):
                See :class:`GPTModel`.
            cache (Tensor, optional):
                See :class:`GPTModel`.
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
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions`.

            Especialy, when `return_dict=use_cache=output_attentions=output_hidden_states=False`,
            returns a tensor `logits` which is the output of the gpt model.
        """
        input_type = type(input_ids) if input_ids is not None else type(inputs_embeds)
        outputs = self.gpt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if isinstance(outputs, input_type):
            hidden_states = outputs
        else:
            hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape((-1, shift_logits.shape[-1])), shift_labels.reshape((-1,)))

        # outputs = [output, all_hidden_states, new_caches, all_self_attentions]
        if not return_dict:
            if isinstance(outputs, input_type):
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

    def prepare_fast_entry(self, kwargs):
        from paddlenlp.ops import FasterGPT

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decode_strategy = kwargs.get("decode_strategy")
        if decode_strategy == "beam_search":
            raise AttributeError("'beam_search' is not supported yet in the fast version of GPT")
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.gpt.config["hidden_size"] // self.gpt.config["num_attention_heads"]
        if size_per_head not in [32, 64, 80, 96, 128]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the fast version of GPT" % size_per_head
            )
        if kwargs["forced_bos_token_id"] is not None:
            # not support for min_length yet in the fast version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the fast version")
        if kwargs["min_length"] != 0:
            # not support for min_length yet in the fast version
            raise AttributeError("'min_length != 0' is not supported yet in the fast version")
        self._fast_entry = FasterGPT(self, use_fp16_decoding=use_fp16_decoding).forward
        return self._fast_entry

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, cache=None, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, -1:, -1:, :]
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
        }

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
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)


class GPTForTokenClassification(GPTPretrainedModel):
    """
    GPT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.

    Args:
        gpt (:class:`GPTModel`):
            An instance of GPTModel.
        num_labels (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of GPT.
            If None, use the same value as `hidden_dropout_prob` of `GPTModel`
            instance `gpt`. Defaults to None.
    """

    def __init__(self, config: GPTConfig):
        super(GPTForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.gpt = GPTModel(config)  # allow gpt to be config
        dropout_p = config.hidden_dropout_prob if config.classifier_dropout is None else config.classifier_dropout
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The GPTForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor, optional):
                See :class:`GPTModel`.
            position_ids(Tensor, optional):
                See :class:`GPTModel`.
            attention_mask (list, optional):
                See :class:`GPTModel`.
            inputs_embeds (Tensor, optional):
                See :class:`GPTModel`.
            labels (Tensor, optional):
                Labels of shape `(batch_size, sequence_length)` for computing the sequence classification/regression loss. Indices should be in
                `[0, ..., num_labels - 1]`. If `num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `num_labels > 1` a classification loss is computed (Cross-Entropy). Defaults to None.
            output_attentions (bool, optional):
                See :class:`GPTModel`.
            output_hidden_states (bool, optional):
                See :class:`GPTModel`.
            return_dict (bool, optional):
                See :class:`GPTModel`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput`.

            Especialy, when `return_dict=output_attentions=output_hidden_states=False`,
            returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_labels]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import GPTForTokenClassification, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
                model = GPTForTokenClassification.from_pretrained('gpt2-medium-en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        input_type = type(input_ids) if input_ids is not None else type(inputs_embeds)
        sequence_output = self.gpt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if isinstance(sequence_output, input_type):
            hidden_states = sequence_output
        else:
            hidden_states = sequence_output[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))

        if not return_dict:
            if isinstance(sequence_output, input_type):
                return (loss, logits) if loss is not None else logits

            outputs = (logits,) + sequence_output[1:]
            return ((loss,) + outputs) if loss is not None else outputs

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=sequence_output.hidden_states,
            attentions=sequence_output.attentions,
        )


class GPTForSequenceClassification(GPTPretrainedModel):
    """
    GPT Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.

    Args:
        gpt (:class:`GPTModel`):
            An instance of GPTModel.
        num_labels (int, optional):
            The number of classes. Defaults to `2`.

    """

    def __init__(self, config: GPTConfig):
        super(GPTForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.gpt = GPTModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias_attr=False)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The GPTForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor, optional):
                See :class:`GPTModel`.
            position_ids(Tensor, optional):
                See :class:`GPTModel`.
            attention_mask (list, optional):
                See :class:`GPTModel`.
            inputs_embeds (Tensor, optional):
                See :class:`GPTModel`.
            labels (Tensor, optional):
                Labels of shape `(batch_size, sequence_length)` for computing the sequence classification/regression loss. Indices should be in
                `[0, ..., num_labels - 1]`. If `num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `num_labels > 1` a classification loss is computed (Cross-Entropy). Defaults to None.
            use_cache (bool, optional):
                See :classL `GPTModel`.
            output_attentions (bool, optional):
                See :class:`GPTModel`.
            output_hidden_states (bool, optional):
                See :class:`GPTModel`.
            return_dict (bool, optional):
                See :class:`GPTModel`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutputWithPast` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutputWithPast`.

            Especialy, when `return_dict=output_attentions=output_hidden_states=False`,
            returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_labels]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import GPTForSequenceClassification, GPTTokenizer

                tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
                model = GPTForSequenceClassification.from_pretrained('gpt2-medium-en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_token_type_ids=False)
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        input_type = type(input_ids) if input_ids is not None else type(inputs_embeds)
        # sequence_output shape [bs, seq_len, hidden_size]
        sequence_output = self.gpt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if isinstance(sequence_output, input_type):
            hidden_states = sequence_output
        else:
            hidden_states = sequence_output[0]
        # logits shape [bs, seq_len, num_class]
        logits = self.score(hidden_states)
        # padding index maybe 0
        eos_token_id = self.gpt.config.eos_token_id or 0
        # sequence_lengths shape [bs,]
        if input_ids is not None:
            sequence_lengths = (input_ids != eos_token_id).astype("int64").sum(axis=-1) - 1
        else:
            inputs_shape = paddle.shape(inputs_embeds)[:-1]
            sequence_lengths = paddle.ones(inputs_shape[:-1], dtype="int64") * (inputs_shape[1] - 1)
            logger.warning(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits.gather_nd(
            paddle.stack([paddle.arange(paddle.shape(logits)[0]), sequence_lengths], axis=-1)
        )

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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            if isinstance(sequence_output, input_type):
                return (loss, pooled_logits) if loss is not None else pooled_logits

            outputs = (pooled_logits,) + sequence_output[1:]
            return ((loss,) + outputs) if loss is not None else outputs

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=sequence_output.past_key_values,
            hidden_states=sequence_output.hidden_states,
            attentions=sequence_output.attentions,
        )


GPTForCausalLM = GPTLMHeadModel
