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

import collections
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list

from .. import PretrainedModel, register_base_model

__all__ = [
    'GPTModel',
    'GPTPretrainedModel',
    'GPTForPretraining',
    'GPTPretrainingCriterion',
    'GPTForGreedyGeneration',
    'GPTLMHeadModel',
    'GPTForTokenClassification',
    'GPTForSequenceClassification',
    'GPTForCausalLM',
    'GPTForGeneration',
    'GPTForStaticGeneration',
]

int_type = "int64"


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None,
                 topo=None,
                 fuse=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse = fuse

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self.fuse:
            assert self.kdim == embed_dim
            assert self.vdim == embed_dim
            self.qkv_proj = nn.Linear(embed_dim,
                                      3 * embed_dim,
                                      weight_attr,
                                      bias_attr=bias_attr)
        else:
            self.q_proj = nn.Linear(embed_dim,
                                    embed_dim,
                                    weight_attr,
                                    bias_attr=bias_attr)
            self.k_proj = nn.Linear(self.kdim,
                                    embed_dim,
                                    weight_attr,
                                    bias_attr=bias_attr)
            self.v_proj = nn.Linear(self.vdim,
                                    embed_dim,
                                    weight_attr,
                                    bias_attr=bias_attr)
        self.out_proj = nn.Linear(embed_dim,
                                  embed_dim,
                                  weight_attr,
                                  bias_attr=bias_attr)

    def _fuse_prepare_qkv(self, query):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer,
                                    [0, 0, self.num_heads, 3 * self.head_dim])
        mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)
        return q, k, v

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        """
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
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

        return (q, k, v) if use_cache is False else (q, k, v, cache)

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
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self,
                query,
                key,
                value,
                attn_mask=None,
                use_cache=False,
                cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if use_cache is False:
            if self.fuse:
                q, k, v = self._fuse_prepare_qkv(query)
            else:
                q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache,
                                               cache)
        # scale dot product attention
        product = layers.matmul(x=q,
                                y=k,
                                transpose_y=True,
                                alpha=self.head_dim**-0.5)

        if attn_mask is not None:
            product = product + attn_mask

        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(weights,
                                self.dropout,
                                training=self.training,
                                mode="upscale_in_train")

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


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(self,
                 decoder_layers,
                 num_layers,
                 norm=None,
                 hidden_size=None,
                 topo=None):
        super(TransformerDecoder, self).__init__()

        self.topo = topo
        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")
        self.checkpoints = []

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                use_cache=False,
                cache=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []
        self.checkpoints = []

        for i, mod in enumerate(self.layers):
            if cache is None:
                if use_cache:
                    output, new_cache = mod(output,
                                            memory,
                                            tgt_mask=tgt_mask,
                                            use_cache=use_cache,
                                            cache=cache)
                    new_caches.append(new_cache)
                else:
                    output = mod(output,
                                 memory,
                                 tgt_mask=tgt_mask,
                                 use_cache=use_cache,
                                 cache=cache)

            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        use_cache=use_cache,
                                        cache=cache[i])
                new_caches.append(new_cache)
            self.checkpoints.append(output.name)

        if self.norm is not None:
            output = self.norm(output)
        return output if use_cache is False else (output, new_caches)

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

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=True,
                 weight_attr=None,
                 bias_attr=None,
                 topo=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(d_model,
                                            nhead,
                                            dropout=attn_dropout,
                                            weight_attr=weight_attrs[0],
                                            bias_attr=bias_attrs[0],
                                            topo=topo)
        self.linear1 = nn.Linear(d_model,
                                 dim_feedforward,
                                 weight_attrs[2],
                                 bias_attr=bias_attrs[2])
        self.linear2 = nn.Linear(dim_feedforward,
                                 d_model,
                                 weight_attrs[2],
                                 bias_attr=bias_attrs[2])

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None):
        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    use_cache, cache)
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.dropout2(
            self.linear2(F.gelu(self.linear1(tgt), approximate=True)))
        tgt = residual + tgt

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        return tgt if use_cache is False else (tgt, incremental_cache)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory,
                                                     type=self.self_attn.Cache)
        return incremental_cache


class GPTEmbeddings(nn.Layer):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 topo=None):
        super(GPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=initializer_range)))

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=initializer_range)))

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings
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

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "gpt-cpm-large-cn": { # 2.6B
            "vocab_size": 30000,
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 10240,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "eos_token_id": 7,
            "bos_token_id": 0,
            "eol_token_id": 3,
        },
        "gpt-cpm-small-cn-distill": { # 109M
            "vocab_size": 30000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "eos_token_id": 7,
            "bos_token_id": 0,
            "eol_token_id": 3,
        },
        "gpt3-13B-en": { # 13B
            "vocab_size": 50304,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 128,
            "intermediate_size": 20480,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
        },
        "gpt3-1.3B-en": { # 1.3B
            "vocab_size": 50304,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 8192,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
        },
        "gpt2-xl-en": { # 1558M
            "vocab_size": 50257,
            "hidden_size": 1600,
            "num_hidden_layers": 48,
            "num_attention_heads": 25,
            "intermediate_size": 6400,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
        },
        "gpt2-large-en": { # 774M
            "vocab_size": 50257,
            "hidden_size": 1280,
            "num_hidden_layers": 36,
            "num_attention_heads": 20,
            "intermediate_size": 5120,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
        },
        "gpt2-medium-en": { #345M
            "vocab_size": 50304,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
        },
        "gpt2-en": { #117M
            "vocab_size": 50257,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
        },
        "gpt2-small-en": { # config for CE
            "vocab_size": 50304,
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "gpt-cpm-large-cn":
            "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt-cpm-large-cn.pdparams",
            "gpt-cpm-small-cn-distill":
            "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt-cpm-small-cn-distill.pdparams",
            "gpt2-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-en.pdparams",
            "gpt2-medium-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-medium-en.pdparams",
            "gpt2-large-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-large-en.pdparams",
            "gpt2-xl-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-xl-en.pdparams",
        }
    }
    base_model_prefix = "gpt"

    def init_weights(self, layer):
        """ Initialization hook """
        # no hook
        return
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(mean=0.0,
                                         std=self.initializer_range if hasattr(
                                             self, "initializer_range") else
                                         self.gpt.config["initializer_range"],
                                         shape=layer.weight.shape))


@register_base_model
class GPTModel(GPTPretrainedModel):
    r"""
    The bare GPT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
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

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 eos_token_id=7,
                 bos_token_id=0,
                 eol_token_id=3,
                 topo=None):
        super(GPTModel, self).__init__()

        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.topo = topo
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embeddings = GPTEmbeddings(vocab_size, hidden_size,
                                        hidden_dropout_prob,
                                        max_position_embeddings,
                                        type_vocab_size, self.initializer_range,
                                        topo)

        decoder_layers = nn.LayerList()
        for i in range(num_hidden_layers):
            decoder_layers.append(
                TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    dropout=hidden_dropout_prob,
                    activation=hidden_act,
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=self.initializer_range)),
                    bias_attr=None,
                    topo=topo))

        self.decoder = TransformerDecoder(decoder_layers,
                                          num_hidden_layers,
                                          norm="LayerNorm",
                                          hidden_size=hidden_size,
                                          topo=topo)

        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                cache=None):
        r'''
        The GPTModel forward method, overrides the `__call__()` special method.

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
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Its data type should be float32.
                The `masked` tokens have `-1e-9` values, and the `unmasked` tokens have `0` values.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            use_cache (bool, optional):
                Whether or not to use cache. Defaults to `False`. If set to `True`, key value states will be returned and
                can be used to speed up decoding.
            cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)`.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.

        Returns:
            Tensor: Returns tensor `encoder_output`, which is the output at the last layer of the model.
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
        '''

        self.checkpoints = []
        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(past_length,
                                         paddle.shape(input_ids)[-1] +
                                         past_length,
                                         dtype=input_ids.dtype)
            position_ids = position_ids.unsqueeze(0)
            # .expand_as(input_ids)
            position_ids = paddle.fluid.layers.expand_as(
                position_ids, input_ids)
        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids)

        # TODO, use registered buffer
        causal_mask = paddle.tensor.triu(paddle.ones(
            (paddle.shape(input_ids)[-1], paddle.shape(input_ids)[-1])) * -1e4,
                                         diagonal=1)

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask
        # The tensor returned by triu not in static graph.
        attention_mask.stop_gradient = True

        encoder_outputs = self.decoder(embedding_output,
                                       memory=None,
                                       tgt_mask=attention_mask,
                                       use_cache=use_cache,
                                       cache=cache)
        self.checkpoints.extend(self.decoder.checkpoints)
        return encoder_outputs


class GPTForPretraining(GPTPretrainedModel):
    """
    GPT Model with pretraining tasks on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, gpt):
        super(GPTForPretraining, self).__init__()
        self.gpt = gpt
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                use_cache=False,
                cache=None):
        r"""

        Args:
            input_ids (Tensor):
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

        outputs = self.gpt(input_ids,
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=use_cache,
                           cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = paddle.matmul(encoder_outputs,
                               self.gpt.embeddings.word_embeddings.weight,
                               transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT. It calculates the final loss.
    """

    def __init__(self, topo=None):
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
        masked_lm_loss = self.loss_func(prediction_scores,
                                        masked_lm_labels.unsqueeze(2))

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

    def __init__(self, gpt, max_predict_len, eol_token_id=3):
        super(GPTForGreedyGeneration, self).__init__()
        self.gpt = gpt
        self.max_predict_len = paddle.to_tensor(max_predict_len, dtype='int32')
        self.eol_token_id = eol_token_id
        self.apply(self.init_weights)

    def model(self,
              input_ids,
              position_ids=None,
              attention_mask=None,
              masked_positions=None,
              use_cache=False,
              cache=None):
        r"""

        Args:
            input_ids (Tensor):
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
        outputs = self.gpt(input_ids,
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=use_cache,
                           cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = paddle.matmul(encoder_outputs,
                               self.gpt.embeddings.word_embeddings.weight,
                               transpose_y=True)

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
        while (cur_len < self.max_predict_len):
            output, cached_kvs = self.model(nid,
                                            use_cache=True,
                                            cache=cached_kvs)

            nid = paddle.argmax(output[:, -1, :], axis=-1).reshape([-1, 1])
            src_ids = paddle.concat([src_ids, nid], axis=1)
            cur_len += 1
            if paddle.max(nid) == self.eol_token_id:
                break
        return src_ids


class GPTLMHead(nn.Layer):

    def __init__(self, hidden_size, vocab_size, embedding_weights=None):
        super(GPTLMHead, self).__init__()
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=paddle.get_default_dtype(),
            is_bias=True) if embedding_weights is None else embedding_weights

    def forward(self, hidden_states):
        logits = paddle.tensor.matmul(hidden_states,
                                      self.decoder_weight,
                                      transpose_y=True)
        return logits


class GPTLMHeadModel(GPTPretrainedModel):
    """
    The GPT Model with a `language modeling` head on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, gpt):
        super(GPTLMHeadModel, self).__init__()
        self.gpt = gpt
        self.lm_head = GPTLMHead(self.gpt.config["hidden_size"],
                                 self.gpt.config["vocab_size"],
                                 self.gpt.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                cache=None):
        r"""

        Args:
            input_ids (Tensor):
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

        outputs = self.gpt(input_ids,
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=use_cache,
                           cache=cache)

        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = self.lm_head(encoder_outputs)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def prepare_faster_entry(self, kwargs):
        from paddlenlp.ops import FasterGPT
        use_fp16_decoding = kwargs.get('use_fp16_decoding', False)
        decode_strategy = kwargs.get('decode_strategy')
        if decode_strategy == "beam_search":
            raise AttributeError(
                "'beam_search' is not supported yet in the faster version of GPT"
            )
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.gpt.config["hidden_size"] // self.gpt.config[
            "num_attention_heads"]
        if size_per_head not in [32, 64, 80, 96, 128]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the faster version of GPT"
                % size_per_head)
        if kwargs['forced_bos_token_id'] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError(
                "'forced_bos_token_id != None' is not supported yet in the faster version"
            )
        if kwargs['min_length'] != 0:
            # not support for min_length yet in the faster version
            raise AttributeError(
                "'min_length != 0' is not supported yet in the faster version")
        self._faster_entry = FasterGPT(
            self, use_fp16_decoding=use_fp16_decoding).forward
        return self._faster_entry

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.common_ops_import.convert_dtype(
                    attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }

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


class GPTForTokenClassification(GPTPretrainedModel):
    """
    GPT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.

    Args:
        gpt (:class:`GPTModel`):
            An instance of GPTModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of GPT.
            If None, use the same value as `hidden_dropout_prob` of `GPTModel`
            instance `gpt`. Defaults to None.
    """

    def __init__(self, gpt, num_classes=2, dropout=None):
        super(GPTForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.gpt = gpt  # allow gpt to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.gpt.
                                  config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.gpt.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        The GPTForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`GPTModel`.
            position_ids(Tensor, optional):
                See :class:`GPTModel`.
            attention_mask (list, optional):
                See :class:`GPTModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

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
        sequence_output = self.gpt(input_ids,
                                   position_ids=position_ids,
                                   attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class GPTForSequenceClassification(GPTPretrainedModel):
    """
    GPT Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.

    Args:
        gpt (:class:`GPTModel`):
            An instance of GPTModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
            
    """

    def __init__(self, gpt, num_classes=2):
        super(GPTForSequenceClassification, self).__init__()
        self.gpt = gpt  # allow gpt to be config
        self.score = nn.Linear(self.gpt.config["hidden_size"],
                               num_classes,
                               bias_attr=False)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        The GPTForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`GPTModel`.
            position_ids(Tensor, optional):
                See :class:`GPTModel`.
            attention_mask (list, optional):
                See :class:`GPTModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

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

        # sequence_output shape [bs, seq_len, hidden_size]
        sequence_output = self.gpt(input_ids,
                                   position_ids=position_ids,
                                   attention_mask=attention_mask)
        # logits shape [bs, seq_len, num_class]
        logits = self.score(sequence_output)
        # padding index maybe 0
        eos_token_id = self.gpt.config.get("eos_token_id", 0)
        # sequence_lengths shape [bs,]
        sequence_lengths = (input_ids != eos_token_id).astype("int64").sum(
            axis=-1) - 1
        pooled_logits = logits.gather_nd(
            paddle.stack(
                [paddle.arange(sequence_output.shape[0]), sequence_lengths],
                axis=-1))

        return pooled_logits


class GPTForGeneration(GPTPretrainedModel):

    def __init__(self, gpt, **kwargs):
        super(GPTForGeneration, self).__init__()
        self.gpt = gpt
        self.apply(self.init_weights)

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.common_ops_import.convert_dtype(
                    attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache
        }

    def model(self,
              input_ids,
              position_ids=None,
              attention_mask=None,
              masked_positions=None,
              use_cache=False,
              cache=None):
        outputs = self.gpt(input_ids,
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=use_cache,
                           cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = paddle.matmul(encoder_outputs,
                               self.gpt.embeddings.word_embeddings.weight,
                               transpose_y=True)
        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def forward(self,
                input_ids=None,
                max_length=20,
                min_length=0,
                decode_strategy='greedy_search',
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                num_beams=1,
                num_beam_groups=1,
                length_penalty=0.0,
                early_stopping=False,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                decoder_start_token_id=None,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                num_return_sequences=1,
                diversity_rate=0.0,
                use_cache=True,
                **model_kwargs):
        assert (
            decode_strategy in ["greedy_search", "sampling", "beam_search"]
        ), "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            decode_strategy)

        bos_token_id = bos_token_id if bos_token_id is not None else getattr(
            self.gpt, 'bos_token_id', None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(
            self.gpt, 'eos_token_id', None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(
            self.gpt, 'pad_token_id', None)
        forced_bos_token_id = forced_bos_token_id if forced_bos_token_id is not None else getattr(
            self.gpt, 'forced_bos_token_id', None)
        forced_eos_token_id = forced_eos_token_id if forced_eos_token_id is not None else getattr(
            self.gpt, 'forced_eos_token_id', None)
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else getattr(
            self.gpt, 'decoder_start_token_id', None)

        # params check
        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs[
                "attention_mask"] = self.prepare_attention_mask_for_generation(
                    input_ids, pad_token_id, eos_token_id)
        self.is_encoder_decoder = False

        model_kwargs["use_cache"] = use_cache
        max_length += input_ids.shape[-1]
        min_length += input_ids.shape[-1]

        logits_processors = self.get_logits_processor(
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty)

        if decode_strategy == 'greedy_search':
            if num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(num_return_sequences))

            return self.greedy_search(input_ids, logits_processors, max_length,
                                      pad_token_id, eos_token_id,
                                      **model_kwargs)

        elif decode_strategy == 'sampling':
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs)

            return self.sample(input_ids, logits_processors, max_length,
                               pad_token_id, eos_token_id, top_k, top_p,
                               temperature, **model_kwargs)
        else:
            raise ValueError(f'Not support {decoding_strategy} strategy yet!')

    def sample(self,
               input_ids,
               logits_processors,
               max_length,
               pad_token_id,
               eos_token_id,
               top_k=None,
               top_p=None,
               temperature=None,
               min_tokens_to_keep=1,
               **model_kwargs):

        def TopKProcess(probs, top_k, min_tokens_to_keep):
            top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
            # Remove all tokens with a probability less than the last token of the top-k
            topk_probs, _ = paddle.topk(probs, k=top_k)
            probs = paddle.where(probs >= topk_probs[:, -1:], probs,
                                 paddle.full_like(probs, 0.0))
            return probs

        def TopPProcess(probs, top_p, min_tokens_to_keep):
            sorted_probs = paddle.sort(probs, descending=True)
            sorted_indices = paddle.argsort(probs, descending=True)
            cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

            # Remove tokens with cumulative probs above the top_p, But keep at
            # least min_tokens_to_keep tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Set 'min_tokens_to_keep - 1' because the first token is kept
                sorted_indices_to_remove[:, :min_tokens_to_keep - 1] = 0
            # Keep the first token
            sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove,
                                                   dtype='int64')
            sorted_indices_to_remove[:, 1:] = (
                sorted_indices_to_remove[:, :-1].clone())
            sorted_indices_to_remove[:, 0] = 0

            # Scatter sorted tensors to original indexing
            sorted_indices = sorted_indices + paddle.arange(
                probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
            condition = paddle.scatter(sorted_indices_to_remove.flatten(),
                                       sorted_indices.flatten(),
                                       sorted_indices_to_remove.flatten())
            condition = paddle.cast(condition, 'bool').reshape(probs.shape)
            probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
            return probs

        batch_size, cur_len = input_ids.shape
        origin_len = input_ids.shape[1]
        unfinished_flag = paddle.full([batch_size, 1], True, dtype='bool')
        scores = paddle.full([batch_size, 1],
                             0.0,
                             dtype=paddle.get_default_dtype())

        # Pre-while call for inference,
        # the value in model_kwargs should be tensor before while loop
        if cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)
            outputs = self.model(**model_inputs)
            model_kwargs["cache"] = outputs[1]

        while cur_len < max_length:
            # prepare model inputs & get model output
            # skip first step for pre-while call
            if cur_len > origin_len:
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, **model_kwargs)
                outputs = self.model(**model_inputs)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)
            next_tokens = paddle.multinomial(probs)

            next_scores = paddle.index_sample(origin_probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(
                    unfinished_flag, next_tokens,
                    paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores,
                                                       cur_len - origin_len,
                                                       unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(
                    unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break
            model_kwargs = self.update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.is_encoder_decoder)

        return input_ids[:, origin_len:], scores

    def greedy_search(self, input_ids, logits_processors, max_length,
                      pad_token_id, eos_token_id, **model_kwargs):
        batch_size, cur_len = input_ids.shape
        origin_len = input_ids.shape[1]  # cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype='bool')
        scores = paddle.full([batch_size, 1],
                             0.0,
                             dtype=paddle.get_default_dtype())

        # Pre-while call for inference,
        # the value in model_kwargs should be tensor before while loop
        if cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)
            outputs = self.model(**model_inputs)
            model_kwargs["cache"] = outputs[1]

        while cur_len < max_length:
            # prepare model inputs & get model output
            # skip first step for pre-while call
            if cur_len > origin_len:
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, **model_kwargs)
                outputs = self.model(**model_inputs)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # [batch_size, vocab_size]
            logits = logits[:, -1, :]
            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)
            # greedy
            probs = F.softmax(logits)
            probs = paddle.log(probs)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(
                    unfinished_flag, next_tokens,
                    paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores,
                                                       cur_len - origin_len,
                                                       unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(
                    unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.is_encoder_decoder)

        return input_ids[:, origin_len:], scores


class GPTForStaticGeneration(GPTPretrainedModel):

    def __init__(self,
                 gpt,
                 max_length=20,
                 min_length=0,
                 decode_strategy='sampling',
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 eos_id=None,
                 **kwargs):
        super(GPTForStaticGeneration, self).__init__()
        self.gpt = gpt
        self.apply(self.init_weights)
        self.vocab_size = gpt.vocab_size
        self.eos_token_id = eos_id or 7

        self.min_dec_len = min_length
        self.max_dec_len = max_length
        self.decoding_strategy = decode_strategy
        self.temperature = temperature
        self.topk = top_k
        self.topp = top_p

    def parallel_matmul(self,
                        lm_output,
                        logit_weights,
                        parallel_output,
                        topo=None):
        if topo is not None and topo.mp_info.size > 1:
            hybrid_groups = fleet.get_hybrid_communicate_group()
            model_parallel_group = hybrid_groups.get_model_parallel_group()

            input_parallel = paddle.distributed.collective._c_identity(
                lm_output, group=model_parallel_group)

            logits = paddle.matmul(input_parallel,
                                   logit_weights,
                                   transpose_y=True)

            if parallel_output:
                return logits

            return paddle.distributed.collective._c_concat(
                logits, group=model_parallel_group)
        else:
            logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
            return logits

    def topk_sampling(self, probs):
        topk_probs, _ = paddle.topk(probs, self.topk)
        ge_cond = paddle.cast(
            paddle.greater_equal(probs,
                                 paddle.unsqueeze(topk_probs[:, -1], [1])),
            "float32")
        old_probs = probs
        probs = probs * ge_cond / paddle.sum(topk_probs, axis=-1, keepdim=True)
        sampling_ids = paddle.fluid.layers.sampling_id(probs)
        probs = old_probs
        return probs, sampling_ids

    def topp_sampling(self, probs):
        sorted_probs, sorted_idx = layers.argsort(probs, descending=True)
        cum_sorted_probs = layers.cumsum(sorted_probs, axis=1, exclusive=True)
        lt_cond = paddle.cast(
            cum_sorted_probs < paddle.full_like(cum_sorted_probs,
                                                fill_value=self.topp),
            "float32")

        old_probs = probs
        candidate_probs = sorted_probs * lt_cond
        probs = candidate_probs / paddle.sum(
            candidate_probs, axis=-1, keepdim=True)
        sampling_ids = layers.sampling_id(probs, dtype="int")
        sampling_ids = paddle.index_sample(sorted_idx,
                                           paddle.unsqueeze(sampling_ids, [1]))
        sampling_ids = paddle.squeeze(sampling_ids, [1])
        probs = old_probs
        return probs, sampling_ids

    def model(self,
              input_ids,
              position_ids=None,
              attention_mask=None,
              masked_positions=None,
              use_cache=False,
              cache=None):
        outputs = self.gpt(input_ids,
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=use_cache,
                           cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = self.parallel_matmul(
            encoder_outputs, self.gpt.embeddings.word_embeddings.weight, False,
            self.gpt.topo)
        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def forward(self,
                input_ids=None,
                max_length=20,
                min_length=0,
                decode_strategy='greedy_search',
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                num_beams=1,
                num_beam_groups=1,
                length_penalty=0.0,
                early_stopping=False,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                decoder_start_token_id=None,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                num_return_sequences=1,
                diversity_rate=0.0,
                use_cache=True,
                **model_kwargs):
        assert (
            decode_strategy in ["greedy_search", "sampling", "beam_search"]
        ), "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            decode_strategy)

        bos_token_id = bos_token_id if bos_token_id is not None else getattr(
            self.gpt, 'bos_token_id', None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(
            self.gpt, 'eos_token_id', None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(
            self.gpt, 'pad_token_id', None)
        forced_bos_token_id = forced_bos_token_id if forced_bos_token_id is not None else getattr(
            self.gpt, 'forced_bos_token_id', None)
        forced_eos_token_id = forced_eos_token_id if forced_eos_token_id is not None else getattr(
            self.gpt, 'forced_eos_token_id', None)
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else getattr(
            self.gpt, 'decoder_start_token_id', None)

        # params check
        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs[
                "attention_mask"] = self.prepare_attention_mask_for_generation(
                    input_ids, pad_token_id, eos_token_id)
        self.is_encoder_decoder = False

        model_kwargs["use_cache"] = use_cache
        max_length += input_ids.shape[-1]
        min_length += input_ids.shape[-1]

        logits_processors = self.get_logits_processor(
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty)

        if decode_strategy == 'sampling':
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs)

            return self.sample(input_ids, logits_processors, max_length,
                               pad_token_id, eos_token_id, top_k, top_p,
                               temperature, **model_kwargs)
        else:
            raise ValueError(f'Not support {decoding_strategy} strategy yet!')

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                max_dec_len=None,
                use_cache=False,
                cache=None):
        """
        Args:
            inputs (dict): include src_ids.
                pos_ids, input_mask and max_dec_len are optional.
        """
        if attention_mask is not None:
            tgt_pos = paddle.sum(attention_mask, axis=-1,
                                 keepdim=True).astype('int64')

        logits, cached_kvs = self.model(input_ids,
                                        position_ids,
                                        attention_mask,
                                        use_cache=True,
                                        cache=None)

        next_id = paddle.argmax(logits[:, -1, :], axis=-1).reshape([-1, 1])

        if max_dec_len is None:
            max_len = paddle.to_tensor(self.max_dec_len,
                                       dtype=int_type,
                                       place=paddle.CPUPlace())
        else:
            max_len = paddle.to_tensor(max_dec_len)

        min_len = paddle.to_tensor(self.min_dec_len,
                                   dtype=int_type,
                                   place=paddle.CPUPlace())

        step_idx = paddle.to_tensor(0, dtype=int_type, place=paddle.CPUPlace())

        placehold_ids = paddle.full_like(input_ids,
                                         fill_value=0,
                                         dtype=next_id.dtype)
        ids = paddle.tensor.array_write(next_id, step_idx)

        # if max_dec_len is not None:
        #     max_len = paddle.tensor.creation._memcpy(max_len,
        #                                              place=paddle.CPUPlace())
        cond = paddle.less_than(step_idx, max_len)

        if attention_mask is not None:
            append_mask = paddle.full_like(x=next_id,
                                           fill_value=1,
                                           dtype=attention_mask.dtype)
            append_mask = append_mask.reshape([-1, 1])

        while cond:
            pre_ids = paddle.tensor.array_read(array=ids, i=step_idx)
            if attention_mask is not None:
                att_mask = paddle.concat([attention_mask, append_mask], axis=-1)
                tgt_pos = tgt_pos + step_idx
            else:
                att_mask = None
                tgt_pos = None

            step_idx.add_(paddle.ones([1], dtype=int_type))
            paddle.tensor.array_write(placehold_ids, i=step_idx, array=ids)

            logits, cached_kvs = self.model(pre_ids,
                                            tgt_pos,
                                            att_mask,
                                            use_cache=True,
                                            cache=cached_kvs)

            logits = paddle.reshape(logits, shape=(-1, self.vocab_size))
            probs = F.softmax(logits / self.temperature)

            if self.decoding_strategy.startswith("sampling"):
                sampling_ids = layers.sampling_id(probs, dtype="int")
            elif self.decoding_strategy.startswith("topk_sampling"):
                probs, sampling_ids = self.topk_sampling(probs)
            elif self.decoding_strategy.startswith("topp_sampling"):
                probs, sampling_ids = self.topp_sampling(probs)
            else:
                raise ValueError(self.decoding_strategy)

            selected_ids = paddle.unsqueeze(sampling_ids, -1)
            paddle.tensor.array_write(selected_ids, i=step_idx, array=ids)

            cond = paddle.to_tensor((step_idx < max_len)
                                    and not paddle.is_empty(x=selected_ids),
                                    dtype='bool')

            if attention_mask is not None:
                attention_mask = att_mask

        ids, _ = layers.tensor_array_to_tensor(ids)
        return ids


GPTForCausalLM = GPTLMHeadModel
