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

import collections
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from paddle.distributed.fleet import fleet
import paddle.incubate as incubate

from paddlenlp.transformers import PretrainedModel, register_base_model
import paddlenlp

__all__ = [
    'GPTModel', 'GPTForPretraining', 'GPTPretrainingCriterion',
    'GPTForGeneration'
]

device = "gpu"
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

        if topo is None or topo.mp_info.size == 1:
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

        else:
            assert self.num_heads % topo.mp_info.size == 0
            self.num_heads = self.num_heads // topo.mp_info.size
            if self.fuse:
                assert self.kdim == embed_dim
                assert self.vdim == embed_dim
                self.qkv_proj = paddlenlp.ops.ColumnParallelLiner(
                    (embed_dim, 3 * embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attr,
                    bias_attr=bias_attr)
            else:
                self.q_proj = paddlenlp.ops.ColumnParallelLiner(
                    (embed_dim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attr,
                    bias_attr=bias_attr)
                self.k_proj = paddlenlp.ops.ColumnParallelLiner(
                    (self.kdim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attr,
                    bias_attr=bias_attr)
                self.v_proj = paddlenlp.ops.ColumnParallelLiner(
                    (self.vdim, embed_dim),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attr,
                    bias_attr=bias_attr)

            self.out_proj = paddlenlp.ops.RowParallelLiner(
                (embed_dim, embed_dim),
                topo.mp_info.size,
                input_is_parallel=True,
                param_attr=weight_attr,
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

            ## if not assign here, assign in While loop
            #layers.assign(k, cache.k)    # update caches
            #layers.assign(v, cache.v)

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

        if self.training:
            weights = incubate.softmax_mask_fuse_upper_triangle(product)
        else:
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
            self.norm = nn.LayerNorm(hidden_size)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")
        self.checkpoints = []

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                use_cache=False,
                cache=None,
                time_step=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []
        self.checkpoints = []

        if isinstance(self.layers, nn.LayerList):
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
                    if use_cache:
                        output, new_cache = mod(output,
                                                memory,
                                                tgt_mask=tgt_mask,
                                                use_cache=use_cache,
                                                cache=cache[i])
                        new_caches.append(new_cache)
                    else:
                        output = mod(output,
                                     memory,
                                     tgt_mask=tgt_mask,
                                     use_cache=use_cache,
                                     cache=cache[i])

                self.checkpoints.append(output.name)
        else:
            # fused_multi_transformer
            output = self.layers(output,
                                 attn_mask=tgt_mask,
                                 caches=cache,
                                 time_step=time_step)
            if cache:
                new_caches = output[1]
                output = output[0]

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

    Cache = collections.namedtuple("Cache", ["kv"])

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
                 topo=None,
                 **kwargs):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self._fuse = kwargs.get('fuse', False)
        if self._fuse:
            nranks, ring_id = 1, -1
            if topo is not None and topo.mp_info.size > 1:
                nranks = topo.mp_info.size
                ring_id = 0
            self.self_attn = incubate.nn.FusedMultiHeadAttention(
                d_model,
                nhead,
                dropout_rate=dropout,
                attn_dropout_rate=attn_dropout,
                normalize_before=normalize_before,
                qkv_weight_attr=weight_attrs[0],
                qkv_bias_attr=bias_attrs[0],
                linear_weight_attr=weight_attrs[0],
                linear_bias_attr=bias_attrs[0],
                epsilon=1e-5,
                nranks=nranks,
                ring_id=ring_id)
            self.ffn = incubate.nn.FusedFeedForward(
                d_model,
                dim_feedforward,
                dropout_rate=act_dropout,
                epsilon=1e-5,
                activation=activation,
                normalize_before=normalize_before,
                act_dropout_rate=0.0,
                linear1_weight_attr=weight_attrs[2],
                linear1_bias_attr=bias_attrs[2],
                linear2_weight_attr=weight_attrs[2],
                linear2_bias_attr=bias_attrs[2],
                nranks=nranks,
                ring_id=ring_id)
        else:
            self.self_attn = MultiHeadAttention(d_model,
                                                nhead,
                                                dropout=attn_dropout,
                                                weight_attr=weight_attrs[0],
                                                bias_attr=bias_attrs[0],
                                                topo=topo)
            if topo is None or topo.mp_info.size == 1:
                self.linear1 = nn.Linear(d_model,
                                         dim_feedforward,
                                         weight_attrs[2],
                                         bias_attr=bias_attrs[2])
                self.linear2 = nn.Linear(dim_feedforward,
                                         d_model,
                                         weight_attrs[2],
                                         bias_attr=bias_attrs[2])
            else:
                self.linear1 = paddlenlp.ops.ColumnParallelLiner(
                    (d_model, dim_feedforward),
                    topo.mp_info.size,
                    gather_out=False,
                    param_attr=weight_attrs[2],
                    bias_attr=bias_attrs[2])
                self.linear2 = paddlenlp.ops.RowParallelLiner(
                    (dim_feedforward, d_model),
                    topo.mp_info.size,
                    input_is_parallel=True,
                    param_attr=weight_attrs[2],
                    bias_attr=bias_attrs[2])

            self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
            self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
            self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
            self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
            self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None):
        if self._fuse:
            if isinstance(cache, self.Cache):
                attn_output, cache_kv_out = self.self_attn(tgt,
                                                           attn_mask=tgt_mask,
                                                           cache=cache.kv)

                ## if not assign here, update caches in While loop
                # layers.assign(cache_kv_out, cache.kv)
                if use_cache:
                    cache = self.Cache(cache_kv_out)
            else:
                attn_output = self.self_attn(tgt, attn_mask=tgt_mask)

            enc_out = self.ffn(attn_output)
            return (enc_out, cache) if use_cache else enc_out

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
    Include embeddings from word, position and token_type embeddings
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
        if topo is None or topo.mp_info.size == 1:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                hidden_size,
                weight_attr=paddle.ParamAttr(name="word_embeddings",
                                             initializer=nn.initializer.Normal(
                                                 mean=0.0,
                                                 std=initializer_range)))
        else:
            self.word_embeddings = paddlenlp.ops.ParallelEmbedding(
                vocab_size,
                hidden_size,
                topo.mp_info.rank,
                topo.mp_info.size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range)))
        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(name="pos_embeddings",
                                         initializer=nn.initializer.Normal(
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
    loading pretrained models. See `PretrainedModel` for more details.
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
        "gpt3-89B-en": { # 89B
            "vocab_size": 51200,
            "hidden_size": 12288,
            "num_hidden_layers": 48,
            "num_attention_heads": 96,
            "intermediate_size": 49152,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
        },
        "gpt3-175B-en": { # 175B
            "vocab_size": 51200,
            "hidden_size": 12288,
            "num_hidden_layers": 96,
            "num_attention_heads": 96,
            "intermediate_size": 49152,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
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
            "vocab_size": 50304,
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
            "gpt2-medium-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-medium-en.pdparams",
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
    """
    The base model of gpt.
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
                 topo=None,
                 **kwargs):
        super(GPTModel, self).__init__()

        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.topo = topo
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        self.pipline_mode = topo is not None and topo.pp_info.size > 1
        if self.pipline_mode:
            self.layer_per_stage = num_hidden_layers // self.topo.pp_info.size

        self.embeddings = GPTEmbeddings(vocab_size, hidden_size,
                                        hidden_dropout_prob,
                                        max_position_embeddings,
                                        type_vocab_size, self.initializer_range,
                                        topo)

        if kwargs.get('fuse_mt', False):
            nranks, ring_id = 1, -1
            if topo is not None and topo.mp_info.size > 1:
                nranks = topo.mp_info.size
                ring_id = 0

            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=self.initializer_range))
            bias_attr = None
            decoder_layers = incubate.nn.FusedMultiTransformer(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                dropout_rate=hidden_dropout_prob,
                activation=hidden_act,
                qkv_weight_attrs=_convert_param_attr_to_list(
                    weight_attr, num_hidden_layers),
                qkv_bias_attrs=_convert_param_attr_to_list(
                    bias_attr, num_hidden_layers),
                linear_weight_attrs=_convert_param_attr_to_list(
                    weight_attr, num_hidden_layers),
                linear_bias_attrs=_convert_param_attr_to_list(
                    bias_attr, num_hidden_layers),
                ffn1_weight_attrs=_convert_param_attr_to_list(
                    weight_attr, num_hidden_layers),
                ffn1_bias_attrs=_convert_param_attr_to_list(
                    bias_attr, num_hidden_layers),
                ffn2_weight_attrs=_convert_param_attr_to_list(
                    weight_attr, num_hidden_layers),
                ffn2_bias_attrs=_convert_param_attr_to_list(
                    bias_attr, num_hidden_layers),
                epsilon=1e-5,
                nranks=nranks,
                ring_id=ring_id)
        else:
            decoder_layers = nn.LayerList()
            for i in range(num_hidden_layers):
                DecoderLayer = TransformerDecoderLayer
                if self.pipline_mode:
                    DecoderLayer = paddlenlp.ops.guard('gpu:{}'.format(
                        i // self.layer_per_stage))(TransformerDecoderLayer)
                decoder_layers.append(
                    DecoderLayer(d_model=hidden_size,
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
                                 topo=topo,
                                 fuse=kwargs.get('fuse', False)))

        if self.pipline_mode:
            Decoder = paddlenlp.ops.guard(
                'gpu:{}'.format(self.topo.pp_info.size - 1))(TransformerDecoder)
        else:
            Decoder = TransformerDecoder

        self.decoder = Decoder(decoder_layers,
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
                cache=None,
                time_step=None):
        self.checkpoints = []
        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(past_length,
                                         paddle.shape(input_ids)[-1] +
                                         past_length,
                                         dtype='int64')
            position_ids = position_ids.unsqueeze(0)
            position_ids = paddle.fluid.layers.expand_as(
                position_ids, input_ids)
        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids)

        tgt_mask = None
        if not self.training:
            tgt_mask = attention_mask
        encoder_outputs = self.decoder(embedding_output,
                                       memory=None,
                                       tgt_mask=tgt_mask,
                                       use_cache=use_cache,
                                       cache=cache,
                                       time_step=time_step)
        self.checkpoints.extend(self.decoder.checkpoints)
        return encoder_outputs


class GPTForPretraining(GPTPretrainedModel):
    """
    The pretraining model of GPT.

    It returns some logits and cached_kvs.
    """

    def __init__(self, gpt):
        super(GPTForPretraining, self).__init__()
        self.gpt = gpt
        self.apply(self.init_weights)

    def parallel_matmul(self, lm_output, logit_weights, parallel_output, topo):
        if topo is not None and topo.mp_info.size > 1:
            input_parallel = paddle.distributed.collective._c_identity(
                lm_output, group=None)

            logits = paddle.matmul(input_parallel,
                                   logit_weights,
                                   transpose_y=True)

            if parallel_output:
                return logits

            return paddle.distributed.collective._c_concat(logits, group=None)
        else:
            logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
            return logits

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                use_cache=False,
                cache=None,
                time_step=None):
        outputs = self.gpt(input_ids,
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=use_cache,
                           cache=cache,
                           time_step=time_step)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = self.parallel_matmul(
            encoder_outputs, self.gpt.embeddings.word_embeddings.weight, True,
            self.gpt.topo)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT.

    It calculates the final loss.
    """

    def __init__(self, topo=None):
        super(GPTPretrainingCriterion, self).__init__()
        if topo is None or topo.mp_info.size == 1:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss_func = paddle.distributed.collective._c_softmax_with_cross_entropy

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        masked_lm_loss = self.loss_func(prediction_scores,
                                        masked_lm_labels.unsqueeze(2))

        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        loss = masked_lm_loss / loss_mask.sum()
        return loss


class GPTForGeneration(GPTPretrainedModel):

    def __init__(self,
                 gpt,
                 max_length=20,
                 min_length=0,
                 decoding_strategy='sampling',
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 eos_id=None,
                 **kwargs):
        super(GPTForGeneration, self).__init__()
        self.gpt = gpt
        self.apply(self.init_weights)
        self.vocab_size = gpt.vocab_size
        self.eos_token_id = eos_id or 7

        self.min_dec_len = min_length
        self.max_dec_len = max_length
        self.decoding_strategy = decoding_strategy
        self.temperature = temperature
        self.topk = top_k
        self.topp = top_p
        self._init_gen_cache = False
        self.generation_caches = None
        # for fused_multi_transformer
        self.generation_time_step = None
        self._dtype = "float32"
        self._fuse = kwargs.get("fuse", False)
        self._fuse_mt = kwargs.get("fuse_mt", False)

    def _init_generation_caches(self, src_ids):
        if self._fuse and self._fuse_mt:
            # output tensor is on CPUPlace
            self.generation_time_step = paddle.shape(src_ids)[1]

        # not fuse, return None
        if self._init_gen_cache or self._fuse is False:
            return self.generation_caches

        self.generation_caches = []
        num_heads = self.gpt.num_attention_heads
        num_layers = self.gpt.num_hidden_layers
        mp_n_head = num_heads // self.gpt.topo.mp_info.size
        hidden_size = self.gpt.hidden_size
        head_size = hidden_size // num_heads
        seq_len = 0
        if self._fuse_mt:
            # FIXME(wangxi): dynamic get max_seq_len + dec_len
            seq_len = 1024
        for i in range(num_layers):
            if self._fuse:
                kv = layers.fill_constant_batch_size_like(
                    input=src_ids,
                    shape=[2, -1, mp_n_head, seq_len, head_size],
                    dtype=self._dtype,
                    value=0,
                    output_dim_idx=1)
                if self._fuse_mt:
                    self.generation_caches.append(kv)
                else:
                    self.generation_caches.append(
                        TransformerDecoderLayer.Cache(kv))
            else:
                k = layers.fill_constant_batch_size_like(
                    input=src_ids,
                    shape=[-1, mp_n_head, 0, head_size],
                    dtype=self._dtype,
                    value=0)
                v = layers.fill_constant_batch_size_like(
                    input=src_ids,
                    shape=[-1, mp_n_head, 0, head_size],
                    dtype=self._dtype,
                    value=0)
                self.generation_caches.append(MultiHeadAttention.Cache(k, v))
        self._init_gen_cache = True
        return self.generation_caches

    def parallel_matmul(self, lm_output, logit_weights, parallel_output, topo):
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
        sampling_ids = layers.sampling_id(probs, dtype="int")
        probs = old_probs
        return probs, sampling_ids

    def topp_sampling(self, probs):
        sorted_probs, sorted_idx = layers.argsort(probs, descending=True)
        cum_sorted_probs = layers.cumsum(sorted_probs, axis=1, exclusive=True)
        lt_cond = paddle.cast(
            paddle.less_than(
                cum_sorted_probs,
                layers.fill_constant_batch_size_like(cum_sorted_probs,
                                                     cum_sorted_probs.shape,
                                                     cum_sorted_probs.dtype,
                                                     self.topp)), "float32")
        old_probs = probs
        candidate_probs = sorted_probs * lt_cond
        probs = candidate_probs / paddle.sum(
            candidate_probs, axis=-1, keep_dim=True)
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
              cache=None,
              time_step=None):
        outputs = self.gpt(input_ids,
                           position_ids=position_ids,
                           attention_mask=attention_mask,
                           use_cache=use_cache,
                           cache=cache,
                           time_step=time_step)
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

    def forward(self, inputs, use_cache=False, cache=None):
        """
        Args:
            inputs (dict): include src_ids.
                pos_ids, input_mask and max_dec_len are optional.
        """
        ######### forward context #########
        input_ids = inputs['src_ids']
        position_ids = inputs['pos_ids'] if 'pos_ids' in inputs else None
        attention_mask = inputs['input_mask'] if 'input_mask' in inputs else None

        causal_mask = paddle.tensor.triu(paddle.ones(
            (paddle.shape(input_ids)[-1], paddle.shape(input_ids)[-1])) * -1e4,
                                         diagonal=1)
        if attention_mask is not None:
            tgt_pos = paddle.sum(attention_mask, axis=-1,
                                 keepdim=True).astype('int64')
            if len(attention_mask.shape) == 2:
                attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2])
            encode_mask = attention_mask + causal_mask
        else:
            encode_mask = causal_mask

        # if cached_kvs are assigned to next step in _prepare_qkv of MultiHeadAttention,
        # need to init the global caches here
        gen_caches = self._init_generation_caches(input_ids)

        logits, cached_kvs = self.model(input_ids,
                                        position_ids,
                                        encode_mask,
                                        use_cache=True,
                                        cache=gen_caches)

        next_id = paddle.argmax(logits[:, -1, :], axis=-1).reshape([-1, 1])
        ####################################

        if 'max_dec_len' not in inputs:
            max_len = layers.fill_constant([1],
                                           dtype=int_type,
                                           value=self.max_dec_len,
                                           force_cpu=True)
        else:
            max_len = inputs['max_dec_len']
        min_len = layers.fill_constant(shape=[1],
                                       dtype=int_type,
                                       value=self.min_dec_len,
                                       force_cpu=True)
        step_idx = layers.fill_constant(shape=[1],
                                        value=0,
                                        dtype='int64',
                                        force_cpu=True)

        placehold_ids = layers.fill_constant_batch_size_like(
            input=inputs["src_ids"],
            value=0,
            shape=[-1, 1],
            dtype=next_id.dtype)
        ids = layers.array_write(next_id, step_idx)

        if 'max_dec_len' in inputs:
            max_len = paddle.tensor.creation._memcpy(max_len,
                                                     place=paddle.CPUPlace())
        cond_int = paddle.full([1], 0, dtype=int_type, name="cond_int")
        cond = paddle.less_than(step_idx, max_len)

        if attention_mask is not None:
            append_mask = layers.fill_constant_batch_size_like(
                input=next_id,
                value=1,
                shape=[-1, 1, 1, 1],
                dtype=attention_mask.dtype)

        while_op = layers.While(cond, is_test=True)
        with while_op.block():
            pre_ids = layers.array_read(array=ids, i=step_idx)
            if attention_mask:
                decode_mask = paddle.concat([attention_mask, append_mask],
                                            axis=-1)
                tgt_pos = tgt_pos + step_idx
                att_mask = (1 - decode_mask) * -1e4
            else:
                att_mask = None
                tgt_pos = None

            layers.increment(x=step_idx, value=1.0, in_place=True)
            layers.array_write(placehold_ids, i=step_idx, array=ids)

            logits, decode_cached_kvs = self.model(
                pre_ids,
                tgt_pos,
                att_mask,
                use_cache=True,
                cache=cached_kvs,
                time_step=self.generation_time_step)

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
            layers.array_write(selected_ids, i=step_idx, array=ids)

            length_cond = paddle.less_than(x=step_idx,
                                           y=max_len,
                                           name="length_cond")
            finish_cond = paddle.logical_not(paddle.is_empty(x=selected_ids),
                                             name="finish_cond")
            paddle.logical_and(x=length_cond,
                               y=finish_cond,
                               out=cond,
                               name="logical_and_cond")

            paddle.assign(layers.cast(cond, dtype='bool'), cond)
            if attention_mask:
                paddle.assign(decode_mask, attention_mask)
            for i in range(len(decode_cached_kvs)):
                if self._fuse:
                    if not self._fuse_mt:
                        paddle.assign(decode_cached_kvs[i].kv, cached_kvs[i].kv)
                else:
                    paddle.assign(decode_cached_kvs[i].k, cached_kvs[i].k)
                    paddle.assign(decode_cached_kvs[i].v, cached_kvs[i].v)
            if self.generation_time_step:
                paddle.increment(self.generation_time_step, value=1.0)

        ids, _ = layers.tensor_array_to_tensor(ids)
        return ids
