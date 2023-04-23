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
import os

import paddle
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.common_ops_import import convert_dtype
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
    get_rng_state_tracker,
)
from paddle.distributed.fleet.utils import recompute
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list

import paddlenlp
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.generation_utils import LogitsProcessorList
from paddlenlp.transformers.model_outputs import (
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from paddlenlp.utils.log import logger

__all__ = [
    "GPTModel",
    "GPTPretrainedModel",
    "GPTForPretraining",
    "GPTPretrainingCriterion",
    "GPTForGreedyGeneration",
    "GPTLMHeadModel",
]


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    # rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


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
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
        weight_attr=None,
        bias_attr=None,
        fuse_attention_qkv=True,
        num_partitions=1,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse_attention_qkv = fuse_attention_qkv

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert self.num_heads % num_partitions == 0
        self.num_heads = self.num_heads // num_partitions

        if self.fuse_attention_qkv:
            assert self.kdim == embed_dim, "embed_dim should be equal to kdim"
            assert self.vdim == embed_dim, "embed_dim should be equal to vidm"

            self.qkv_proj = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim, 3 * embed_dim, weight_attr=weight_attr, has_bias=True, gather_output=False
            )
        else:
            self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim, embed_dim, weight_attr=weight_attr, has_bias=True, gather_output=False
            )

            self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.kdim, embed_dim, weight_attr=weight_attr, has_bias=True, gather_output=False
            )

            self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.vdim, embed_dim, weight_attr=weight_attr, has_bias=True, gather_output=False
            )

        self.out_proj = fleet.meta_parallel.RowParallelLinear(
            embed_dim, embed_dim, weight_attr=weight_attr, has_bias=True, input_is_parallel=True
        )

    def _fuse_prepare_qkv(self, query):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, self.num_heads, 3 * self.head_dim])
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
                input=key, shape=[-1, self.num_heads, 0, self.head_dim], dtype=key.dtype, value=0
            )
            v = layers.fill_constant_batch_size_like(
                input=key, shape=[-1, self.num_heads, 0, self.head_dim], dtype=key.dtype, value=0
            )
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
        # compute q ,k ,v
        if use_cache is False:
            if self.fuse_attention_qkv:
                q, k, v = self._fuse_prepare_qkv(query)
            else:
                q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache, cache)
        # scale dot product attention
        product = paddle.matmul(x=q * (self.head_dim**-0.5), y=k, transpose_y=True)

        softmax_mask_fuse_upper_triangle = strtobool(os.getenv("softmax_mask_fuse_upper_triangle", True))
        if softmax_mask_fuse_upper_triangle:
            weights = incubate.softmax_mask_fuse_upper_triangle(product)
        else:
            if attn_mask is not None:
                product = product + attn_mask
            weights = F.softmax(product)

        if self.dropout:
            with get_rng_state_tracker().rng_state("local_seed"):
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


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(self, decoder_layers, num_layers, norm=None, hidden_size=None, use_recompute=False):
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        self.use_recompute = use_recompute
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")
        self.checkpoints = []

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, use_cache=False, cache=None):
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
                    output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache)
                    new_caches.append(new_cache)
                else:
                    # TODO(shenliang03) support random state for mp
                    output = (
                        recompute(mod, output, memory, tgt_mask, use_cache, cache)
                        if self.use_recompute
                        else mod(output, memory, tgt_mask, use_cache, cache)
                    )
            else:
                output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache[i])
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

    def __init__(
        self,
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
        num_partitions=1,
        enable_fuse_transformer=False,
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()

        self.enable_fuse_transformer = enable_fuse_transformer
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        if self.enable_fuse_transformer:
            hcg = fleet.get_hybrid_communicate_group()
            mp_nranks = hcg.get_model_parallel_world_size()
            mp_group = hcg.get_model_parallel_group()
            ring_id = mp_group.id if mp_nranks > 1 else -1
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
                nranks=mp_nranks,
                ring_id=ring_id,
            )
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
                nranks=mp_nranks,
                ring_id=ring_id,
            )
        else:
            self.self_attn = MultiHeadAttention(
                d_model,
                nhead,
                dropout=attn_dropout,
                weight_attr=weight_attrs[0],
                bias_attr=bias_attrs[0],
                num_partitions=num_partitions,
            )

            self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
                d_model, dim_feedforward, weight_attr=weight_attrs[2], gather_output=False, has_bias=True
            )

            self.linear2 = fleet.meta_parallel.RowParallelLinear(
                dim_feedforward, d_model, weight_attr=weight_attrs[2], input_is_parallel=True, has_bias=True
            )

            self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
            self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
            self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
            self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
            self.activation = getattr(F, activation)

    def forward(self, tgt, memory=None, tgt_mask=None, use_cache=False, cache=None):
        if self.enable_fuse_transformer:
            if use_cache:
                attn_output, cache_kv_out = self.self_attn(tgt, attn_mask=tgt_mask, cache=cache.kv)
            else:
                attn_output = self.self_attn(tgt, attn_mask=tgt_mask)

            enc_out = self.ffn(attn_output)
            return (enc_out, cache_kv_out) if use_cache else enc_out

        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)

        with get_rng_state_tracker().rng_state("global_seed"):
            tgt = residual + self.dropout1(tgt)

        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)

        with get_rng_state_tracker().rng_state("global_seed"):
            tgt = self.dropout2(self.linear2(F.gelu(self.linear1(tgt), approximate=True)))

        tgt = residual + tgt

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        return tgt if use_cache is False else (tgt, incremental_cache)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        return incremental_cache


class GPTEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
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

        self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                name="pos_embeddings", initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
            ),
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings

        # with get_rng_state_tracker().rng_state('global_seed'):
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
        "gpt-cpm-large-cn": {  # 2.6B
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
            "num_partitions": 1,
            "use_recompute": False,
        },
        "gpt-cpm-small-cn-distill": {  # 109M
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
            "num_partitions": 1,
            "use_recompute": False,
        },
        "gpt3-89B-en": {  # 89B
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
        "gpt3-175B-en": {  # 175B
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
        "gpt3-13B-en": {  # 13B
            "vocab_size": 50304,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "intermediate_size": 20480,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 2048,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "eos_token_id": 50256,
            "eol_token_id": 198,
            "num_partitions": 1,
            "use_recompute": False,
        },
        "gpt3-1.3B-en": {  # 1.3B
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
            "num_partitions": 1,
            "use_recompute": False,
        },
        "gpt2-medium-en": {  # 345M
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
            "num_partitions": 1,
            "use_recompute": False,
        },
        "gpt2-en": {  # 117M
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
            "num_partitions": 1,
            "use_recompute": False,
        },
        "gpt2-small-en": {  # config for CE
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
            "num_partitions": 1,
            "use_recompute": False,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "gpt-cpm-large-cn": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt-cpm-large-cn.pdparams",
            "gpt-cpm-small-cn-distill": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt-cpm-small-cn-distill.pdparams",
            "gpt2-medium-en": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-medium-en.pdparams",
        }
    }
    base_model_prefix = "gpt"

    def init_weights(self, layer):
        """Initialization hook"""
        # no hook
        return
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
    """
    The base model of gpt.
    """

    def __init__(
        self,
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
        num_partitions=1,
        use_recompute=False,
        enable_fuse_transformer=False,
    ):
        super(GPTModel, self).__init__()

        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.enable_fuse_transformer = enable_fuse_transformer

        self.embeddings = GPTEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            self.initializer_range,
        )

        self.bias = paddle.tril(paddle.ones([1, 1, max_position_embeddings, max_position_embeddings], dtype="int64"))
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
                        initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range)
                    ),
                    bias_attr=None,
                    num_partitions=num_partitions,
                    enable_fuse_transformer=self.enable_fuse_transformer,
                )
            )

        self.decoder = TransformerDecoder(
            decoder_layers,
            num_hidden_layers,
            norm="LayerNorm",
            hidden_size=hidden_size,
            use_recompute=use_recompute,
        )

        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(self, input_ids, position_ids=None, attention_mask=None, use_cache=False, cache=None):
        self.checkpoints = []
        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(past_length, paddle.shape(input_ids)[-1] + past_length, dtype="int64")
            position_ids = position_ids.unsqueeze(0)
            # .expand_as(input_ids)
            position_ids = paddle.expand_as(position_ids, input_ids)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        softmax_mask_fuse_upper_triangle = strtobool(os.getenv("softmax_mask_fuse_upper_triangle", True))

        if self.enable_fuse_transformer or not softmax_mask_fuse_upper_triangle:
            length = paddle.shape(input_ids)[-1]
            # TODO, use registered buffer
            causal_mask = self.bias[:, :, 0:length, :length]

            if attention_mask is not None:
                if attention_mask.dtype != paddle.int64:
                    attention_mask = paddle.cast(attention_mask, dtype=paddle.int64)
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - (attention_mask & causal_mask)) * -1e4
            else:
                attention_mask = (1.0 - causal_mask) * -1e4
            # The tensor returned by triu not in static graph.

        # The tensor returned by triu not in static graph.
        # attention_mask.stop_gradient = True

        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            # tgt_mask=None,
            use_cache=use_cache,
            cache=cache,
        )
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
        # extra_parameters using for sharding stage3 to register extra_parameters
        # TODO(Baibaifan): add additional extra parameters mode of semi-automatic registration later
        self.extra_parameters = [self.gpt.embeddings.word_embeddings.weight]

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
        use_cache=False,
        cache=None,
    ):
        outputs = self.gpt(
            input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = parallel_matmul(encoder_outputs, self.gpt.embeddings.word_embeddings.weight, True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT.

    It calculates the final loss.
    """

    def __init__(self, pad_token_id=None):
        super(GPTPretrainingCriterion, self).__init__()
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        self.parallel_loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        hcg = fleet.get_hybrid_communicate_group()
        self.mp_size = hcg.get_model_parallel_world_size()
        self.pad_token_id = pad_token_id

    def forward(self, prediction_scores, masked_lm_labels, loss_mask=None):
        if self.mp_size > 1:
            masked_lm_loss = self.parallel_loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))
        else:
            masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))

        with paddle.amp.auto_cast(False):
            masked_lm_loss = masked_lm_loss.astype("float32")
            if loss_mask is not None:
                loss_mask = loss_mask.reshape([-1])
                masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
                loss = masked_lm_loss / loss_mask.sum()
            else:
                assert self.pad_token_id is not None
                masked_lm_loss = masked_lm_loss[masked_lm_labels != self.pad_token_id]
                loss = paddle.mean(masked_lm_loss)

        return loss


class GPTForGreedyGeneration(GPTPretrainedModel):
    """
    The generate model for GPT-2.
    It use the greedy stategy and generate the next word with highest probablity.
    """

    def __init__(self, gpt, max_predict_len):
        super(GPTForGreedyGeneration, self).__init__()
        self.gpt = gpt
        self.max_predict_len = paddle.to_tensor(max_predict_len, dtype="int32")
        self.apply(self.init_weights)

    def model(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
        use_cache=False,
        cache=None,
    ):
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

    def forward(self, input_ids, end_id):
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
            if paddle.max(nid) == end_id:
                break
        return src_ids


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

    def __init__(self, gpt, num_labels):
        super(GPTForSequenceClassification, self).__init__()

        # self.gpt = GPTModel(config)  # allow gpt to be config
        self.gpt = gpt
        self.score = nn.Linear(gpt.config["hidden_size"], num_labels, bias_attr=False)
        self.apply(self.init_weights)
        self.num_classes = num_labels
        # self.extra_parameters = [self.gpt.embeddings.word_embeddings.weight]

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
        input_type = type(input_ids) if input_ids is not None else type(inputs_embeds)
        # sequence_output shape [bs, seq_len, hidden_size]
        sequence_output = self.gpt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            # inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        if isinstance(sequence_output, input_type):
            hidden_states = sequence_output
        else:
            hidden_states = sequence_output[0]
        # logits shape [bs, seq_len, num_class]
        logits = self.score(hidden_states)
        # padding index maybe 0
        eos_token_id = self.gpt.config["eos_token_id"] or 0
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
            if self.num_classes == 1:
                loss_fct = paddle.nn.MSELoss()
                loss = loss_fct(pooled_logits, labels)
            elif labels.dtype == paddle.int64 or labels.dtype == paddle.int32:
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape((-1, self.num_classes)), labels.reshape((-1,)))
            else:
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            if isinstance(sequence_output, input_type):
                return (loss, pooled_logits) if loss is not None else pooled_logits

            outputs = (pooled_logits,) + sequence_output[1:]
            return ((loss,) + outputs) if loss is not None else outputs

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            # past_key_values=sequence_output.past_key_values,
            # hidden_states=sequence_output.hidden_states,
            # attentions=sequence_output.attentions,
        )


# these Layers is just for PipelineParallel


class GPTPretrainingCriterionPipe(GPTPretrainingCriterion):
    """Extends GPTPretrainingCriterion to meet the input standard."""

    def forward(self, prediction_scores, args):
        masked_lm_labels = args[0]
        loss_mask = args[1]
        loss = super().forward(prediction_scores, masked_lm_labels, loss_mask)
        return loss


class EmbeddingPipe(GPTEmbeddings):
    """Extends GPTEmbeddings to forward attention_mask through the pipeline."""

    @property
    def embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, tensors):
        input_ids, position_ids = tensors
        embeddings = super().forward(input_ids=input_ids, position_ids=position_ids)
        return embeddings


class GPTForPretrainingPipe(PipelineLayer):
    """GPTForPretraining adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    def __init__(
        self,
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
        num_partitions=1,
        topology=None,
        use_recompute=False,
        enable_fuse_transformer=False,
    ):

        # forward desc
        self.descs = []

        self.descs.append(
            SharedLayerDesc(
                "embed",
                EmbeddingPipe,
                shared_weight_attr="embedding_weight",
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                hidden_dropout_prob=hidden_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=type_vocab_size,
                initializer_range=initializer_range,
            )
        )

        for _ in range(num_hidden_layers):
            self.descs.append(
                LayerDesc(
                    TransformerDecoderLayer,
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    dropout=hidden_dropout_prob,
                    activation=hidden_act,
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
                    bias_attr=None,
                    num_partitions=num_partitions,
                    enable_fuse_transformer=enable_fuse_transformer,
                )
            )

        self.descs.append(LayerDesc(nn.LayerNorm, normalized_shape=hidden_size))

        def _logits_helper(embedding, output):
            return parallel_matmul(output, embedding.embedding_weight, True)

        self.descs.append(
            SharedLayerDesc(
                "embed",
                EmbeddingPipe,
                forward_func=_logits_helper,
                shared_weight_attr="embedding_weight",
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                hidden_dropout_prob=hidden_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                type_vocab_size=type_vocab_size,
                initializer_range=initializer_range,
            )
        )

        super().__init__(
            layers=self.descs,
            loss_fn=GPTPretrainingCriterionPipe(),
            topology=topology,
            seg_method="layer:TransformerDecoderLayer",
            recompute_interval=1 if use_recompute else 0,
            recompute_ctx={
                "mp_group": fleet.fleet._hcg.get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
        )


class GPTForGeneration(paddlenlp.transformers.GPTPretrainedModel):
    """
    GPT Model with pretraining tasks on top.
    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.
    """

    def __init__(self, config):
        super(GPTForGeneration, self).__init__(config)
        # super(GPTForGeneration, self).__init__()
        self.gpt = paddlenlp.transformers.GPTModel(config)
        self.config = config
        self.max_length = self.config.get("max_dec_len", 20)
        self.min_length = self.config.get("min_dec_len", 0)
        self.decode_strategy = self.config.get("decode_strategy", "sampling")
        self.temperature = self.config.get("temperature", 1.0)
        self.top_k = self.config.get("top_k", 1)
        self.top_p = self.config.get("top_p", 1.0)
        # self.use_topp_sampling = self.config.get('use_topp_sampling', False)
        self.inference = self.config.get("inference", False)
        self.repetition_penalty = self.config.get("repetition_penalty", 1.0)
        self.num_beams = self.config.get("num_beams", 1)
        self.num_beam_groups = self.config.get("num_blength_penaltyeam_groups", 1)
        self.length_penalty = self.config.get("", 0.0)
        self.early_stopping = self.config.get("early_stopping", False)
        self.bos_token_id = self.config.get("bos_token_id", None)
        self.eos_token_id = self.config.get("eos_token_id", None)
        self.pad_token_id = self.config.get("pad_token_id", None)
        self.decoder_start_token_id = self.config.get("decoder_start_token_id", None)
        self.forced_bos_token_id = self.config.get("forced_bos_token_id", None)
        self.forced_eos_token_id = self.config.get("forced_eos_token_id", None)
        self.num_return_sequences = self.config.get("num_return_sequences", 1)
        self.diversity_rate = self.config.get("diversity_rate", 0.0)
        self.use_cache = self.config.get("use_cache", True)

    def prepare_input_ids_for_generation(self, bos_token_id, encoder_output=None):
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    def update_scores_for_generation(self, scores, next_scores, length, unfinished_flag):
        unfinished_scores = (scores * length + next_scores) / (length + 1)
        scores = paddle.where(unfinished_flag, unfinished_scores, scores)
        return scores

    def get_logits_processor(
        self,
        min_length=None,
        max_length=None,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_rate=0.0,
        repetition_penalty=None,
    ):
        processors = LogitsProcessorList()

        # FIXME
        # if min_length is not None and eos_token_id is not None and min_length > -1:
        #     print(min_length, eos_token_id)
        #     processors.append(
        #         MinLengthLogitsProcessor(min_length, eos_token_id))

        # if num_beam_groups > 1 and diversity_rate > 0.0:
        #     processors.append(
        #         HammingDiversityLogitsProcessor(
        #             diversity_rate=diversity_rate, num_beams=num_beams, num_beam_groups=num_beam_groups
        #         )
        #     )
        # if repetition_penalty is not None and repetition_penalty != 1.0:
        #     processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        # if forced_bos_token_id is not None:
        #     processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        # if forced_eos_token_id is not None:
        #     processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))

        # TODO
        # Add more pre_processing for distribution

        return processors

    def expand_inputs_for_generation(self, input_ids, expand_size, attention_mask=None, **model_kwargs):

        index = paddle.tile(paddle.arange(paddle.shape(input_ids)[0]).unsqueeze(-1), [1, expand_size]).reshape([-1])

        input_ids = paddle.gather(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.gather(attention_mask, index)

        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.gather(token_type_ids, index)

        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.gather(position_ids, index)

        if "seq_len" in model_kwargs and model_kwargs["seq_len"] is not None:
            seq_len = model_kwargs["seq_len"]
            model_kwargs["seq_len"] = paddle.gather(seq_len, index)

        if "encoder_output" in model_kwargs and model_kwargs["encoder_output"] is not None:
            encoder_output = model_kwargs["encoder_output"]
            model_kwargs["encoder_output"] = paddle.gather(encoder_output, index)

        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.gather(role_ids, index)

        return input_ids, model_kwargs

    # def prepare_inputs_for_generation(self,
    #                                   input_ids,
    #                                   use_cache=False,
    #                                   cache=None,
    #                                   **kwargs):
    #     # only last token for inputs_ids if cache is defined in kwargs
    #     position_ids = kwargs.get("position_ids", None)
    #     attention_mask = kwargs.get("attention_mask", None)
    #     if attention_mask is not None:
    #         if len(attention_mask.shape) == 4:
    #             attention_mask = attention_mask[:, -1, -1, :]
    #         if "int" in paddle.common_ops_import.convert_dtype(
    #                 attention_mask.dtype):
    #             attention_mask = (1.0 - attention_mask) * -1e4
    #     return {
    #         "input_ids": input_ids,
    #         "position_ids": position_ids,
    #         "attention_mask": attention_mask,
    #         "cache": cache
    #     }

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
            # "use_cache": use_cache,
            "cache": cache,
        }

    def update_model_kwargs_for_generation(self, next_tokens, outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache

        # print("Debug", outputs)

        if isinstance(outputs, tuple):
            model_kwargs["cache"] = outputs[1]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = position_ids[:, -1:] + 1

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == "bool":
                attention_mask = paddle.cast(attention_mask, "int64")
            if len(attention_mask.shape) == 4:
                attention_mask = nn.Pad2D([0, 0, 0, 1], mode="replicate")(attention_mask)
                attention_mask = nn.Pad2D([0, 1, 0, 0], value=-1e4)(attention_mask)
                dtype = convert_dtype(attention_mask.dtype)
                if "int" in dtype:
                    attention_mask[:, :, -1, -1] = 1
                elif "float" in dtype:
                    attention_mask[:, :, -1, -1] = 0.0
                else:
                    raise ValueError("The data type of input `attention_mask` must " "be bool, int or float")
            else:
                attention_mask = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="int64")], axis=-1
                )
            model_kwargs["attention_mask"] = attention_mask

        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        model_kwargs["res"] = paddle.concat([model_kwargs["res"], next_tokens], axis=1)

        return model_kwargs

    def sample(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs
    ):
        def TopKProcess(probs, top_k, min_tokens_to_keep):
            top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
            # Remove all tokens with a probability less than the last token of the top-k
            topk_probs, _ = paddle.topk(probs, k=top_k)
            probs = paddle.where(probs >= topk_probs[:, -1:], probs, paddle.full_like(probs, 0.0))
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
                sorted_indices_to_remove[:, : min_tokens_to_keep - 1] = 0
            # Keep the first token
            sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            # Scatter sorted tensors to original indexing
            sorted_indices = sorted_indices + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
            condition = paddle.scatter(
                sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
            )
            condition = paddle.cast(condition, "bool").reshape(probs.shape)
            probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
            return probs

        batch_size, cur_len = input_ids.shape
        # used for compute on gpu, avoid memcpy D2H
        cur_len_gpu = paddle.full([1], cur_len, dtype="int64")

        origin_len = input_ids.shape[1]
        # used for compute on gpu, avoid memcpy D2H
        origin_len_gpu = paddle.full([1], origin_len, dtype="int64")

        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        res = paddle.assign(input_ids)
        model_kwargs["res"] = res

        # use_cache is immutable, we split it off other mutable kwargs.
        assert "use_cache" in model_kwargs
        immutable = {"use_cache": model_kwargs["use_cache"]}
        del model_kwargs["use_cache"]

        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **args, **immutable)
            ret = self.gpt(**model_inputs, **immutable)
            return ret

        def _post_process_(outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs):

            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            logits = paddle.matmul(logits, self.gpt.embeddings.word_embeddings.weight, transpose_y=True)

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            if temperature is None or temperature == 1.0:
                probs = paddle.assign(origin_probs)
                origin_probs = paddle.log(origin_probs)
            else:
                origin_probs = paddle.log(origin_probs)
                logits = logits / temperature
                probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                # if self.use_topp_sampling:
                #     try:
                #         from ppfleetx_ops import topp_sampling
                #     except ImportError:
                #         raise ImportError(
                #             "please install ppfleetx_ops by 'cd ppfleetx/ops && python setup_cuda.py install'!"
                #         )
                #     top_ps_tensor = paddle.full(
                #         shape=[paddle.shape(probs)[0]],
                #         fill_value=top_p,
                #         dtype=probs.dtype)
                #     next_tokens = topp_sampling(probs, top_ps_tensor)
                # else:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)

            # if not self.use_topp_sampling:
            next_tokens = paddle.multinomial(probs)

            next_scores = paddle.index_sample(origin_probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            input_ids = next_tokens

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            model_kwargs = self.update_model_kwargs_for_generation(
                next_tokens, outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )

            return input_ids, scores, unfinished_flag, model_kwargs

        # Note(GuoxiaWang):Pre-while call for inference, simulate a do while loop statement
        # the value in model_kwargs should be tensor before while loop
        outputs = _forward_(**model_kwargs)

        input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
            outputs, input_ids, cur_len_gpu, origin_len_gpu, scores, unfinished_flag, model_kwargs
        )
        if not self.inference:
            cur_len += 1
        else:
            # Note(ZhenyuLi): Avoid the synchronization caused by scale in dy2static
            paddle.increment(cur_len)
        paddle.increment(cur_len_gpu)

        attn_mask = model_kwargs["attention_mask"]
        # make the shape of attention_mask = (-1, -1, -1, -1) in dy2static.
        model_kwargs["attention_mask"] = paddle.reshape(attn_mask, paddle.shape(attn_mask))
        model_kwargs["cache"] = outputs[1] if isinstance(outputs, tuple) else None
        while cur_len < max_length:
            # Note(GuoxiaWang): Remove outputs = _forward_(**model_kwargs)
            # and change it to pass directly to _post_process_ to avoid
            # closed-loop problem of dynamic-to-static model
            input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
                _forward_(**model_kwargs),
                input_ids,
                cur_len_gpu,
                origin_len_gpu,
                scores,
                unfinished_flag,
                model_kwargs,
            )
            if not self.inference:
                cur_len += 1
            else:
                # Note(ZhenyuLi): Avoid the synchronization caused by scale in dy2static
                paddle.increment(cur_len)
            paddle.increment(cur_len_gpu)

            if not paddle.any(unfinished_flag):
                break

        return model_kwargs["res"][:, origin_len:], scores

    def forward(self, input_ids=None, **model_kwargs):
        max_length = self.max_length
        min_length = self.min_length
        decode_strategy = self.decode_strategy
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p
        repetition_penalty = self.repetition_penalty
        num_beams = self.num_beams
        num_beam_groups = self.num_beam_groups
        # length_penalty = self.length_penalty
        # early_stopping = self.early_stopping
        bos_token_id = self.bos_token_id
        eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id
        decoder_start_token_id = self.decoder_start_token_id
        forced_bos_token_id = self.forced_bos_token_id
        forced_eos_token_id = self.forced_eos_token_id
        num_return_sequences = self.num_return_sequences
        diversity_rate = self.diversity_rate
        use_cache = self.use_cache

        assert decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
        ], "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            decode_strategy
        )

        bos_token_id = bos_token_id if bos_token_id is not None else getattr(self.gpt, "bos_token_id", None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self.gpt, "eos_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(self.gpt, "pad_token_id", None)
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else getattr(self.gpt, "forced_bos_token_id", None)
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else getattr(self.gpt, "forced_eos_token_id", None)
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else getattr(self.gpt, "decoder_start_token_id", None)
        )

        # params check
        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        if model_kwargs.get("position_ids", None) is None:
            model_kwargs["position_ids"] = paddle.arange(
                0, paddle.shape(model_kwargs["attention_mask"])[-1], dtype=input_ids.dtype
            ).unsqueeze(0)

        self.is_encoder_decoder = False

        model_kwargs["use_cache"] = use_cache

        if self.inference:
            # Note(ZhenyuLi): Avoid the synchronization caused by scale in dy2static
            min_len = input_ids.shape[-1]
            max_len = input_ids.shape[-1]
            paddle.increment(min_len, min_length)
            paddle.increment(max_len, max_length)
        else:
            input_len = input_ids.shape[-1]
            max_len = max_length + input_len
            min_len = min_length + input_len

        logits_processors = self.get_logits_processor(
            min_length=min_len,
            max_length=max_len,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty,
        )

        if decode_strategy == "sampling":
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs
                )

            ret = self.sample(
                input_ids,
                logits_processors,
                max_len,
                pad_token_id,
                eos_token_id,
                top_k,
                top_p,
                temperature,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Not support {decode_strategy} strategy yet!")
        return ret


class GPTLMHead(nn.Layer):
    def __init__(self, hidden_size, vocab_size, embedding_weights=None):
        super(GPTLMHead, self).__init__()
        self.decoder_weight = (
            self.create_parameter(shape=[vocab_size, hidden_size], dtype=paddle.get_default_dtype(), is_bias=True)
            if embedding_weights is None
            else embedding_weights
        )

    def forward(self, hidden_states):
        logits = paddle.tensor.matmul(hidden_states, self.decoder_weight, transpose_y=True)
        return logits


class GPTLMHeadModel(GPTPretrainedModel):
    """
    The GPT Model with a `language modeling` head on top.
    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.
    """

    def __init__(self, gpt, pad_token_id=None):
        super(GPTLMHeadModel, self).__init__()
        self.gpt = gpt
        # def __init__(self, config: paddlenlp.transformers.GPTConfig):
        #     super(GPTLMHeadModel, self).__init__(config)

        # self.gpt = GPTModel(config)
        # self.gpt = GPTModel(
        #     vocab_size= config.vocab_size,
        #     hidden_size=config.hidden_size,
        #     num_hidden_layers=config.num_hidden_layers,
        #     num_attention_heads=config.num_attention_heads,
        #     intermediate_size=config.intermediate_size,
        #     hidden_act=config.hidden_act,
        #     hidden_dropout_prob=config.hidden_dropout_prob,
        #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        #     max_position_embeddings=config.max_position_embeddings,
        #     type_vocab_size=config.type_vocab_size,
        #     initializer_range=config.initializer_range,
        #     pad_token_id=0,
        #     eos_token_id=7,
        #     bos_token_id=0,
        #     eol_token_id=3,
        #     num_partitions=config.num_partitions,
        #     use_recompute=config.use_recompute,
        #     fuse=False,
        # )
        self.criterion = GPTPretrainingCriterion(pad_token_id=pad_token_id)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        loss_mask=None,
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
            # inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        if isinstance(outputs, input_type):
            hidden_states = outputs
        else:
            hidden_states = outputs[0]

        logits = parallel_matmul(hidden_states, self.gpt.embeddings.word_embeddings.weight, True)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            loss = self.criterion(shift_logits, shift_labels)
            # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(shift_logits.reshape((-1, shift_logits.shape[-1])), shift_labels.reshape((-1,)))

            # loss = self.criterion(logits, labels, loss_mask)
            return loss

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
