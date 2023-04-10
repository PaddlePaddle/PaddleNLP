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

import collections
import logging
import math

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.autograd import PyLayer
from paddle.common_ops_import import convert_dtype
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
    get_rng_state_tracker,
)
from paddle.distributed.fleet.utils import recompute
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from ppfleetx.distributed.apis import env
from ppfleetx.utils.log import logger

from .processor import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from .sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    GatherOp,
    RowSequenceParallelLinear,
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None
try:
    from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd
except:
    FusedDropoutAdd = None
FusedDropoutAdd = None


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def parallel_matmul(lm_output, logit_weights, parallel_output):
    """ """
    hcg = env.get_hcg()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()

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
        output_layer_weight_attr=None,
        bias_attr=None,
        fuse_attn_qkv=False,
        scale_qk_coeff=1.0,
        num_partitions=1,
        fused_linear=False,
        use_recompute=False,
        recompute_granularity="full",
        sequence_parallel=False,
        do_recompute=True,
        use_flash_attn=False,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse_attn_qkv = fuse_attn_qkv
        self.scale_qk_coeff = scale_qk_coeff
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.do_recompute = do_recompute
        self.sequence_parallel = sequence_parallel
        self.use_flash_attn = use_flash_attn if flash_attention else None

        if sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
            RowParallelLinear = RowSequenceParallelLinear
        else:
            ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
            RowParallelLinear = fleet.meta_parallel.RowParallelLinear

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert self.num_heads % num_partitions == 0, "num_heads {} must be divisible by num_partitions {}".format(
            self.num_heads, num_partitions
        )
        self.num_heads = self.num_heads // num_partitions

        if self.fuse_attn_qkv:
            assert self.kdim == embed_dim
            assert self.vdim == embed_dim

            self.qkv_proj = ColumnParallelLinear(
                embed_dim,
                3 * embed_dim,
                mp_group=env.get_hcg().get_model_parallel_group(),
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False,
                fuse_matmul_bias=fused_linear,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                embed_dim,
                embed_dim,
                mp_group=env.get_hcg().get_model_parallel_group(),
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False,
                fuse_matmul_bias=fused_linear,
            )

            self.k_proj = ColumnParallelLinear(
                self.kdim,
                embed_dim,
                mp_group=env.get_hcg().get_model_parallel_group(),
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False,
                fuse_matmul_bias=fused_linear,
            )

            self.v_proj = ColumnParallelLinear(
                self.vdim,
                embed_dim,
                mp_group=env.get_hcg().get_model_parallel_group(),
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False,
                fuse_matmul_bias=fused_linear,
            )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            mp_group=env.get_hcg().get_model_parallel_group(),
            weight_attr=output_layer_weight_attr,
            has_bias=True,
            input_is_parallel=True,
            fuse_matmul_bias=fused_linear,
        )

    def _fuse_prepare_qkv(self, query, use_cache=False, cache=None):
        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, -1, 3 * self.head_dim])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)

        assert not isinstance(cache, self.StaticCache), "cache currently does not support the StaticCache type"

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=1)
            v = tensor.concat([cache.v, v], axis=1)
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
        q = tensor.reshape(x=q, shape=[0, 0, -1, self.head_dim])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=1)
            v = tensor.concat([cache.v, v], axis=1)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v, cache) if use_cache else (q, k, v, None)

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
        k = tensor.reshape(x=k, shape=[0, 0, -1, self.head_dim])
        v = tensor.reshape(x=v, shape=[0, 0, -1, self.head_dim])
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

    def _flash_attention(self, q, k, v, attn_mask=None):
        if self.sequence_parallel:
            perm = [1, 0, 2, 3]
            q = tensor.transpose(x=q, perm=perm)
            k = tensor.transpose(x=k, perm=perm)
            v = tensor.transpose(x=v, perm=perm)
        out, weights = flash_attention(
            q, k, v, self.dropout, causal=True, return_softmax=self.need_weights, training=self.training
        )
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        if self.sequence_parallel:
            perm = [1, 0, 2]
            out = tensor.transpose(x=out, perm=perm)
        return (out, weights) if self.need_weights else out

    def core_attn(self, q, k, v, attn_mask=None):
        perm = [1, 2, 0, 3] if self.sequence_parallel else [0, 2, 1, 3]
        q = tensor.transpose(x=q, perm=perm)
        k = tensor.transpose(x=k, perm=perm)
        v = tensor.transpose(x=v, perm=perm)

        # scale dot product attention
        scale_qk_coeff = self.scale_qk_coeff * self.head_dim**0.5
        product = paddle.matmul(x=q.scale(1.0 / scale_qk_coeff), y=k, transpose_y=True)

        if self.scale_qk_coeff != 1.0:
            product = product.scale(self.scale_qk_coeff)

        # softmax_mask_fuse_upper_triangle is not supported sif paddle is not compiled with cuda/rocm
        if not paddle.is_compiled_with_cuda():
            attn_mask = get_triangle_upper_mask(product, attn_mask)

        if attn_mask is not None:
            product = product + attn_mask
            weights = F.softmax(product)
        else:
            weights = incubate.softmax_mask_fuse_upper_triangle(product)

        if self.dropout:
            with get_rng_state_tracker().rng_state("local_seed"):
                weights = F.dropout(weights, self.dropout, training=self.training, mode="upscale_in_train")

        out = paddle.matmul(weights, v)

        # combine heads
        if self.sequence_parallel:
            out = tensor.transpose(out, perm=[2, 0, 1, 3])
        else:
            out = tensor.transpose(out, perm=[0, 2, 1, 3])
        # If sequence_parallel is true, out shape is [s, b, h] after reshape
        # else out shape is [b, s, h]
        out = tensor.reshape(x=out, shape=[0, 0, -1])

        return (out, weights) if self.need_weights else out

    def forward(self, query, key, value, attn_mask=None, use_cache=False, cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # if sequence_parallel is true, query, key, value shape are [s, b, h],
        # else their shape are [b, s, h], n is mp parallelism.
        # no matter sequence_parallel is true or false,
        # after reshape, q, k, v shape should be [b, num_heads/n, s, head_dim]
        # compute q ,k ,v
        if self.fuse_attn_qkv:
            q, k, v, cache = self._fuse_prepare_qkv(query, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache, cache)

        if self.use_flash_attn and attn_mask is None:
            attn_func = self._flash_attention
        else:
            attn_func = self.core_attn

        if self.use_recompute and self.recompute_granularity == "core_attn" and self.do_recompute:
            out = recompute(attn_func, q, k, v, attn_mask)
        else:
            out = attn_func(q, k, v, attn_mask=attn_mask)

        if self.need_weights:
            out, weights = out

        # project to output
        # if sequence_parallel is true, out shape are [s/n, b, h],
        # else their shape are [b, s, h], n is mp parallelism.
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

    def __init__(
        self,
        decoder_layers,
        num_layers,
        norm=None,
        hidden_size=None,
        use_recompute=False,
        recompute_granularity="full",
        sequence_parallel=False,
        no_recompute_layers=None,
    ):
        super(TransformerDecoder, self).__init__()

        if no_recompute_layers is None:
            no_recompute_layers = []
        self.no_recompute_layers = no_recompute_layers

        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.sequence_parallel = sequence_parallel
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)
            # if sequence parallel is true,
            # register hook to all_reduce gradient of weight, bias
            if self.sequence_parallel:
                mark_as_sequence_parallel_parameter(self.norm.weight)
                mark_as_sequence_parallel_parameter(self.norm.bias)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, use_cache=False, cache=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []

        for i, mod in enumerate(self.layers):
            if cache is None:
                if use_cache:
                    output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache)
                    new_caches.append(new_cache)
                else:
                    if (
                        self.use_recompute
                        and self.recompute_granularity == "full"
                        and i not in self.no_recompute_layers
                    ):
                        output = recompute(mod, output, memory, tgt_mask, use_cache, cache)
                    else:
                        output = mod(output, memory, tgt_mask, use_cache, cache)

            else:
                output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache[i])
                new_caches.append(new_cache)

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
        output_layer_weight_attr=None,
        bias_attr=None,
        num_partitions=1,
        fused_linear=False,
        fuse_attn_qkv=False,
        scale_qk_coeff=1.0,
        recompute_attn=False,
        use_recompute=False,
        recompute_granularity="full",
        sequence_parallel=False,
        do_recompute=True,
        skip_quant_tensors=[],
        use_flash_attn=False,
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.sequence_parallel = sequence_parallel
        self.do_recompute = do_recompute

        if sequence_parallel:
            ColumnParallelLinear = ColumnSequenceParallelLinear
            RowParallelLinear = RowSequenceParallelLinear
        else:
            ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
            RowParallelLinear = fleet.meta_parallel.RowParallelLinear

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)
        output_layer_weight_attrs = _convert_param_attr_to_list(output_layer_weight_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            output_layer_weight_attr=output_layer_weight_attrs[0],
            num_partitions=num_partitions,
            fused_linear=fused_linear,
            fuse_attn_qkv=fuse_attn_qkv,
            scale_qk_coeff=scale_qk_coeff,
            use_recompute=use_recompute,
            recompute_granularity=recompute_granularity,
            sequence_parallel=sequence_parallel,
            do_recompute=do_recompute,
            use_flash_attn=use_flash_attn,
        )

        self.linear1 = ColumnParallelLinear(
            d_model,
            dim_feedforward,
            mp_group=env.get_hcg().get_model_parallel_group(),
            weight_attr=weight_attrs[2],
            gather_output=False,
            has_bias=True,
            fuse_matmul_bias=fused_linear,
        )

        self.linear2 = RowParallelLinear(
            dim_feedforward,
            d_model,
            mp_group=env.get_hcg().get_model_parallel_group(),
            weight_attr=output_layer_weight_attrs[2],
            input_is_parallel=True,
            has_bias=True,
            fuse_matmul_bias=fused_linear,
        )

        if "linear1" in skip_quant_tensors:
            self.linear1.skip_quant = True

        if "linear2" in skip_quant_tensors:
            self.linear2.skip_quant = True

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        if self.sequence_parallel:
            # if sequence parallel is true, register hook to all_reduce gradient of bias
            mark_as_sequence_parallel_parameter(self.norm1.weight)
            mark_as_sequence_parallel_parameter(self.norm1.bias)
            mark_as_sequence_parallel_parameter(self.norm2.weight)
            mark_as_sequence_parallel_parameter(self.norm2.bias)
        if not FusedDropoutAdd:
            self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
            self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        else:
            self.fused_dropout_add1 = FusedDropoutAdd(dropout, mode="upscale_in_train")
            self.fused_dropout_add2 = FusedDropoutAdd(act_dropout, mode="upscale_in_train")

        self.activation = getattr(F, activation)

    def forward(self, tgt, memory=None, tgt_mask=None, use_cache=False, cache=None):
        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            if self.use_recompute and self.recompute_granularity == "full_attn" and self.do_recompute:
                tgt = recompute(self.self_attn, tgt, None, None, tgt_mask, use_cache, cache)
            else:
                tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        # If use sequence_parallel, different input partition in dropout
        # should use different seed.
        if self.sequence_parallel:
            current_seed = "local_seed"
        else:
            current_seed = "global_seed"
        with get_rng_state_tracker().rng_state(current_seed):
            if not FusedDropoutAdd:
                tgt = residual + self.dropout1(tgt)
            else:
                tgt = self.fused_dropout_add1(tgt, residual)

        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)

        with get_rng_state_tracker().rng_state(current_seed):
            if not FusedDropoutAdd:
                tgt = residual + self.linear2(F.gelu(self.linear1(tgt), approximate=True))
            else:
                tgt = self.fused_dropout_add2(self.linear2(F.gelu(self.linear1(tgt), approximate=True)), residual)

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        return tgt if use_cache is False else (tgt, incremental_cache)

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
        sequence_parallel=False,
        freeze_embedding=False,
    ):
        super(GPTEmbeddings, self).__init__()

        self.sequence_parallel = sequence_parallel
        self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            mp_group=env.get_hcg().get_model_parallel_group(),
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        if freeze_embedding:
            self.word_embeddings.weight.learning_rate = 0.0
            self.position_embeddings.weight.learning_rate = 0.0

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings
        # if sequence parallel is true, change embedding shape [b, s, h] to [s, b, h]
        # set the sequence dim as first, so the split in sequence dim is data-continuous
        if self.sequence_parallel:
            embeddings = paddle.transpose(embeddings, perm=[1, 0, 2])
            embeddings = ScatterOp.apply(embeddings)
            with get_rng_state_tracker().rng_state("local_seed"):
                embeddings = self.dropout(embeddings)
        else:
            embeddings = self.dropout(embeddings)
        return embeddings


class GPTModelHybrid(nn.Layer):
    def __init__(
        self,
        vocab_size=51200,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        ffn_hidden_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        num_partitions=1,
        use_recompute=False,
        fused_linear=False,
        fuse_attn_qkv=False,
        scale_qk_by_layer_num=True,
        recompute_granularity="full",
        sequence_parallel=False,
        no_recompute_layers=None,
        skip_tensor_map={},
        freeze_embedding=False,
        use_flash_attn=False,
        fused_softmax_with_triangular=False,
    ):

        super(GPTModelHybrid, self).__init__()

        if no_recompute_layers is None:
            no_recompute_layers = []
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.fused_softmax_with_triangular = fused_softmax_with_triangular

        if use_flash_attn:
            if flash_attention:
                logger.info("Flash-attention enabled.")
            else:
                use_flash_attn = False
                logger.warning("Flash-attention is not support in this Paddle version.")

        hcg = env.get_hcg()
        mp_size = hcg.get_model_parallel_world_size()
        if mp_size <= 1:
            sequence_parallel = False
            logging.warning("If mp_size <= 1, sequence_parallel strategy will be turned off in GPTModelHybrid model.")

        self.embeddings = GPTEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            self.initializer_range,
            sequence_parallel,
            freeze_embedding,
        )
        self.sequence_parallel = sequence_parallel

        decoder_layers = nn.LayerList()
        for i in range(num_layers):
            decoder_layers.append(
                TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=ffn_hidden_size,
                    dropout=hidden_dropout_prob,
                    activation="gelu",
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range)
                    ),
                    output_layer_weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=self.initializer_range / math.sqrt(2.0 * num_layers)
                        )
                    ),
                    bias_attr=None,
                    num_partitions=num_partitions,
                    fused_linear=fused_linear,
                    fuse_attn_qkv=fuse_attn_qkv,
                    scale_qk_coeff=num_layers if scale_qk_by_layer_num else 1.0,
                    use_recompute=use_recompute,
                    recompute_granularity=recompute_granularity,
                    sequence_parallel=sequence_parallel,
                    do_recompute=i not in no_recompute_layers,
                    skip_quant_tensors=skip_tensor_map.get("block_{}".format(i), []),
                    use_flash_attn=use_flash_attn,
                )
            )

        self.decoder = TransformerDecoder(
            decoder_layers,
            num_layers,
            norm="LayerNorm",
            hidden_size=hidden_size,
            use_recompute=use_recompute,
            recompute_granularity=recompute_granularity,
            sequence_parallel=sequence_parallel,
            no_recompute_layers=no_recompute_layers,
        )

    def forward(self, input_ids, position_ids=None, attention_mask=None, use_cache=False, cache=None):

        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(attention_mask)[-1] - 1
            position_ids = paddle.arange(past_length, paddle.shape(input_ids)[-1] + past_length, dtype=input_ids.dtype)
            position_ids = position_ids.unsqueeze(0)
            # .expand_as(input_ids)
            position_ids = paddle.expand_as(position_ids, input_ids)
        # if sequence_parallel is true, embedding_output shape is [s/n, b, h]
        # else its shape is [b, s, h], n is mp parallelism
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # fused_softmax_with_triangular is only suppported on GPU/DCU.
        # If on non-GPU devices, we use user defined mask and non-fused softmax.
        if not self.fused_softmax_with_triangular or not paddle.is_compiled_with_cuda():
            # TODO, use registered buffer
            causal_mask = paddle.tensor.triu(
                paddle.ones((paddle.shape(input_ids)[-1], paddle.shape(input_ids)[-1])) * -1e4, diagonal=1
            )
            if attention_mask is not None:
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask + causal_mask
            else:
                attention_mask = causal_mask
            # The tensor returned by triu not in static graph.
            attention_mask.stop_gradient = True

        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=None
            if (self.fused_softmax_with_triangular and self.training and paddle.is_compiled_with_cuda())
            else attention_mask,  # use softmax_mask_fuse_upper_triangle
            use_cache=use_cache,
            cache=cache,
        )

        if self.sequence_parallel:
            encoder_outputs = GatherOp.apply(encoder_outputs)

        return encoder_outputs


class GPTForPretrainingHybrid(nn.Layer):
    """
    GPT Model with pretraining tasks on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, gpt):
        super(GPTForPretrainingHybrid, self).__init__()
        self.gpt = gpt
        # extra_parameters using for sharding stage3 to register extra_parameters
        self.extra_parameters = [get_attr(self.gpt.embeddings.word_embeddings, "weight")]

    def forward(
        self, input_ids, position_ids=None, attention_mask=None, masked_positions=None, use_cache=False, cache=None
    ):

        outputs = self.gpt(
            input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = parallel_matmul(encoder_outputs, get_attr(self.gpt.embeddings.word_embeddings, "weight"), True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterionHybird(nn.Layer):
    """
    Criterion for GPT. It calculates the final loss.
    """

    def __init__(self, topo=None, sequence_parallel=False):
        super(GPTPretrainingCriterionHybird, self).__init__()
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        self.parallel_loss_func = fleet.meta_parallel.ParallelCrossEntropy(
            mp_group=env.get_hcg().get_model_parallel_group()
        )
        self.sequence_parallel = sequence_parallel

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
        hcg = env.get_hcg()
        mp_size = hcg.get_model_parallel_world_size()
        if self.sequence_parallel:
            masked_lm_labels = masked_lm_labels.transpose([1, 0])
            loss_mask = loss_mask.transpose([1, 0])

        if mp_size > 1:
            if paddle.is_compiled_with_cuda() and True:
                masked_lm_loss = self.parallel_loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))
            else:
                prediction_scores = ConcatSoftmaxInput.apply(
                    prediction_scores, group=env.get_hcg().get_model_parallel_group()
                )
                masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))
        else:
            masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))
        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        loss = masked_lm_loss / loss_mask.sum()
        return loss


# these Layers is just for PipelineParallel


class GPTPretrainingCriterionPipe(GPTPretrainingCriterionHybird):
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
        return get_attr(self.word_embeddings, "weight")

    def forward(self, tensors):
        input_ids, position_ids = tensors
        embeddings = super().forward(input_ids=input_ids, position_ids=position_ids)
        return embeddings


class LayerNormPipe(nn.Layer):
    def __init__(
        self,
        normalized_shape,
        epsilon=1e-05,
        weight_attr=None,
        bias_attr=None,
        name=None,
        sequence_parallel=False,
        is_last=False,
    ):
        super(LayerNormPipe, self).__init__()
        self.sequence_parallel = sequence_parallel
        self.is_last = is_last
        self.norm = nn.LayerNorm(
            normalized_shape=normalized_shape, epsilon=epsilon, weight_attr=weight_attr, bias_attr=bias_attr, name=name
        )
        if self.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.norm.weight)
            mark_as_sequence_parallel_parameter(self.norm.bias)

    def forward(self, input):
        output = self.norm(input)
        if self.sequence_parallel and self.is_last:
            output = GatherOp.apply(output)
        return output


class GPTForPretrainingPipe(PipelineLayer):
    """GPTForPretraining adapted for pipeline parallelism.

    The largest change is flattening the GPTModel class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        ffn_hidden_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        num_partitions=1,
        topology=None,
        use_recompute=False,
        fused_linear=False,
        fuse_attn_qkv=False,
        scale_qk_by_layer_num=True,
        recompute_granularity="full",
        virtual_pp_degree=1,
        sequence_parallel=False,
        no_recompute_layers=None,
        pp_recompute_interval=1,
        use_flash_attn=False,
        fused_softmax_with_triangular=False,
    ):

        # forward desc
        self.descs = []

        if no_recompute_layers is None:
            no_recompute_layers = []
        else:
            if recompute_granularity == "full":
                assert len(no_recompute_layers) == 0, "for pp with full recompute, no_recompute_layers is not support"

        if use_flash_attn:
            if flash_attention:
                logger.info("Flash-attention enabled.")
            else:
                use_flash_attn = False
                logger.warning("Flash-attention is not support in this Paddle version.")

        hcg = env.get_hcg()
        mp_size = hcg.get_model_parallel_world_size()
        if mp_size <= 1:
            sequence_parallel = False
            logging.warning(
                "If mp_size <= 1, sequence_parallel strategy will be turned off in GPTForPretrainingPipe model."
            )

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
                initializer_range=0.02,
                sequence_parallel=sequence_parallel,
            )
        )

        for i in range(num_layers):
            self.descs.append(
                LayerDesc(
                    TransformerDecoderLayer,
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=ffn_hidden_size,
                    dropout=hidden_dropout_prob,
                    activation=hidden_act,
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
                    output_layer_weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=initializer_range / math.sqrt(2.0 * num_layers)
                        )
                    ),
                    bias_attr=None,
                    num_partitions=num_partitions,
                    fused_linear=fused_linear,
                    fuse_attn_qkv=fuse_attn_qkv,
                    scale_qk_coeff=num_layers if scale_qk_by_layer_num else 1.0,
                    use_recompute=use_recompute,
                    recompute_granularity=recompute_granularity,
                    sequence_parallel=sequence_parallel,
                    do_recompute=i not in no_recompute_layers,
                    use_flash_attn=use_flash_attn,
                )
            )

        self.descs.append(
            LayerDesc(LayerNormPipe, normalized_shape=hidden_size, sequence_parallel=sequence_parallel, is_last=True)
        )

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
                initializer_range=0.02,
            )
        )

        recompute_interval = 0
        if recompute and recompute_granularity == "full":
            assert pp_recompute_interval <= num_layers // (
                virtual_pp_degree * env.get_hcg().topology().get_dim_size("pipe")
            ), "pp recompute interval should smaller than num layers of each pp chunk"
            recompute_interval = pp_recompute_interval

        seg_method = "layer:TransformerDecoderLayer"
        if num_layers % env.get_hcg().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"

        super().__init__(
            layers=self.descs,
            loss_fn=GPTPretrainingCriterionPipe(sequence_parallel=sequence_parallel),
            topology=env.get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": env.get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=virtual_pp_degree,
        )


class GPTForGenerationHybrid(nn.Layer):
    """
    GPT Model with pretraining tasks on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, gpt, configs):
        super(GPTForGenerationHybrid, self).__init__()
        self.gpt = gpt
        # extra_parameters using for sharding stage3 to register extra_parameters
        self.extra_parameters = [get_attr(self.gpt.embeddings.word_embeddings, "weight")]
        self.configs = configs

        self.max_length = self.configs.get("max_dec_len", 20)
        self.min_length = self.configs.get("min_dec_len", 0)
        self.decode_strategy = self.configs.get("decode_strategy", "sampling")
        self.temperature = self.configs.get("temperature", 1.0)
        self.top_k = self.configs.get("top_k", 0)
        self.top_p = self.configs.get("top_p", 1.0)
        self.repetition_penalty = self.configs.get("repetition_penalty", 1.0)
        self.num_beams = self.configs.get("num_beams", 1)
        self.num_beam_groups = self.configs.get("num_beam_groups", 1)
        self.length_penalty = self.configs.get("length_penalty", 0.0)
        self.early_stopping = self.configs.get("early_stopping", False)
        self.bos_token_id = self.configs.get("bos_token_id", None)
        self.eos_token_id = self.configs.get("eos_token_id", None)
        self.pad_token_id = self.configs.get("pad_token_id", None)
        self.decoder_start_token_id = self.configs.get("decoder_start_token_id", None)
        self.forced_bos_token_id = self.configs.get("forced_bos_token_id", None)
        self.forced_eos_token_id = self.configs.get("forced_eos_token_id", None)
        self.num_return_sequences = self.configs.get("num_return_sequences", 1)
        self.diversity_rate = self.configs.get("diversity_rate", 0.0)
        self.use_cache = self.configs.get("use_cache", True)

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
            attention_mask = (input_ids == pad_token_id).astype(paddle.get_default_dtype()) * -1e9
        else:
            attention_mask = paddle.zeros_like(input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    def update_scores_for_generation(self, scores, next_scores, length, unfinished_flag):
        # update scores

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

        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if num_beam_groups > 1 and diversity_rate > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_rate=diversity_rate, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
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

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, cache=None, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.common_ops_import.convert_dtype(attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        return {"input_ids": input_ids, "position_ids": position_ids, "attention_mask": attention_mask, "cache": cache}

    def update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple):
            model_kwargs["cache"] = outputs[1]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[:, -1:] + 1], axis=-1)

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
        origin_len = input_ids.shape[1]
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        # use_cache is immutable, we split it off other mutable kwargs.
        assert "use_cache" in model_kwargs
        immutable = {"use_cache": model_kwargs["use_cache"]}
        del model_kwargs["use_cache"]

        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **args, **immutable)
            return self.gpt(**model_inputs, **immutable)

        def _post_process_(outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs):

            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            logits = parallel_matmul(logits, get_attr(self.gpt.embeddings.word_embeddings, "weight"), False)

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
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )

            return input_ids, scores, unfinished_flag, model_kwargs

        # Note(GuoxiaWang):Pre-while call for inference, simulate a do while loop statement
        # the value in model_kwargs should be tensor before while loop
        outputs = _forward_(**model_kwargs)

        input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
            outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs
        )
        cur_len += 1

        attn_mask = model_kwargs["attention_mask"]
        # make the shape of attention_mask = (-1, -1, -1, -1) in dy2static.
        model_kwargs["attention_mask"] = paddle.reshape(attn_mask, paddle.shape(attn_mask))
        model_kwargs["cache"] = outputs[1] if isinstance(outputs, tuple) else None
        while cur_len < max_length:
            # Note(GuoxiaWang): Remove outputs = _forward_(**model_kwargs)
            # and change it to pass directly to _post_process_ to avoid
            # closed-loop problem of dynamic-to-static model
            input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
                _forward_(**model_kwargs), input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs
            )
            cur_len += 1

            if not paddle.any(unfinished_flag):
                break

        return input_ids[:, origin_len:], scores

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
                max_length,
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


def get_triangle_upper_mask(x, mask):
    if mask is not None:
        return mask
    mask = paddle.full_like(x, -np.inf)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


class ConcatSoftmaxInput(PyLayer):
    @staticmethod
    def forward(ctx, inp, group=None):
        inputs = []
        paddle.distributed.all_gather(inputs, inp, group=group)
        with paddle.no_grad():
            cat = paddle.concat(inputs, axis=-1)
        ctx.cat_args = group
        return cat

    @staticmethod
    def backward(ctx, grad):
        group = ctx.cat_args
        with paddle.no_grad():
            grads = paddle.split(grad, paddle.distributed.get_world_size(group), axis=-1)
        grad = grads[paddle.distributed.get_rank(group)]
        return grad
