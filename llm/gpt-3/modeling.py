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
import contextlib
import math
from functools import partial

import numpy as np
import paddle
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from configuration import (
    GPT_PRETRAINED_INIT_CONFIGURATION,
    GPT_PRETRAINED_RESOURCE_FILES_MAP,
    GPTConfig,
)
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.model_outputs import CausalLMOutputWithCrossAttentions

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None
try:
    from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd
except:
    FusedDropoutAdd = None


def get_triangle_upper_mask(x, mask):
    if mask is not None:
        return mask
    if paddle.is_compiled_with_xpu():
        # xpu does not support set constant to -np.inf
        mask = paddle.full_like(x, -1e4)
    else:
        mask = paddle.full_like(x, -np.inf)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def parallel_matmul(x, y, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    if is_fleet_init and tensor_parallel_degree > 1 and y.is_distributed:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=True)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=True)
        return logits


def seed_guard_context(name=None):
    if name in get_rng_state_tracker().states_:
        return get_rng_state_tracker().rng_state(name)
    else:
        return contextlib.nullcontext()


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
    ):
        super(MultiHeadAttention, self).__init__()

        self.config = config

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False

        self.use_flash_attention = config.use_flash_attention if flash_attention else None

        self.head_dim = config.hidden_size // config.num_attention_heads
        assert (
            self.head_dim * config.num_attention_heads == config.hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.num_attention_heads = config.num_attention_heads  # default, without tensor parallel
        if config.tensor_parallel_degree > 1:
            assert config.num_attention_heads % config.tensor_parallel_degree == 0
            self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_degree

            if config.fuse_attention_qkv:
                self.qkv_proj = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    3 * config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                    fuse_matmul_bias=config.fused_linear,
                )
            else:
                self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                    fuse_matmul_bias=config.fused_linear,
                )

                self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                    fuse_matmul_bias=config.fused_linear,
                )

                self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                    fuse_matmul_bias=config.fused_linear,
                )

            self.out_proj = fleet.meta_parallel.RowParallelLinear(
                config.hidden_size,
                config.hidden_size,
                has_bias=True,
                input_is_parallel=True,
                fuse_matmul_bias=config.fused_linear,
            )
        else:
            if self.config.fuse_attention_qkv:
                self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias_attr=True)
            else:
                self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=True)
                self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=True)
                self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=True)

            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias_attr=True)

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
            k = paddle.full(
                shape=[key.shape[0], self.num_attention_heads, 0, self.head_dim], dtype=key.dtype, fill_value=0
            )
            v = paddle.full(
                shape=[key.shape[0], self.num_attention_heads, 0, self.head_dim], dtype=key.dtype, fill_value=0
            )
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def _flash_attention(self, q, k, v, attn_mask=None, output_attentions=False):
        out, weights = flash_attention(
            q,
            k,
            v,
            self.config.hidden_dropout_prob,
            causal=True,
            return_softmax=output_attentions,
            training=self.training,
        )
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        return (out, weights) if output_attentions else out

    def core_attn(self, q, k, v, attn_mask=None, output_attentions=False):
        perm = [0, 2, 1, 3]
        q = tensor.transpose(x=q, perm=perm)
        k = tensor.transpose(x=k, perm=perm)
        v = tensor.transpose(x=v, perm=perm)

        # scale dot product attention

        scale_qk_coeff = self.config.scale_qk_coeff * self.head_dim**0.5
        product = paddle.matmul(x=q.scale(1.0 / scale_qk_coeff), y=k, transpose_y=True)

        if self.config.scale_qk_coeff != 1.0:
            product = product.scale(self.config.scale_qk_coeff)

        # softmax_mask_fuse_upper_triangle is not supported sif paddle is not compiled with cuda/rocm
        if not paddle.is_compiled_with_cuda():
            attn_mask = get_triangle_upper_mask(product, attn_mask)

        if attn_mask is not None:
            product = product + attn_mask
            weights = F.softmax(product)
        else:
            weights = incubate.softmax_mask_fuse_upper_triangle(product)

        if self.config.hidden_dropout_prob:

            with seed_guard_context("lcoal_seed"):
                weights = F.dropout(
                    weights, self.config.hidden_dropout_prob, training=self.training, mode="upscale_in_train"
                )

        out = paddle.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, -1])

        return (out, weights) if output_attentions else out

    def forward(self, query, key, value, attn_mask=None, use_cache=False, cache=None, output_attentions=False):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if self.config.fuse_attention_qkv:
            q, k, v, cache = self._fuse_prepare_qkv(query, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache, cache)

        if self.use_flash_attention and attn_mask is None:
            attn_func = self._flash_attention
        else:
            attn_func = self.core_attn
        has_gradient = (not q.stop_gradient) or (not k.stop_gradient) or (not v.stop_gradient)
        if self.enable_recompute and self.config.recompute_granularity == "core_attn" and has_gradient:
            out = recompute(attn_func, q, k, v, attn_mask, output_attentions, use_reentrant=False)
        else:
            out = attn_func(q, k, v, attn_mask=attn_mask, output_attentions=output_attentions)

        if output_attentions:
            out, weights = out

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if output_attentions:
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
        config,
        decoder_layers,
    ):
        super(TransformerDecoder, self).__init__()

        self.config = config
        self.layers = decoder_layers
        self.norm = nn.LayerNorm(config.hidden_size, epsilon=1e-5)

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False

    def forward(
        self, tgt, tgt_mask=None, memory=None, memory_mask=None, use_cache=False, cache=None, output_attentions=False
    ):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []
        all_self_attentions = [] if output_attentions else None

        for i, mod in enumerate(self.layers):
            if cache is None:
                if use_cache:
                    output, new_cache = mod(
                        output,
                        tgt_mask=tgt_mask,
                        memory=memory,
                        use_cache=use_cache,
                        cache=cache,
                        output_attentions=output_attentions,
                    )
                    new_caches.append(new_cache)
                else:
                    has_gradient = not output.stop_gradient
                    if self.enable_recompute and self.config.recompute_granularity == "full" and has_gradient:
                        output = recompute(
                            mod, output, tgt_mask, memory, use_cache, cache, output_attentions, use_reentrant=False
                        )
                    else:
                        output = mod(output, tgt_mask, memory, use_cache, cache, output_attentions)

            else:
                output, new_cache = mod(
                    output,
                    tgt_mask=tgt_mask,
                    memory=memory,
                    use_cache=use_cache,
                    cache=cache[i],
                    output_attentions=output_attentions,
                )
                new_caches.append(new_cache)

            if output_attentions:
                output, weights = output
                all_self_attentions.append(weights)

        if self.norm is not None:
            output = self.norm(output)

        outputs = [output]
        if output_attentions:
            outputs.append(all_self_attentions)
        if use_cache:
            outputs.append(new_caches)
        return output if len(outputs) == 1 else tuple(outputs)

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

    def __init__(self, config: GPTConfig):

        super(TransformerDecoderLayer, self).__init__()

        self.config = config

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False

        if not FusedDropoutAdd:
            config.use_fused_dropout_add = False

        self.self_attn = MultiHeadAttention(config=config)

        if config.tensor_parallel_degree > 1:
            self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                gather_output=False,
                has_bias=True,
                fuse_matmul_bias=self.config.fused_linear,
            )
            self.linear2 = fleet.meta_parallel.RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                input_is_parallel=True,
                has_bias=True,
                fuse_matmul_bias=self.config.fused_linear,
            )
        else:
            self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=True)
            self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=True)

        self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=1e-5)

        if not config.use_fused_dropout_add:
            self.dropout1 = nn.Dropout(config.hidden_dropout_prob, mode="upscale_in_train")
            self.dropout2 = nn.Dropout(config.hidden_dropout_prob, mode="upscale_in_train")
        else:
            self.fused_dropout_add1 = FusedDropoutAdd(config.hidden_dropout_prob, mode="upscale_in_train")
            self.fused_dropout_add2 = FusedDropoutAdd(config.hidden_dropout_prob, mode="upscale_in_train")

        self.activation = getattr(F, config.hidden_activation)

    def forward(self, tgt, tgt_mask=None, memory=None, use_cache=False, cache=None, output_attentions=False):
        residual = tgt

        if self.config.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            has_gradient = not tgt.stop_gradient
            if self.enable_recompute and self.config.recompute_granularity == "full_attn" and has_gradient:
                tgt = recompute(
                    self.self_attn, tgt, None, None, tgt_mask, use_cache, cache, output_attentions, use_reentrant=False
                )
            else:
                tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache, output_attentions)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache, output_attentions)

        if output_attentions:
            tgt, weights = tgt

        with seed_guard_context("global_seed"):
            if not self.config.use_fused_dropout_add:
                tgt = residual + self.dropout1(tgt)
            else:
                tgt = self.fused_dropout_add1(tgt, residual)

        if not self.config.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.config.normalize_before:
            tgt = self.norm2(tgt)

        with seed_guard_context("global_seed"):
            if not self.config.use_fused_dropout_add:
                tgt = residual + self.linear2(F.gelu(self.linear1(tgt), approximate=True))
            else:
                tgt = self.fused_dropout_add2(self.linear2(F.gelu(self.linear1(tgt), approximate=True)), residual)

        if not self.config.normalize_before:
            tgt = self.norm2(tgt)

        if output_attentions:
            tgt = (tgt, weights)
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
        config,
    ):
        super(GPTEmbeddings, self).__init__()

        self.config = config

        if config.tensor_parallel_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
            )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "gpt"
    config_class = GPTConfig
    pretrained_init_configuration = GPT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = GPT_PRETRAINED_RESOURCE_FILES_MAP

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}
            base_actions = {
                # Column Linear
                "layers.0.linear1.weight": partial(fn, is_column=True),
                "layers.0.linear1.bias": partial(fn, is_column=True),
                # Row Linear
                "word_embeddings.weight": partial(fn, is_column=False),
                "layers.0.self_attn.out_proj.weight": partial(fn, is_column=False),
                "layers.0.linear2.weight": partial(fn, is_column=False),
            }

            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.qkv_proj.bias"] = partial(fn, is_column=True)
            else:
                base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(
            layer,
            (
                nn.Linear,
                nn.Embedding,
                fleet.meta_parallel.VocabParallelEmbedding,
                fleet.meta_parallel.ColumnParallelLinear,
                fleet.meta_parallel.RowParallelLinear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )
        # Layer.apply is DFS https://github.com/PaddlePaddle/Paddle/blob/a6f5021fcc58b21f4414bae6bf4731ef6971582c/python/paddle/nn/layer/layers.py#L527-L530
        # sublayer is init first
        # scale RowParallelLinear weight
        with paddle.no_grad():
            if isinstance(layer, TransformerDecoderLayer):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.linear2.weight.scale_(factor)
            if isinstance(layer, MultiHeadAttention):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.out_proj.weight.scale_(factor)


@register_base_model
class GPTModel(GPTPretrainedModel):
    """
    The base model of gpt.
    """

    def __init__(self, config: GPTConfig):
        super(GPTModel, self).__init__(config)

        self.config = config

        self.embeddings = GPTEmbeddings(config)

        decoder_layers = nn.LayerList()
        for i in range(config.num_hidden_layers):
            decoder_layers.append(TransformerDecoderLayer(config))

        self.decoder = TransformerDecoder(
            config,
            decoder_layers,
        )

    def forward(
        self, input_ids, position_ids=None, attention_mask=None, use_cache=False, cache=None, output_attentions=False
    ):
        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(attention_mask)[-1] - 1
            position_ids = paddle.arange(past_length, paddle.shape(input_ids)[-1] + past_length, dtype="int64")
            position_ids = position_ids.unsqueeze(0)
            input_shape = paddle.shape(input_ids)
            position_ids = paddle.expand(position_ids, input_shape)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        if not self.config.fused_softmax_with_triangular or not paddle.is_compiled_with_cuda():
            # TODO, use registered buffer
            causal_mask = paddle.tensor.tril(
                paddle.ones((paddle.shape(input_ids)[-1], paddle.shape(input_ids)[-1]), dtype="int64"),
            )
            if attention_mask is not None:
                if attention_mask.dtype != paddle.int64:
                    attention_mask = paddle.cast(attention_mask, dtype=paddle.int64)
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - (attention_mask & causal_mask)) * -1e4
            else:
                attention_mask = (1.0 - causal_mask) * -1e4

        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=None
            if (self.config.fused_softmax_with_triangular and self.training)
            else attention_mask,  # use softmax_mask_fuse_upper_triangle
            use_cache=use_cache,
            cache=cache,
            output_attentions=output_attentions,
        )
        return encoder_outputs


class GPTPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT.

    It calculates the final loss.
    """

    def __init__(self, config):
        super(GPTPretrainingCriterion, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1 and config.tensor_parallel_output:
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy(ignore_index=config.ignore_index)
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=config.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels, loss_mask=None):
        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
            masked_lm_loss = masked_lm_loss[masked_lm_loss > 0].astype("float32")
            loss = paddle.mean(masked_lm_loss)

        return loss


class GPTForCausalLM(GPTPretrainedModel):
    """
    The GPT Model with a `language modeling` head on top.
    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.
    """

    def __init__(self, config: GPTConfig):
        super(GPTForCausalLM, self).__init__(config)
        self.config = config
        self.gpt = GPTModel(config)
        self.criterion = GPTPretrainingCriterion(config)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        cache=None,
        labels=None,
        output_attentions=None,
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        outputs = self.gpt(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache=cache,
            output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        if isinstance(outputs, input_type):
            hidden_states = outputs
        else:
            hidden_states = outputs[0]

        tensor_parallel_output = (
            self.config.tensor_parallel_output and labels is not None and self.config.tensor_parallel_degree > 1
        )
        logits = parallel_matmul(hidden_states, self.gpt.embeddings.word_embeddings.weight, tensor_parallel_output)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

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
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and float(paddle.any(input_ids == pad_token_id))
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return paddle.unsqueeze(attention_mask, axis=[1, 2])
