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
import contextlib
import math
from functools import partial

import numpy as np
import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    pass
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from paddle.utils import try_import

from ...utils.converter import StateDictNameMapping
from ...utils.log import logger
from .. import PretrainedModel, linear_utils, register_base_model
from ..linear_utils import Linear
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ..model_utils import dy2st_nocheck_guard_context
from ..utils import caculate_llm_flops
from .configuration import (
    GPT_PRETRAINED_INIT_CONFIGURATION,
    GPT_PRETRAINED_RESOURCE_FILES_MAP,
    GPTConfig,
)

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None
try:
    from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd
except:
    FusedDropoutAdd = None

OriginLayerNorm = paddle.nn.LayerNorm


__all__ = [
    "GPTModel",
    "GPTPretrainedModel",
    "GPTPretrainingCriterion",
    "GPTForGreedyGeneration",
    "GPTLMHeadModel",
    "GPTForTokenClassification",
    "GPTForSequenceClassification",
    "GPTForCausalLM",
    "GPTEmbeddings",
    "GPTDecoderLayer",
    "GPTLayerNorm",
]


def get_triangle_upper_mask(x, mask=None):
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


def parallel_matmul(x: paddle.Tensor, y: paddle.Tensor, transpose_y=True, tensor_parallel_output=True):
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
        logits = paddle.matmul(input_parallel, y, transpose_y=transpose_y)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=transpose_y)
        return logits


def seed_guard_context(name=None):
    if (
        not isinstance(paddle.base.framework._current_expected_place(), paddle.core.CPUPlace)
        and name in get_rng_state_tracker().states_
    ):
        # todo fix it
        #  ValueError: Length of gpu state list should be equal to the gpu device count
        #  /usr/local/lib/python3.10/dist-packages/paddle/incubate/framework/random.py:119: ValueError
        # return contextlib.nullcontext()
        return get_rng_state_tracker().rng_state(name)
    else:
        return contextlib.nullcontext()


def fast_layer_norm(input, weight, bias, eps):
    fast_ln_lib = try_import("fast_ln")
    return fast_ln_lib.fast_ln(input, weight, bias, eps)[0]


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


def _check_normalized_shape(normalized_shape):
    if isinstance(normalized_shape, (list, tuple)):
        assert len(normalized_shape) == 1


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    """

    Cache = collections.namedtuple("Cache", ["k", "v"])

    def __init__(
        self,
        config,
    ):
        super(MultiHeadAttention, self).__init__()

        self.config = config

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False

        self.use_flash_attention = config.use_flash_attention if flash_attention else False

        self.head_dim = config.hidden_size // config.num_attention_heads
        assert (
            self.head_dim * config.num_attention_heads == config.hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.num_attention_heads = config.num_attention_heads  # default, without tensor parallel

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

        if config.tensor_parallel_degree > 1:
            assert config.num_attention_heads % config.tensor_parallel_degree == 0
            self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_degree

            if config.fuse_attention_qkv:
                self.qkv_proj = ColumnParallelLinear(
                    config.hidden_size,
                    3 * config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                    fuse_matmul_bias=config.use_fused_linear,
                )
            else:
                self.q_proj = ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                    fuse_matmul_bias=config.use_fused_linear,
                )

                self.k_proj = ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                    fuse_matmul_bias=config.use_fused_linear,
                )

                self.v_proj = ColumnParallelLinear(
                    config.hidden_size,
                    config.hidden_size,
                    has_bias=True,
                    gather_output=False,
                    fuse_matmul_bias=config.use_fused_linear,
                )

            self.out_proj = RowParallelLinear(
                config.hidden_size,
                config.hidden_size,
                has_bias=True,
                input_is_parallel=True,
                fuse_matmul_bias=config.use_fused_linear,
            )
        else:
            if self.config.fuse_attention_qkv:
                self.qkv_proj = Linear(config.hidden_size, 3 * config.hidden_size, bias_attr=True)
            else:
                self.q_proj = Linear(config.hidden_size, config.hidden_size, bias_attr=True)
                self.k_proj = Linear(config.hidden_size, config.hidden_size, bias_attr=True)
                self.v_proj = Linear(config.hidden_size, config.hidden_size, bias_attr=True)

            self.out_proj = Linear(config.hidden_size, config.hidden_size, bias_attr=True)

    def _fuse_prepare_qkv(self, query, use_cache=False, past_key_value=None):
        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs / n, seq_len, num_head, head_dim] (n is model parallelism)
            target_shape = [-1, self.config.seq_length, self.num_attention_heads, 3 * self.head_dim]
        else:
            target_shape = [0, 0, self.num_attention_heads, 3 * self.head_dim]

        # bs, seq_len, num_head * 3*head_dim
        mix_layer = self.qkv_proj(query)
        # bs, seq_len, num_head, 3*head_dim
        mix_layer = paddle.reshape_(mix_layer, target_shape)
        # query_states, key_states, value_states => bs, seq_len, num_head, head_dim
        query_states, key_states, value_states = paddle.split(mix_layer, num_or_sections=3, axis=-1)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            # concat along seqlen dimension
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        past_key_value = (key_states, value_states) if use_cache else None

        return query_states, key_states, value_states, past_key_value

    def _prepare_qkv(self, query, key, value, use_cache=False, past_key_value=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        """
        if self.config.sequence_parallel:
            # [bs, seq_len, num_head * head_dim] -> [bs/n, seq_len, num_head * head_dim] (n is model parallelism)
            target_shape = [-1, self.config.seq_length, self.num_attention_heads, self.head_dim]
        else:
            target_shape = [0, 0, self.num_attention_heads, self.head_dim]

        query_states = self.q_proj(query)
        # [bs, seq_len, num_head, head_dim]
        query_states = tensor.reshape(x=query_states, shape=target_shape)

        key_states = self.k_proj(key)
        # [bs, seq_len, num_head, head_dim]
        key_states = tensor.reshape(x=key_states, shape=target_shape)

        value_states = self.v_proj(value)
        # [bs, seq_len, num_head, head_dim]
        value_states = tensor.reshape(x=value_states, shape=target_shape)

        # [bs, seq_len, num_head, head_dim]
        if past_key_value is not None:
            # reuse k, v, self_attention
            # concat along seqlen dimension
            key_states = paddle.concat([past_key_value[0], key_states], axis=1)
            value_states = paddle.concat([past_key_value[1], value_states], axis=1)

        past_key_value = (key_states, value_states) if use_cache else None

        return query_states, key_states, value_states, past_key_value

    def _flash_attention(self, q, k, v, attention_mask=None, output_attentions=False):
        with seed_guard_context("local_seed"):
            out, weights = flash_attention(
                query=q,
                key=k,
                value=v,
                dropout=self.config.attention_probs_dropout_prob,
                causal=q.shape[1] != 1,
                return_softmax=output_attentions,
                training=self.training,
            )
        # [bs, seq_len, num_head, head_dim] -> [bs, seq_len, num_head * head_dim]
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        return (out, weights) if output_attentions else out

    def _core_attention(self, q, k, v, attention_mask=None, output_attentions=False):
        # [bs, seq_len, num_head, head_dim] -> [bs, num_head, seq_len, head_dim]
        perm = [0, 2, 1, 3]
        q = tensor.transpose(x=q, perm=perm)
        k = tensor.transpose(x=k, perm=perm)
        v = tensor.transpose(x=v, perm=perm)
        # scale dot product attention
        product = paddle.matmul(x=q * ((self.config.scale_qk_coeff * self.head_dim) ** -0.5), y=k, transpose_y=True)
        if self.config.scale_qk_coeff != 1.0:
            product = product.scale(self.config.scale_qk_coeff)

        # softmax_mask_fuse_upper_triangle is not supported sif paddle is not compiled with cuda/rocm
        if not paddle.is_compiled_with_cuda():
            attention_mask = get_triangle_upper_mask(product, attention_mask)

        if attention_mask is not None:
            product = product + attention_mask.astype(product.dtype)
            weights = F.softmax(product)
        else:
            weights = incubate.softmax_mask_fuse_upper_triangle(product)

        if self.config.attention_probs_dropout_prob:
            with seed_guard_context("local_seed"):
                weights = F.dropout(
                    weights, self.config.attention_probs_dropout_prob, training=self.training, mode="upscale_in_train"
                )

        out = paddle.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])  # bs, seq_len, num_head, head_dim
        out = tensor.reshape(x=out, shape=[0, 0, -1])  # bs, seq_len, dim

        return (out, weights) if output_attentions else out

    def forward(
        self, query, key, value, attention_mask=None, use_cache=False, past_key_value=None, output_attentions=False
    ):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        if self.config.fuse_attention_qkv:
            # [bs, seq_len, num_head, head_dim]
            q, k, v, past_key_value = self._fuse_prepare_qkv(query, use_cache, past_key_value)
        else:
            # [bs, seq_len, num_head, head_dim]
            q, k, v, past_key_value = self._prepare_qkv(query, key, value, use_cache, past_key_value)

        if self.config.use_flash_attention:
            # Flash Attention now ignore attention mask
            # Current Flash Attention doesn't support attn maskt
            # Paddle Flash Attention input [batch_size, seq_len, num_heads, head_dim]
            # Torch Flash Attention input (batch_size, seqlen, nheads, headdim)
            # bsz, q_len, num_heads, head_dim = q.shape
            # TODO: Support attention mask for flash attention
            attention_func = self._flash_attention
        else:
            # scale dot product attention
            # [bs, seq_len, num_head,]
            attention_func = self._core_attention

        has_gradient = (not q.stop_gradient) or (not k.stop_gradient) or (not v.stop_gradient)
        if self.enable_recompute and self.config.recompute_granularity == "core_attn" and has_gradient:
            outputs = recompute(attention_func, q, k, v, attention_mask, output_attentions, use_reentrant=False)
        else:
            outputs = attention_func(q, k, v, attention_mask=attention_mask, output_attentions=output_attentions)

        if output_attentions:
            out, weights = outputs
        else:
            out = outputs

        # if sequence_parallel is true, out shape are [bs, seq_len, num_head * head_dim / n]
        # else their shape are [bs, q_len, num_head * head_dim / n], n is mp parallelism.

        if self.config.sequence_parallel:
            bs, seq_len, dim = out.shape
            out = out.reshape([bs * seq_len, dim])  # [bs, seq_len, dim / n] => [bs * seq_len, dim / n]

        # project to output
        out = self.out_proj(out)
        # if sequence_parallel is true, out shape are [bs * seq_len / n, dim]
        # else their shape are [bs, seq_len, dim], n is mp parallelism.

        outs = [out]
        if output_attentions:
            outs.append(weights)
        if use_cache:
            outs.append(past_key_value)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(self, config, decoder_layers, norm=None, hidden_size=None):
        super(TransformerDecoder, self).__init__()

        self.config = config
        self.layers = decoder_layers
        self.norm = GPTLayerNorm(config, config.hidden_size, epsilon=1e-5)

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.norm.weight)
            mark_as_sequence_parallel_parameter(self.norm.bias)

        # Note that we will actually perform a recompute only if both enable_recompute and layerwise_recompute are set to True
        # Enable_recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        past_key_value: paddle.Tensor,
        attention_mask: paddle.Tensor,
        use_cache: bool,
        output_attentions: paddle.Tensor,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_attentions)

            return custom_forward

        # GPTDecoderLayer
        # def forward(
        #     self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None, output_attentions=False
        # ):
        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            use_cache,
            past_key_value,
            use_reentrant=self.config.recompute_use_reentrant,
        )
        return hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """

        # [bs * seq_len, embed_dim] -> [seq_len * bs / n, embed_dim] (sequence_parallel)

        output = hidden_states
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for i, mod in enumerate(self.layers):
            has_gradient = not output.stop_gradient
            # def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None, output_attentions=False):
            if self.enable_recompute and has_gradient and self.config.recompute_granularity == "full_attn":
                outputs = self.recompute_training(
                    layer_module=mod,
                    hidden_states=output,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_value=None,
                    output_attentions=output_attentions,
                )
            else:
                outputs = mod(
                    output,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_value=past_key_values[i] if past_key_values is not None else None,
                    output_attentions=output_attentions,
                )

            # outputs = hidden_states if both use_cache and output_attentions are False
            # Otherwise, outputs = (hidden_states, attention if output_attentions, cache if use_cache)
            output = outputs[0] if (use_cache or output_attentions) else outputs
            all_self_attentions = all_self_attentions + (outputs[1],) if output_attentions else None
            all_hidden_states = all_hidden_states + (output,) if output_hidden_states else None
            next_decoder_cache = next_decoder_cache + (outputs[-1],) if use_cache else None

        if self.norm is not None:
            output = self.norm(output)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            temp_list = [output, next_cache, all_hidden_states, all_self_attentions]

            if not (use_cache or output_attentions or output_hidden_states):
                return output

            return tuple(v for v in temp_list if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )


class GPTDecoderLayer(nn.Layer):
    """
    The transformer decoder layer.

    It contains multiheadattention and some linear layers.
    """

    def __init__(self, config: GPTConfig):
        super(GPTDecoderLayer, self).__init__()
        self.config = config

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False

        if not FusedDropoutAdd:
            config.use_fused_dropout_add = False

        self.self_attn = MultiHeadAttention(config=config)

        if config.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

        # TODO:config.fuse_attention_ffn @DrownFish19
        if config.tensor_parallel_degree > 1:
            self.linear1 = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                gather_output=False,
                has_bias=True,
                fuse_matmul_bias=self.config.use_fused_linear,
            )

            self.linear2 = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                input_is_parallel=True,
                has_bias=True,
                fuse_matmul_bias=self.config.use_fused_linear,
            )
        else:
            self.linear1 = Linear(config.hidden_size, config.intermediate_size, bias_attr=True)
            self.linear2 = Linear(config.intermediate_size, config.hidden_size, bias_attr=True)

        self.norm1 = GPTLayerNorm(config, config.hidden_size, epsilon=1e-5)
        self.norm2 = GPTLayerNorm(config, config.hidden_size, epsilon=1e-5)

        if config.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.norm1.weight)
            mark_as_sequence_parallel_parameter(self.norm1.bias)
            mark_as_sequence_parallel_parameter(self.norm2.weight)
            mark_as_sequence_parallel_parameter(self.norm2.bias)

        if config.use_fused_dropout_add:
            self.fused_dropout_add1 = FusedDropoutAdd(config.attention_probs_dropout_prob, mode="upscale_in_train")
            self.fused_dropout_add2 = FusedDropoutAdd(config.hidden_dropout_prob, mode="upscale_in_train")
        else:
            self.dropout1 = nn.Dropout(config.attention_probs_dropout_prob, mode="upscale_in_train")
            self.dropout2 = nn.Dropout(config.hidden_dropout_prob, mode="upscale_in_train")

        if config.hidden_activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = getattr(F, config.hidden_activation)

    def forward(
        self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None, output_attentions=False
    ):
        # when sequence_parallel=True:
        # hidden_states => [bs * seq_len / n, embed_dim]
        residual = hidden_states

        if self.config.normalize_before:
            hidden_states = self.norm1(hidden_states)
        # self.self_attn:
        # def forward(
        #     self, query, key, value, attention_mask=None, use_cache=False, past_key_value=None, output_attentions=False
        # ):
        # self.self_attn(...) --> hidden_states, weights, (past_key_value)
        has_gradient = not hidden_states.stop_gradient
        if self.enable_recompute and has_gradient and self.config.recompute_granularity == "full_attn":
            hidden_states = recompute(
                self.self_attn,
                hidden_states,
                None,
                None,
                attention_mask,
                use_cache,
                past_key_value,
                output_attentions,
                use_reentrant=False,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, None, None, attention_mask, use_cache, past_key_value, output_attentions
            )
        # when sequence_parallel=True:
        # hidden_states => [bs * seq_len / n, embed_dim]
        incremental_cache = hidden_states[-1] if use_cache else None
        attention_weights = hidden_states[1] if output_attentions else None
        hidden_states = hidden_states[0] if (use_cache or output_attentions) else hidden_states

        # Use a ternary operator for a more concise assignment of current_seed
        current_seed = "local_seed" if self.config.sequence_parallel else "global_seed"

        # The 'with' block ensures the correct seed context is used
        with seed_guard_context(current_seed):
            if self.config.use_fused_dropout_add:
                hidden_states = self.fused_dropout_add1(hidden_states, residual)
            else:
                hidden_states = residual + self.dropout1(hidden_states)

        if not self.config.normalize_before:
            hidden_states = self.norm1(hidden_states)

        residual = hidden_states
        if self.config.normalize_before:
            hidden_states = self.norm2(hidden_states)

        # when sequence_parallel=True:
        # hidden_states => [bs * seq_len / n, embed_dim]
        with seed_guard_context(current_seed):
            if not self.config.use_fused_dropout_add:
                hidden_states = residual + self.dropout2(
                    self.linear2(self.activation(self.linear1(hidden_states), approximate=True))
                )
            else:
                hidden_states = self.fused_dropout_add2(
                    self.linear2(self.activation(self.linear1(hidden_states), approximate=True)), residual
                )
        if not self.config.normalize_before:
            hidden_states = self.norm2(hidden_states)

        if not (output_attentions or use_cache):
            return hidden_states

        temp_list = [
            hidden_states,
            attention_weights,
            incremental_cache,
        ]

        return tuple(v for v in temp_list if v is not None)


class GPTEmbeddings(nn.Layer):
    """
    Include embeddings from word and position embeddings.
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

    def forward(self, input_ids, position_ids=None, inputs_embeddings=None):
        if input_ids is not None:
            input_shape = input_ids.shape
            inputs_embeddings = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeddings.shape[:-1]

        if position_ids is None:
            ones = paddle.ones(input_shape, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeddings + position_embeddings

        if self.config.sequence_parallel:
            bs, seq_len, hidden_size = embeddings.shape
            # [bs, seq_len, dim] -> [bs * seq_len, dim]
            embeddings = paddle.reshape_(embeddings, [bs * seq_len, hidden_size])
            # [bs * seq_len / n, dim] (n is mp parallelism)
            embeddings = ScatterOp.apply(embeddings)

        # Use a ternary operator for a more concise assignment of current_seed
        current_seed = "local_seed" if self.config.sequence_parallel else "global_seed"
        # The 'with' block ensures the correct seed context is used
        with seed_guard_context(current_seed):
            embeddings = self.dropout(embeddings)

        return embeddings


class GPTLayerNorm(OriginLayerNorm):
    def __init__(self, config, normalized_shape, epsilon=1e-05, weight_attr=None, bias_attr=None, name=None):
        super().__init__(
            normalized_shape=normalized_shape, epsilon=epsilon, weight_attr=weight_attr, bias_attr=bias_attr
        )

        self.config = config
        _check_normalized_shape(self._normalized_shape)

    def forward(self, input):
        if self.config.use_fast_layer_norm:
            return fast_layer_norm(input, self.weight, self.bias, self._epsilon)
        return super().forward(input)


class GPTPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained GPT models. It provides GPT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
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

    @classmethod
    def _get_fuse_or_split_param_mappings(cls, config: GPTConfig, is_fuse=False):
        # return parameter fuse utils
        from paddlenlp.transformers.conversion_utils import split_or_fuse_func

        fn = split_or_fuse_func(is_fuse=is_fuse)

        # last key is fused key, other keys are to be fused.
        fuse_qkv_keys = (
            "decoder.layers.0.self_attn.q_proj.weight",
            "decoder.layers.0.self_attn.k_proj.weight",
            "decoder.layers.0.self_attn.v_proj.weight",
            "decoder.layers.0.self_attn.qkv_proj.weight",
        )
        fuse_qkv_bias_keys = (
            "decoder.layers.0.self_attn.q_proj.bias",
            "decoder.layers.0.self_attn.k_proj.bias",
            "decoder.layers.0.self_attn.v_proj.bias",
            "decoder.layers.0.self_attn.qkv_proj.bias",
        )
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        fuse_attention_qkv = getattr(config, "fuse_attention_qkv", False)

        final_actions = {}
        if is_fuse:
            if fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for keys in [fuse_qkv_keys, fuse_qkv_bias_keys]:
                        new_keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in keys])
                        final_actions[new_keys] = partial(
                            fn, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                        )
        else:
            if not fuse_attention_qkv:
                for i in range(config.num_hidden_layers):
                    for keys in [fuse_qkv_keys, fuse_qkv_bias_keys]:
                        new_keys = tuple([key.replace("layers.0.", f"layers.{i}.") for key in keys])
                        final_actions[new_keys] = partial(
                            fn, split_nums=3, is_qkv=True, num_heads=num_heads, num_key_value_heads=num_key_value_heads
                        )
        return final_actions

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
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        if isinstance(
            layer,
            (
                nn.Linear,
                nn.Embedding,
                mpu.VocabParallelEmbedding,
                mpu.RowParallelLinear,
                mpu.ColumnParallelLinear,
                linear_utils.RowSequenceParallelLinear,
                linear_utils.ColumnSequenceParallelLinear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                if layer.weight.is_distributed:
                    with rng_tracker():
                        layer.weight.set_value(
                            paddle.tensor.normal(
                                mean=0.0,
                                std=self.config.initializer_range,
                                shape=layer.weight.shape,
                            )
                        )
                else:
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
            if isinstance(layer, GPTDecoderLayer):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.linear2.weight.scale_(factor)
            if isinstance(layer, MultiHeadAttention):
                factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                layer.out_proj.weight.scale_(factor)


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

        self.config = config

        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id
        self.eol_token_id = config.eol_token_id
        self.vocab_size = config.vocab_size

        self.bias = paddle.tril(
            paddle.ones([1, 1, config.max_position_embeddings, config.max_position_embeddings], dtype="int64")
        )

        self.embeddings = GPTEmbeddings(config)

        decoder_layers = nn.LayerList()
        for i in range(config.num_hidden_layers):
            decoder_layers.append(GPTDecoderLayer(config))

        self.decoder = TransformerDecoder(
            config,
            decoder_layers,
        )

    def get_model_flops(self, batch_size=1, seq_length=None, **kwargs):
        if seq_length is None:
            if hasattr(self.config, "seq_length"):
                seq_length = self.config.seq_length
            else:
                seq_length = 2048

        return caculate_llm_flops(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            layer_num=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            seq_length=seq_length,
            recompute=False,
        )

    def get_hardware_flops(self, batch_size=1, seq_length=None, recompute=False, **kwargs):
        if seq_length is None:
            if hasattr(self.config, "seq_length"):
                seq_length = self.config.seq_length
            else:
                seq_length = 2048

        return caculate_llm_flops(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            layer_num=self.config.num_hidden_layers,
            vocab_size=self.config.vocab_size,
            seq_length=seq_length,
            recompute=recompute,
            recompute_granularity=self.config.recompute_granularity,
        )

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape, past_key_values_length=past_key_values_length
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(expanded_attn_mask, 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
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
            past_key_values (list, optional):
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

        if self.config.sequence_parallel and use_cache:
            raise ValueError("We currently only support sequence parallel without cache.")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape((-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        # input_shape => bs, seq_len

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.decoder.layers))

        if position_ids is None:
            past_length = 0
            if past_key_values[0] is not None:
                # bs, seq_len, num_head, head_dim
                past_length = past_key_values[0][0].shape[1]
            position_ids = paddle.arange(past_length, input_shape[-1] + past_length, dtype="int64")
            position_ids = position_ids.unsqueeze(0)
            position_ids = paddle.expand(position_ids, input_shape)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeddings=inputs_embeds
        )

        # TODO, use registered buffer
        length = input_shape[-1]
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
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
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
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
                outputs[idx] = all_hidden_states

        return outputs


class GPTPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT. It calculates the final loss.
    """

    def __init__(self, config):
        super(GPTPretrainingCriterion, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1 and config.tensor_parallel_output:
            self.loss_func = mpu.ParallelCrossEntropy(ignore_index=config.ignore_index)
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none", ignore_index=config.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels, loss_mask=None):
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
        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
            # skip ignore_index which loss == 0
            if loss_mask is None:
                loss_mask = (masked_lm_loss > 0).astype("float32")
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
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
        use_cache=False,
        past_key_values=None,
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
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
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
        output, cached_kvs = self.model(input_ids, use_cache=True, past_key_values=None)
        src_ids = input_ids
        nid = paddle.argmax(output[:, -1, :], axis=-1).reshape([-1, 1])
        src_ids = paddle.concat([src_ids, nid], axis=1)
        cur_len = 0
        with dy2st_nocheck_guard_context():
            while cur_len < self.max_predict_len:
                output, cached_kvs = self.model(nid, use_cache=True, past_key_values=cached_kvs)
                nid = paddle.argmax(output[:, -1, :], axis=-1).reshape([-1, 1])
                src_ids = paddle.concat([src_ids, nid], axis=1)
                cur_len += 1
                if paddle.max(nid) == self.eol_token_id:
                    break
        return src_ids


class GPTLMHead(nn.Layer):
    def __init__(self, config: GPTConfig, embedding_weights=None):
        super(GPTLMHead, self).__init__()
        self.config = config
        self.transpose_y = True

        if embedding_weights is not None:
            self.transpose_y = True
            self.weight = embedding_weights
        else:
            if config.tensor_parallel_degree > 1:
                vocab_size = config.vocab_size // config.tensor_parallel_degree
            else:
                vocab_size = config.vocab_size

            if vocab_size != config.vocab_size:
                with get_rng_state_tracker().rng_state():
                    self.weight = self.create_parameter(
                        shape=[vocab_size, config.hidden_size],
                        dtype=paddle.get_default_dtype(),
                    )
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
            # Must set distributed attr for Tensor Parallel !
            self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
            if self.weight.is_distributed:
                self.weight.split_axis = 0

    def forward(self, hidden_states, tensor_parallel_output=None):
        if self.config.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
            hidden_states = paddle.reshape_(hidden_states, [-1, self.config.seq_length, self.config.hidden_size])

        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output

        logits = parallel_matmul(
            hidden_states, self.weight, transpose_y=self.transpose_y, tensor_parallel_output=tensor_parallel_output
        )
        return logits


class GPTForCausalLM(GPTPretrainedModel):
    """
    The GPT Model with a `language modeling` head on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, config: GPTConfig):
        super(GPTForCausalLM, self).__init__(config)
        self.gpt = GPTModel(config)
        self.lm_head = GPTLMHead(config, embedding_weights=self.gpt.embeddings.word_embeddings.weight)

        self.tie_weights()
        self.criterion = GPTPretrainingCriterion(config)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
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
            past_key_values (Tensor, optional):
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
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
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
            loss = self.criterion(logits, labels)
            # # Shift so that tokens < n predict n
            # shift_logits = logits[:, :-1, :]
            # shift_labels = labels[:, 1:]
            # # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(shift_logits.reshape((-1, shift_logits.shape[-1])), shift_labels.reshape((-1,)))

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

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, past_key_values=None, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        # attention_mask = kwargs.get("attention_mask", None)
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
            "use_cache": use_cache,
            "past_key_values": past_key_values,
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
        self.classifier = Linear(config.hidden_size, config.num_labels)

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
        self.score = Linear(config.hidden_size, config.num_labels, bias_attr=False)

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
            inputs_shape = inputs_embeds.shape[:-1]
            sequence_lengths = paddle.ones(inputs_shape[:-1], dtype="int64") * (inputs_shape[1] - 1)
            logger.warning(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits.gather_nd(paddle.stack([paddle.arange(logits.shape[0]), sequence_lengths], axis=-1))

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


GPTLMHeadModel = GPTForCausalLM
