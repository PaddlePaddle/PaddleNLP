# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""Paddle BLOOM model."""
from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.converter import StateDictNameMapping, init_name_mappings
from paddlenlp.utils.log import logger

from .configuration import BloomConfig
from .processor import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)

__all__ = [
    "BloomModel",
    "BloomForPretraining",
    "BloomForCausalLM",
    "BloomForSequenceClassification",
    "BloomForTokenClassification",
    "BloomForGeneration",
]


def parallel_matmul(x: Tensor, y: Tensor, parallel_output=True):
    is_fleet_init = True
    world_size = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        world_size = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False
    if is_fleet_init and world_size > 1:
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        input_parallel = paddle.distributed.collective._c_identity(x, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, y, transpose_y=True)
        if parallel_output:
            return logits
        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(x, y, transpose_y=True)
        return logits


def split_tensor_along_last_dim(tensor: Tensor, num_partitions: int, contiguous_split_chunks: bool = False):
    """Split a tensor along its last dimension -> query/key/value layer
    Args:
        tensor: ([`paddle.Tensor`], *required*):
            input tensor to split
        num_partitions ([`int`], *required*):
            number of partitions to split the tensor
        contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
            If True, make each chunk contiguous in memory.
    """
    return paddle.split(tensor, 3, axis=-1)


def _make_causal_mask(input_ids_shape, past_key_values_length: int) -> Tensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = paddle.ones((target_length, target_length + past_key_values_length), dtype="bool")
    # ONNX doesn't support `Tensor.triu` properly, thus we use this workaround
    seq_ids = paddle.arange(target_length)
    mask[:, past_key_values_length:] = seq_ids[:, None] >= seq_ids[None, :]

    expanded_mask = mask.unsqueeze(axis=[0, 1]).expand(
        [batch_size, 1, target_length, target_length + past_key_values_length]
    )
    return expanded_mask


def _expand_2d_mask(mask: Tensor, tgt_length: int) -> Tensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask.stop_gradient = True
    return mask.unsqueeze(axis=[1, 2]).expand([batch_size, 1, tgt_length, src_length])


def build_alibi_tensor(attention_mask: Tensor, num_heads: int, dtype) -> Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`paddle.dtype`, *optional*, default=`paddle.bfloat16`):
            dtype of the output tensor
    """
    # _, seq_length = attention_mask.shape[0], attention_mask.shape[-1]
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = paddle.full([], 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=paddle.float32)
    powers = paddle.arange(1, 1 + closest_power_of_2, dtype=paddle.float32)
    slopes = paddle.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = paddle.to_tensor(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=paddle.float32)
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = paddle.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=paddle.float32)
        slopes = paddle.concat([slopes, paddle.pow(extra_base, extra_powers)], axis=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.astype(paddle.float32).cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    # return alibi
    return paddle.cast(alibi, dtype)
    # return paddle.cast(alibi.reshape([batch_size * num_heads, 1, seq_length]), dtype)


def dropout_add(x: Tensor, residual: Tensor, prob: float, training: bool) -> Tensor:
    """
    Dropout add function

    Args:
        x (`paddle.tensor`, *required*):
            input tensor
        residual (`paddle.tensor`, *required*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def pre_process_alibi_for_pad(alibi, attention_mask, num_heads):
    """
    Args:
    Pre-process the alibi tensor for padding.
        alibi: ([`paddle.tensor`], *required*):
            alibi tensor to pre-process
        attention_mask: ([`paddle.tensor`], *required*):
            attention mask to pre-process"""

    # Sanity check if we are not inferring less tokens than the total sequence length
    # This usually happens when the inference is done with past_key_values
    # In this case we re-create the alibi tensor with the correct sequence length
    if attention_mask.shape[-1] != alibi.shape[-1]:
        alibi = build_alibi_tensor(attention_mask, num_heads, alibi.dtype).repeat_interleave(
            attention_mask.shape[0], axis=0
        )
    # Get the indexes of the padding tokens
    index_x0, index_y0 = paddle.where(attention_mask == 0.0)
    index_x1, index_y1 = paddle.where(attention_mask == 1.0)

    # Clone the embeddings  - we can detach because the embeddings are not learned
    # Get a refence tensor
    slice_reference_alibi = build_alibi_tensor(attention_mask, num_heads, alibi.dtype)

    # Loop over the batch where the padding is and replace the alibi tensor by the reference tensor
    # Only where you do not have padding. Replace padding tokens by zeros
    # This operation can be seen as a shifting operation.
    for i, index in enumerate(paddle.unique(index_x0)):
        slice_to_modify = paddle.zeros_like(slice_reference_alibi)
        index_shift = index_y1[index_x1 == index]
        shift_value = len(index_shift)
        slice_to_modify[:, :, index_shift] = slice_reference_alibi[:, :, :shift_value]
        alibi[index * num_heads : (index + 1) * num_heads] = slice_to_modify
    return alibi


def bloom_gelu_forward(x):
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`paddle.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + paddle.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_back(g, x):
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + paddle.erf(x * 0.70710678)) +
    0.3989423 * x * paddle.exp(-0.5 * x * x)

    Args:
        g (`paddle.tensor`, *required*):
            gradient output tensor
        x (`paddle.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = paddle.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


def baddbmm(input, batch1, batch2, beta=1.0, alpha=1.0):
    return beta * input + alpha * paddle.matmul(batch1, batch2)


class GeLUFunction(PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        return bloom_gelu_back(grad_output, input)


class BloomGelu(nn.Layer):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    paddlescriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not paddlescriptable: https://github.com/pypaddle/pypaddle/issues/22329

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return bloom_gelu_forward(x)
        # if self.training and in_dygraph_mode():
        #    return GeLUFunction.apply(x)
        # else:
        #    return bloom_gelu_forward(x)


class BloomAttention(nn.Layer):
    def __init__(self, config, layer_number=None):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if config.tensor_parallel_degree > 1:
            assert self.num_heads % config.tensor_parallel_degree == 0
            self.num_heads = self.num_heads // config.tensor_parallel_degree

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        if config.tensor_parallel_degree > 1:
            self.query_key_value = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size, 3 * self.hidden_size, has_bias=True, gather_output=False
            )
        else:
            self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias_attr=True)

        if config.tensor_parallel_degree > 1:
            self.dense = fleet.meta_parallel.RowParallelLinear(
                self.hidden_size, self.hidden_size, has_bias=True, input_is_parallel=True
            )
        else:
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(self, fused_qkv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`paddle.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.reshape([batch_size, seq_length, self.num_heads, 3, self.head_dim])
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        Merge heads together over the last dimenstion

        Args:
            x: (`paddle.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            paddle.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.reshape([batch_size, self.num_heads, seq_length, self.head_dim])

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.transpose([0, 2, 1, 3])

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape([batch_size, seq_length, self.num_heads * self.head_dim])

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        alibi: Tensor,
        attention_mask: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose([0, 2, 1, 3])
        key_layer = key_layer.transpose([0, 2, 3, 1])
        value_layer = value_layer.transpose([0, 2, 1, 3])
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size, self.num_heads, head_dim, kv_length]
            #  - value: [batch_size, self.num_heads, kv_length, head_dim]
            key_layer = paddle.concat((past_key, key_layer), axis=3)
            value_layer = paddle.concat((past_value, value_layer), axis=2)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        _, _, _, kv_length = key_layer.shape

        query_layer = query_layer.reshape([batch_size * self.num_heads, q_length, self.head_dim])
        key_layer = key_layer.reshape([batch_size * self.num_heads, self.head_dim, kv_length])
        value_layer = value_layer.reshape([batch_size * self.num_heads, kv_length, self.head_dim])

        # [batch_size * num_heads, q_length, kv_length]
        # alibi:[batch_size * num_heads, q_length, kv_length]
        # we use `Tensor.baddbmm` instead of `paddle.baddbmm` as the latter isn't supported by TorchScript v1.11
        attention_scores = baddbmm(
            alibi, batch1=query_layer, batch2=key_layer, beta=self.beta, alpha=self.inv_norm_factor
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        # attention_scores = matmul_result.reshape([batch_size, self.num_heads, q_length, kv_length])

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = query_layer.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype != paddle.float32:
            attention_scores = paddle.cast(attention_scores, paddle.float32)
            attn_weights = attention_scores + attention_mask
            attention_probs = paddle.cast(F.softmax(attn_weights, axis=-1, dtype=paddle.float32), dtype=input_dtype)
        else:
            attn_weights = attention_scores + attention_mask
            attention_probs = F.softmax(attn_weights, axis=-1)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.reshape([batch_size * self.num_heads, q_length, kv_length])

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = paddle.matmul(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pypaddle/pypaddle/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = paddle.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            # output attentions should be: [batch_size, self.num_heads, q_length, kv_length]
            attention_probs = attention_probs.reshape([batch_size, self.num_heads, q_length, kv_length])
            outputs += (attention_probs,)

        return outputs


class BloomMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        if config.tensor_parallel_degree > 1:
            self.dense_h_to_4h = fleet.meta_parallel.ColumnParallelLinear(
                hidden_size, 4 * hidden_size, gather_output=False, has_bias=True
            )

            self.dense_4h_to_h = fleet.meta_parallel.RowParallelLinear(
                4 * hidden_size, hidden_size, input_is_parallel=True, has_bias=True
            )

        else:
            self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
            self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout
        self.gelu_impl = BloomGelu()

    def forward(self, hidden_states, residual):
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = paddle.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + nn.functional.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output


class BloomBlock(nn.Layer):
    def __init__(self, config, layer_number=None):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)
        self.n_head = config.n_head
        self.self_attention = BloomAttention(config, layer_number=layer_number)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, epsilon=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi=None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.

        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]
        return outputs  # hidden_states, present, attentions


class BloomPreTrainedModel(PretrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BloomConfig
    base_model_prefix = "bloom"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BloomBlock"]

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
                "h.0.self_attention.query_key_value.weight": partial(fn, is_column=True),
                "h.0.self_attention.query_key_value.bias": partial(fn, is_column=True),
                "h.0.mlp.dense_h_to_4h.bias": partial(fn, is_column=True),
                "h.0.mlp.dense_h_to_4h.weight": partial(fn, is_column=True),
                # Row Linear
                "word_embeddings.weight": partial(fn, is_column=False),
                "h.0.self_attention.dense.weight": partial(fn, is_column=False),
                "h.0.mlp.dense_4h_to_h.weight": partial(fn, is_column=False),
            }
            for key, action in base_actions.items():
                if "h.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("h.0.", f"h.{i}.")] = action
                final_actions[key] = action
            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.n_layer)

        return mappings

    def _init_weights(self, layer):
        """Initialize the weights."""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(
                paddle.tensor.normal(mean=0.0, std=self.config.initializer_range, shape=layer.weight.shape)
            )
            if getattr(layer, "bias", None) is not None:
                layer.weight.set_value(paddle.zeros(shape=layer.weight.shape, dtype=paddle.get_default_dtype()))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BloomModel):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_to_bloom_cache(past_key_value: Tuple[Tuple[Tensor, Tensor]]) -> Tuple[Tuple[Tensor, Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape([batch_size_times_num_heads, head_dim, seq_length]),
                layer_past[1].reshape([batch_size_times_num_heads, seq_length, head_dim]),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[Tensor, Tensor]], batch_size: int
    ) -> Tuple[Tuple[Tensor, Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].reshape([batch_size, num_heads, head_dim, seq_length]),
                layer_past[1].reshape([batch_size, num_heads, seq_length, head_dim]),
            )
            for layer_past in past_key_value
        )

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"

        head_mask = paddle.cast(head_mask, dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`paddle.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `paddle.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    @classmethod
    def _get_name_mappings(cls, config: BloomConfig) -> list[StateDictNameMapping]:
        hard_mapping = [
            "word_embeddings.weight",
            "word_embeddings_layernorm.weight",
            "word_embeddings_layernorm.bias",
            "ln_f.weight",
            "ln_f.bias",
        ]
        for i in range(config.n_layer):
            hard_mapping.extend(
                [
                    f"h.{i}.input_layernorm.weight",
                    f"h.{i}.input_layernorm.bias",
                    [
                        f"h.{i}.self_attention.query_key_value.weight",
                        None,
                        "transpose",
                    ],
                    f"h.{i}.self_attention.query_key_value.bias",
                    [f"h.{i}.self_attention.dense.weight", None, "transpose"],
                    f"h.{i}.self_attention.dense.bias",
                    f"h.{i}.post_attention_layernorm.weight",
                    f"h.{i}.post_attention_layernorm.bias",
                    [f"h.{i}.mlp.dense_h_to_4h.weight", None, "transpose"],
                    [f"h.{i}.mlp.dense_4h_to_h.weight", None, "transpose"],
                    f"h.{i}.mlp.dense_h_to_4h.bias",
                    f"h.{i}.mlp.dense_4h_to_h.bias",
                ]
            )

        init_name_mappings(hard_mapping)

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mapping)]
        model_class_name = config.architectures[0]

        if model_class_name != "BloomModel":
            for mapping in mappings:
                mapping.source_name = "transformer." + mapping.source_name
                mapping.target_name = "bloom." + mapping.target_name

        if model_class_name == "BloomForSequenceClassification":
            mappings.append(StateDictNameMapping("score.weight", None, "transpose"))
        if model_class_name == "BloomForTokenClassification":
            mappings.append(StateDictNameMapping("classifier.weight", None, "transpose"))
            mappings.append(StateDictNameMapping("classifier.bias"))

        return mappings


class BloomModel(BloomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = 0

        # Recompute defaults to False and is controlled by Trainer
        self.enable_recompute = False
        self.embed_dim = config.hidden_size
        self.n_head = config.n_head

        # Embedding + LN Embedding
        # self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        if config.tensor_parallel_degree > 1:
            self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range)
                ),
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        self.word_embeddings_layernorm = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.LayerList([BloomBlock(config, layer_number=i) for i in range(config.n_layer)])

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int, int], past_key_values_length: int, num_heads: int, dtype
    ) -> Tensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        if len(attention_mask.shape) == 2:
            expanded_attn_mask = _expand_2d_mask(attention_mask, tgt_length=src_length)
        elif len(attention_mask.shape) == 3:
            # [batch_size,tgt_length, src_length] -> [batch_size, 1, tgt_length, src_length]
            expanded_attn_mask = attention_mask.unsqueeze(1)
        elif len(attention_mask.shape) == 4:
            expanded_attn_mask = attention_mask

        if combined_attention_mask is not None:
            expanded_attn_mask = expanded_attn_mask & combined_attention_mask

        mask_shape = expanded_attn_mask.shape
        expanded_attn_mask = expanded_attn_mask.expand([mask_shape[0], num_heads, mask_shape[2], mask_shape[3]])
        zero = paddle.zeros(expanded_attn_mask.shape, dtype=dtype)
        neg_inf = paddle.full(expanded_attn_mask.shape, paddle.finfo(dtype).min, dtype=dtype)
        expanded_attn_mask = paddle.where(expanded_attn_mask, zero, neg_inf)
        batch_size, num_heads, sq_len, kv_len = expanded_attn_mask.shape
        return expanded_attn_mask.reshape([batch_size * num_heads, sq_len, kv_len])

    def set_input_embeddings(self, new_embeddings: Tensor):
        self.word_embeddings = new_embeddings

    @paddle.jit.not_to_static
    def recompute_training(
        self, block, hidden_states, layer_past, attention_mask, head_mask, use_cache, output_attentions, alibi
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(block),
            hidden_states,
            layer_past,
            attention_mask,
            head_mask,
            use_cache,
            output_attentions,
            alibi,
            use_reentrant=False,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> Union[Tuple[Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        past_key_values = kwargs.get("cache", past_key_values)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[3]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is None:
            attention_mask = paddle.ones([batch_size, seq_length_with_past], dtype="bool")
        elif attention_mask.dtype != paddle.bool:
            attention_mask = paddle.cast(attention_mask, "bool")

        if len(attention_mask.shape) > 2:
            _attention_mask = paddle.ones([batch_size, seq_length_with_past], dtype="bool")
            alibi = build_alibi_tensor(_attention_mask, self.config.n_head, dtype=hidden_states.dtype)
        else:
            alibi = build_alibi_tensor(attention_mask, self.config.n_head, dtype=hidden_states.dtype)

        if self.config.tensor_parallel_degree > 1:
            block_size = self.config.n_head // self.config.tensor_parallel_degree
            alibi = alibi[
                :, self.config.tensor_parallel_rank * block_size : (self.config.tensor_parallel_rank + 1) * block_size
            ]
            alibi = alibi.reshape([batch_size * block_size, 1, seq_length_with_past])
            causal_mask = self._prepare_attn_mask(
                attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=past_key_values_length,
                num_heads=block_size,
                dtype=hidden_states.dtype,
            )
        else:
            alibi = alibi.reshape([batch_size * self.config.n_head, 1, seq_length_with_past])
            causal_mask = self._prepare_attn_mask(
                attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=past_key_values_length,
                num_heads=self.config.n_head,
                dtype=hidden_states.dtype,
            )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            has_gradient = not hidden_states.stop_gradient
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.enable_recompute and has_gradient:
                outputs = self.recompute_training(
                    block,
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BloomLMHead(nn.Layer):
    def __init__(self, config, embedding_weights=None):
        super(BloomLMHead, self).__init__()
        self.decoder_weight = (
            self.create_parameter(shape=[config.vocab_size, config.hidden_size], dtype=paddle.get_default_dtype())
            if embedding_weights is None
            else embedding_weights
        )
        self.config = config

    def forward(self, hidden_states, parallel_output):
        logits = parallel_matmul(hidden_states, self.decoder_weight, parallel_output=parallel_output)
        return logits


class BloomPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT.
    It calculates the final loss.
    """

    def __init__(self, ignore_index=-100, tensor_parallel_degree=1, tensor_parallel_output=False):
        super(BloomPretrainingCriterion, self).__init__()
        if tensor_parallel_degree > 1 and tensor_parallel_output:
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        self.ignore_index = ignore_index

    def forward(self, prediction_scores, masked_lm_labels, loss_mask=None):
        masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))
        with paddle.amp.auto_cast(False):
            masked_lm_loss = masked_lm_loss.astype("float32")
            if loss_mask is not None:
                loss_mask = loss_mask.reshape([-1])
                masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
                loss = masked_lm_loss / loss_mask.sum()
            else:
                masked_lm_loss = masked_lm_loss[masked_lm_labels != self.ignore_index]
                loss = paddle.mean(masked_lm_loss)

        return loss


class BloomForPretraining(BloomPreTrainedModel):
    """
    The pretraining model of Bloom.
    It returns some logits and cached_kvs.
    """

    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.bloom = BloomModel(config)
        self.criterion = BloomPretrainingCriterion(tensor_parallel_degree=config.tensor_parallel_degree)
        self.extra_parameters = [self.bloom.word_embeddings.weight]

    def forward(
        self,
        input_ids,
        labels=None,
        loss_mask=None,
        attention_mask=None,
        use_cache=False,
        cache=None,
    ):
        outputs = self.bloom(input_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        logits = parallel_matmul(
            encoder_outputs[0],
            self.bloom.word_embeddings.weight,
            parallel_output=False,
        )
        if labels is None:
            return logits

        loss = self.criterion(logits, labels, loss_mask)
        return loss, logits


class BloomForCausalLM(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.decoder_weight"]
    _keys_to_ignore_on_save = [r"lm_head.decoder_weight"]
    _tied_weights_keys = ["lm_head.decoder_weight"]

    def __init__(self, config):
        super().__init__(config)
        self.bloom = BloomModel(config)
        self.lm_head = BloomLMHead(config, self.bloom.word_embeddings.weight)
        self.criterion = BloomPretrainingCriterion(
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_output=True,
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple):
            model_kwargs["cache"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["cache"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                if len(attention_mask.shape) == 2:
                    model_kwargs["attention_mask"] = paddle.concat(
                        [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)],
                        axis=-1,
                    )
                elif len(attention_mask.shape) == 4:
                    model_kwargs["attention_mask"] = paddle.concat(
                        [attention_mask, paddle.ones([*attention_mask.shape[:3], 1], dtype=attention_mask.dtype)],
                        axis=-1,
                    )[:, :, -1:, :]
        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, cache=None, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        attention_mask = kwargs.get("attention_mask", None)
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "cache": cache, "use_cache": True}

    # TODO(wawltor) attention_mask is not need
    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        attention_mask = paddle.ones_like(input_ids, dtype="bool")
        attention_mask = (input_ids != pad_token_id).astype("bool")
        return attention_mask

    def forward(
        self,
        input_ids=None,
        cache=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.bloom(
            input_ids,
            past_key_values=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states, self.config.tensor_parallel_output)

        loss = None
        if labels is not None:
            loss = self.criterion(lm_logits, labels)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[Tensor]], beam_idx: Tensor) -> Tuple[Tuple[Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(tuple(past_state.index_select(0, beam_idx) for past_state in layer_past) for layer_past in past)


class BloomForSequenceClassification(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bloom = BloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias_attr=False)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.bloom(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
            sequence_length = input_ids.shape[1]
        else:
            batch_size = inputs_embeds.shape[0]
            sequence_length = inputs_embeds.shape[1]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            pooled_logits = logits[:, -1]
        else:
            if input_ids is not None:
                # select the last word of batch sentence
                sequence_lengths = paddle.where(input_ids != self.config.pad_token_id, 1, 0).sum(axis=-1) - 1
                sequence_lengths += paddle.to_tensor([i * input_ids.shape[1] for i in range(batch_size)])
                pooled_logits = paddle.index_select(
                    logits.reshape([batch_size * sequence_length, -1]), sequence_lengths, axis=0
                )

            else:
                pooled_logits = logits[:, -1]
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype == paddle.int64:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class BloomForTokenClassification(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bloom = BloomModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[Tensor], TokenClassifierOutput]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.bloom(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class BloomForGeneration(BloomPreTrainedModel):
    """
    Bloom Model with pretraining tasks on top.

    Args:
        bloom (:class:`BloomModel`):
            An instance of :class:`BloomModel`.

    """

    def __init__(self, config: BloomConfig):
        # when running generation, it must be True
        config.use_cache = True

        super(BloomForGeneration, self).__init__(config)
        self.bloom = BloomModel(config)
        self.config = config

        self.max_length = self.config.get("max_dec_len", 20)
        self.min_length = self.config.get("min_dec_len", 0)
        self.decode_strategy = self.config.get("decode_strategy", "sampling")
        self.temperature = self.config.get("temperature", 1.0)
        self.top_k = self.config.get("top_k", 0)
        self.top_p = self.config.get("top_p", 1.0)
        self.use_topp_sampling = self.config.get("use_topp_sampling", False)
        self.inference = self.config.get("inference", False)
        self.repetition_penalty = self.config.get("repetition_penalty", 1.0)
        self.num_beams = self.config.get("num_beams", 1)
        self.num_beam_groups = self.config.get("num_beam_groups", 1)
        self.length_penalty = self.config.get("length_penalty", 0.0)
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
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("bool")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="bool")
        return attention_mask

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

        # if min_length is not None and eos_token_id is not None and min_length > -1:
        #     processors.append(
        #         MinLengthLogitsProcessor(min_length, eos_token_id))

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
        attention_mask = kwargs.get("attention_mask", None)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "cache": cache}

    def update_model_kwargs_for_generation(self, next_tokens, outputs, model_kwargs, is_encoder_decoder=False):
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

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="bool")], axis=-1
                )

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

        batch_size, cur_len = paddle.shape(input_ids)

        # used for compute on gpu, avoid memcpy D2H
        cur_len_gpu = paddle.full([1], cur_len)

        origin_len = paddle.shape(input_ids)[1]
        # used for compute on gpu, avoid memcpy D2H
        origin_len_gpu = paddle.full([1], origin_len)

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
            return self.bloom(**model_inputs, **immutable)

        def _post_process_(outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs):

            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # logits = paddle.matmul(
            #     logits,
            #     self.bloom.embeddings.word_embeddings.weight,
            #     transpose_y=True)

            # x_dims_mapping = [self.bloom.mesh.dp] + [
            #     None for i in range(len(logits.shape) - 1)
            # ]
            # w_dims_mapping = [self.bloom.mesh.mp, None]
            # matmul = auto.shard_op(paddle.matmul, self.bloom.mesh[-1],
            #                        [x_dims_mapping, w_dims_mapping, None])

            logits = paddle.matmul(logits, self.bloom.word_embeddings.weight, transpose_y=True)

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
                if self.use_topp_sampling:
                    try:
                        from ppfleetx_ops import topp_sampling
                    except ImportError:
                        raise ImportError(
                            "please install ppfleetx_ops by 'cd ppfleetx/ops && python setup_cuda.py install'!"
                        )
                    top_ps_tensor = paddle.full(shape=[paddle.shape(probs)[0]], fill_value=top_p, dtype=probs.dtype)
                    next_tokens = topp_sampling(probs, top_ps_tensor)
                else:
                    probs = TopPProcess(probs, top_p, min_tokens_to_keep)

            if not self.use_topp_sampling:
                # TODO(wj-Mcat): multinomial do not support fp16, so convert it to fp32
                # refer to: https://github.com/PaddlePaddle/Paddle/issues/51852
                next_tokens = paddle.multinomial(paddle.cast(probs, paddle.float32))
                # next_tokens = paddle.multinomial(probs)

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
        max_length = paddle.to_tensor(max_length)
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

        bos_token_id = bos_token_id if bos_token_id is not None else getattr(self.config, "bos_token_id", None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self.config, "eos_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(self.config, "pad_token_id", None)
        forced_bos_token_id = (
            forced_bos_token_id
            if forced_bos_token_id is not None
            else getattr(self.config, "forced_bos_token_id", None)
        )
        forced_eos_token_id = (
            forced_eos_token_id
            if forced_eos_token_id is not None
            else getattr(self.config, "forced_eos_token_id", None)
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else getattr(self.config, "decoder_start_token_id", None)
        )

        # params check
        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
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
            min_len = int(input_ids.shape[-1])
            max_len = int(input_ids.shape[-1])
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
