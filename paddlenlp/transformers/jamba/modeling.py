# coding=utf-8
# Copyright 2024 AI21 Labs Ltd. and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Paddle Jamba model."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)

from ...utils.initializer import normal_, zeros_
from ..activations import ACT2FN
from ..model_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPast
from ..model_utils import PretrainedModel
from .configuration import JambaConfig

try:
    from mamba_ssm_paddle.ops.selective_scan_interface import (
        mamba_inner_fn,
        selective_scan_fn,
    )
    from mamba_ssm_paddle.ops.triton.selective_state_update import (
        selective_state_update,
    )
except ImportError:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

try:
    from mamba_ssm_paddle.ops.causal_conv1d_interface import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)
from paddle.amp.auto_cast import amp_global_state

from paddlenlp.utils.log import logger

from ..llama.modeling import parallel_matmul

_flash_supports_window_size = False

_CONFIG_FOR_DOC = "JambaConfig"


def is_autocast_enabled():
    tracer = paddle.framework._dygraph_tracer()
    return False if tracer._amp_level == paddle.core.AmpLevel.O0 else True


def get_triangle_upper_mask(x, mask=None):
    if mask is not None:
        return mask
    # [bsz, n_head, q_len, kv_seq_len]
    shape = x.shape
    #  [bsz, 1, q_len, kv_seq_len]
    shape[1] = 1
    mask = paddle.full(shape, paddle.finfo(x.dtype).min, dtype=x.dtype)
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all()


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


# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func with gate->router
def load_balancing_loss_func(
    router_logits: paddle.Tensor,
    num_experts: paddle.Tensor = None,
    top_k=2,
    attention_mask: Optional[paddle.Tensor] = None,
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Paddle.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_logits (Union[`paddle.Tensor`, Tuple[paddle.Tensor]):
            Logits from the `router`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`paddle.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if router_logits is None or not isinstance(router_logits, tuple):
        return 0

    if isinstance(router_logits, tuple):
        concatenated_router_logits = paddle.concat([layer_router for layer_router in router_logits], axis=0)

    routing_weights = paddle.nn.functional.softmax(concatenated_router_logits, axis=-1)

    _, selected_experts = paddle.topk(routing_weights, top_k, axis=-1)

    expert_mask = paddle.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None or attention_mask.ndim == 4:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = paddle.mean(expert_mask.cast("float32"), axis=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = paddle.mean(routing_weights, axis=0)
    else:
        if attention_mask.ndim == 2:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_router_logits.shape[0] // (batch_size * sequence_length)
            if attention_mask.dtype == paddle.bool:
                attention_mask = attention_mask.cast("float32")
            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = (
                attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape([-1, top_k, num_experts])
            )

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = paddle.sum(expert_mask.cast("float32") * expert_attention_mask, axis=0) / paddle.sum(
                expert_attention_mask, axis=0
            )

            # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
            router_per_expert_attention_mask = (
                attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape([-1, num_experts])
            )

            # Compute the average probability of routing to these experts
            router_prob_per_expert = paddle.sum(
                routing_weights * router_per_expert_attention_mask, axis=0
            ) / paddle.sum(router_per_expert_attention_mask, axis=0)

    overall_loss = paddle.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Jamba
class JambaRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        JambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = self.create_parameter(
            [
                hidden_size,
            ],
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.cast(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.cast(input_dtype)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(x, axis=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


@dataclass
class HybridMambaAttentionDynamicCache:
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(self, config, batch_size, dtype=paddle.float16):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
        conv_kernel_size = config.mamba_d_conv
        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "mamba":
                self.conv_states += [paddle.zeros([batch_size, intermediate_size, conv_kernel_size], dtype=dtype)]
                self.ssm_states += [paddle.zeros([batch_size, intermediate_size, ssm_state_size], dtype=dtype)]
            else:
                self.conv_states += [paddle.to_tensor([[]] * batch_size)]
                self.ssm_states += [paddle.to_tensor([[]] * batch_size)]
                self.transformer_layers.append(i)

        self.key_cache = [paddle.to_tensor([[]] * batch_size) for _ in range(config.num_hidden_layers)]
        self.value_cache = [paddle.to_tensor([[]] * batch_size) for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: paddle.Tensor,
        value_states: paddle.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # bsz, num_key_value_heads, q_len, self.head_dim
            self.key_cache[layer_idx] = paddle.concat([self.key_cache[layer_idx], key_states], axis=2)
            self.value_cache[layer_idx] = paddle.concat([self.value_cache[layer_idx], value_states], axis=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: paddle.Tensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0)
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0)
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        key_val = self.key_cache[layer_idx]
        if key_val.ndim == 2 and key_val.shape[-1] == 0:
            return 0
        return key_val.shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def __getitem__(self, layer_idx: int) -> List[Tuple[paddle.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Jamba
class JambaAttention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: JambaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if config.tensor_parallel_degree > 1:
            assert (
                self.num_heads % config.tensor_parallel_degree == 0
            ), f"num_heads: {self.num_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_heads = self.num_heads // config.tensor_parallel_degree

            assert (
                self.num_key_value_heads % config.tensor_parallel_degree == 0
            ), f"num_key_value_heads: {self.num_key_value_heads}, tensor_parallel_degree: {config.tensor_parallel_degree}"
            self.num_key_value_heads = self.num_key_value_heads // config.tensor_parallel_degree

            ColumnParallelLinear = mpu.ColumnParallelLinear
            RowParallelLinear = mpu.RowParallelLinear
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                gather_output=False,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                has_bias=False,
                gather_output=False,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                has_bias=False,
                gather_output=False,
            )
            self.o_proj = RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=False)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = paddle.matmul(query_states, key_states, transpose_y=True) / math.sqrt(self.head_dim)

        if attention_mask is None:
            attention_mask = get_triangle_upper_mask(attn_weights)

        # [bs, num_heads, kv_seq_len, head_dim]
        kv_seq_len = value_states.shape[2]

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])
        if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )
        attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32).cast(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(attn_weights, value_states)

        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3]).contiguous()
        attn_output = attn_output.reshape([bsz, q_len, -1])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Adapted from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with Mistral->Jamba
class JambaFlashAttention2(JambaAttention):
    """
    Jamba flash attention module. This module inherits from `JambaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == paddle.float32:
            if is_autocast_enabled():
                target_dtype = amp_global_state().amp_dtype
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.cast(target_dtype)
            key_states = key_states.cast(target_dtype)
            value_states = value_states.cast(target_dtype)

        # Reashape to the expected shape for Flash Attention
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            dropout_p=dropout_rate,
            training=self.training,
        )
        attn_output = attn_output.reshape([bsz, q_len, -1]).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class JambaMambaMixer(nn.Layer):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: JambaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1D(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias_attr=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.use_fast_kernels = config.use_mamba_kernels and is_fast_path_available

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias_attr=self.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias_attr=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias_attr=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = paddle.arange(1, self.ssm_state_size + 1, dtype=paddle.float32)[None, :]
        A = A.expand([self.intermediate_size, -1])

        self.A_log = self.create_parameter(
            shape=A.shape,
            default_initializer=nn.initializer.Assign(paddle.log(A)),
        )
        self.D = self.create_parameter(
            shape=[
                self.intermediate_size,
            ],
            default_initializer=nn.initializer.Constant(1),
        )
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=self.use_bias)

        self.dt_layernorm = JambaRMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.b_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config"
            )

    def cuda_kernels_forward(self, hidden_states: paddle.Tensor, cache: HybridMambaAttentionDynamicCache = None):
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache is not None
            and cache.has_previous_state
            and seq_len == 1
            and cache.conv_states[self.layer_idx].shape[0] == cache.ssm_states[self.layer_idx].shape[0] == batch_size
        )
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose([0, 2, 1])

        # We can't use `mamba_inner_fn` even if in training and without cache params because we have the
        # inner layernorms which isn't supported by this fused kernel
        hidden_states, gate = projected_states.chunk(2, axis=1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.reshape([self.conv1d.weight.shape[0], self.conv1d.weight.shape[2]])
        if use_precomputed_states:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache.conv_states[self.layer_idx],
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache is not None:
                conv_states = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0),
                    data_format="NCL",
                )
                cache.conv_states[self.layer_idx].copy_(conv_states.cast(cache.dtype), False)
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose([0, 2, 1]))
        time_step, B, C = paddle.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], axis=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        # Here we need to apply dt_proj without the bias, as the bias is added in the selective scan kernel.
        # This is a hack to apply dt_proj while still using the forward pass of `paddle.nn.Linear`, which is needed
        # in order to make quantization work. Quantization code replaces `paddle.nn.Linear` layers with quantized
        # linear layers, and requires to call the forward pass directly.
        # The original code here was: ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
        time_proj_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        discrete_time_step = self.dt_proj(time_step).transpose([0, 2, 1])
        self.dt_proj.bias = time_proj_bias

        A = -paddle.exp(self.A_log.cast("float32"))
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = time_proj_bias.cast("float32") if time_proj_bias is not None else None
        if use_precomputed_states:
            scan_outputs = selective_state_update(
                cache.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose([0, 2, 1]),
                C.transpose([0, 2, 1]),
                self.D.cast("float32"),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache is not None:
                cache.ssm_states[self.layer_idx].copy_(ssm_state.cast(cache.dtype), False)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose([0, 2, 1]))

        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache: HybridMambaAttentionDynamicCache = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose([0, 2, 1])                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, axis=1)

        use_cache = isinstance(cache, HybridMambaAttentionDynamicCache)
        # 2. Convolution sequence transformation
        if use_cache and cache.ssm_states[self.layer_idx].shape[0] == batch_size:
            if self.training:
                # In training mode, we don't want to perform in-place operations on ssm_state so we can compute the backwards pass
                ssm_state = cache.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache.ssm_states[self.layer_idx]

            if cache.has_previous_state and seq_len == 1 and \
                    cache.conv_states[self.layer_idx].shape[0] == batch_size:
                conv_state = cache.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = paddle.roll(conv_state, shifts=-1, axis=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache.conv_states[self.layer_idx] = conv_state
                hidden_states = paddle.sum(conv_state * self.conv1d.weight[:, 0, :], axis=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).cast(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0),
                    data_format="NCL",
                )
                cache.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = paddle.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                dtype=dtype,
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose([0, 2, 1]))
        time_step, B, C = paddle.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], axis=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        discrete_time_step = self.dt_proj(time_step)                                              # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose([0, 2, 1])      # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -paddle.exp(self.A_log.cast("float32"))                                              # [intermediate_size, ssm_state_size]
        discrete_A = paddle.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])         # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].cast("float32")        # [batch, intermediate_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].cast("float32")
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediate_size, ssm_state]
            scan_output = paddle.matmul(ssm_state.cast(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediate_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = paddle.stack(scan_outputs, axis=-1)                                # [batch, intermediate_size, seq_len]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if use_cache:
            cache.ssm_states[self.layer_idx] = ssm_state

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose([0, 2, 1]))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(self, hidden_states, cache: HybridMambaAttentionDynamicCache = None):
        if self.use_fast_kernels:
            if not is_fast_path_available:
                raise ValueError(
                    "Fast Mamba kernels are not available. Make sure to they are installed and that the mamba module is on a CUDA device"
                )
            return self.cuda_kernels_forward(hidden_states, cache)
        return self.slow_forward(hidden_states, cache)


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Jamba
class JambaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if config.tensor_parallel_degree > 1:
            ColumnParallelLinear = mpu.ColumnParallelLinear
            RowParallelLinear = mpu.RowParallelLinear
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeMLPForwardBackward(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        assert not x.stop_gradient, "x should not be stop_gradient"
        ctx.shape_list = [x.shape, gate_weight.shape, up_weight.shape, down_weight.shape]
        ctx.dtype_list = [x.dtype, gate_weight.dtype, up_weight.dtype, down_weight.dtype]
        return paddle.zeros_like(x)

    @staticmethod
    def backward(ctx, grad):
        return tuple(paddle.zeros(shape, dtype=dtype) for shape, dtype in zip(ctx.shape_list, ctx.dtype_list))


# Adapted from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock with Mistral->Jamba
class JambaSparseMoeBlock(nn.Layer):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config: JambaConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias_attr=False)
        self.experts = nn.LayerList([JambaMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.reshape([-1, hidden_dim])
        # router_logits: [batch_size * seq_len, num_experts]
        router_logits = self.router(hidden_states)

        with paddle.amp.auto_cast(False):
            routing_weights = F.softmax(router_logits.astype("float32"), axis=1)
        routing_weights, selected_experts = paddle.topk(routing_weights, self.top_k, axis=-1)
        # we cast back to input dtype
        routing_weights = routing_weights.cast(hidden_states.dtype)

        final_hidden_states = paddle.zeros(
            [batch_size * sequence_length, hidden_dim],
            dtype=hidden_states.dtype,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated.
        # shape: [num_experts, top_k, batch_size * seq_len]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).transpose([2, 1, 0])

        # NOTE: we need to do some fake gradient for sharding parallel training.
        try:
            hcg = fleet.get_hybrid_communicate_group()
            sharding_parallel_world_size = hcg.get_sharding_parallel_world_size()
            if sharding_parallel_world_size > 1 and self.training:
                logger.warning_once(
                    f"Sharding parallel world size is {sharding_parallel_world_size}, we need to do some fake gradient."
                )
                for expert_id in range(self.num_experts):
                    expert_layer = self.experts[expert_id]
                    final_hidden_states += (
                        FakeMLPForwardBackward.apply(
                            hidden_states,
                            expert_layer.gate_proj.weight,
                            expert_layer.up_proj.weight,
                            expert_layer.down_proj.weight,
                        )
                        * routing_weights[0, 0]
                    )
        except:
            pass

        # Loop over all available experts in the model and perform the computation on each expert.
        for expert_id in range(self.num_experts):
            expert_layer = self.experts[expert_id]
            idx, top_x = paddle.where(expert_mask[expert_id])

            if top_x.shape[0] == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = paddle.gather(hidden_states, top_x.squeeze(-1))
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx]

            top_x = top_x.squeeze()
            if top_x.shape == []:
                top_x = paddle.to_tensor([top_x.item()])
            final_hidden_states.index_add_(top_x, 0, current_hidden_states.cast(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape([batch_size, sequence_length, hidden_dim])
        return final_hidden_states, router_logits


class JambaAttentionDecoderLayer(nn.Layer):
    def __init__(self, config: JambaConfig, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx]
        if config.use_flash_attention:
            self.self_attn = JambaFlashAttention2(config, layer_idx)
        else:
            self.self_attn = JambaAttention(config, layer_idx)

        ffn_layer_class = JambaSparseMoeBlock if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`HybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`paddle.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # residual connection after attention
        hidden_states = residual + hidden_states

        # feed-forward (experts/MLP)
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, router_logits = ff_outputs
        else:
            hidden_states, router_logits = ff_outputs, None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class JambaMambaDecoderLayer(nn.Layer):
    def __init__(self, config: JambaConfig, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx]
        self.mamba = JambaMambaMixer(config=config, layer_idx=layer_idx)

        ffn_layer_class = JambaSparseMoeBlock if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`HybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`paddle.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache=past_key_value,
        )
        self_attn_weights = None

        # residual connection after mamba
        hidden_states = residual + hidden_states

        # feed-forward (experts/MLP)
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, router_logits = ff_outputs
        else:
            hidden_states, router_logits = ff_outputs, None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class JambaPretrainedModel(PretrainedModel):
    config_class = JambaConfig
    base_model_prefix = "jamba"
    supports_gradient_checkpointing = True
    _no_split_modules = ["JambaAttentionDecoderLayer", "JambaMambaDecoderLayer"]

    @classmethod
    def _get_name_mappings(cls, config: JambaConfig) -> List[StateDictNameMapping]:
        mappings: List[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["final_layernorm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_type_name = config.layers_block_type[layer_index]

            if layer_type_name == "mamba":
                layer_mappings = [
                    [f"layers.{layer_index}.mamba.A_log"],
                    [f"layers.{layer_index}.mamba.D"],
                    [f"layers.{layer_index}.mamba.conv1d.weight"],
                    [f"layers.{layer_index}.mamba.conv1d.bias"],
                    # linear
                    [f"layers.{layer_index}.mamba.in_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mamba.x_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mamba.dt_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mamba.dt_proj.bias"],
                    [f"layers.{layer_index}.mamba.out_proj.weight", None, "transpose"],
                    # layernorm
                    [f"layers.{layer_index}.mamba.dt_layernorm.weight"],
                    [f"layers.{layer_index}.mamba.b_layernorm.weight"],
                    [f"layers.{layer_index}.mamba.c_layernorm.weight"],
                ]
                if config.mamba_proj_bias:
                    layer_mappings.extend(
                        [
                            [f"layers.{layer_index}.mamba.in_proj.bias"],
                            [f"layers.{layer_index}.mamba.out_proj.bias"],
                        ]
                    )
            elif layer_type_name == "attention":
                layer_mappings = [
                    [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.k_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.v_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                ]
            else:
                raise ValueError(f"{layer_type_name} is not a valid layer type.")

            num_experts = config.layers_num_experts[layer_index]
            if num_experts > 1:
                layer_mappings.append([f"layers.{layer_index}.feed_forward.router.weight", None, "transpose"])
            for expert_idx in range(num_experts):
                expert_tag = f"experts.{expert_idx}." if num_experts > 1 else ""
                layer_mappings.extend(
                    [
                        [f"layers.{layer_index}.feed_forward.{expert_tag}gate_proj.weight", None, "transpose"],
                        [f"layers.{layer_index}.feed_forward.{expert_tag}up_proj.weight", None, "transpose"],
                        [f"layers.{layer_index}.feed_forward.{expert_tag}down_proj.weight", None, "transpose"],
                    ]
                )
            layer_mappings.extend(
                [
                    [f"layers.{layer_index}.input_layernorm.weight"],
                    [f"layers.{layer_index}.pre_ff_layernorm.weight"],
                ]
            )

            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)
        # base-model prefix "JambaModel"
        if "JambaModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "jamba." + mapping[1]
            if not config.tie_word_embeddings:
                model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: JambaConfig, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(config: JambaConfig):
            final_actions = {
                # Column Linear
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
            }

            if not config.vocab_size % config.tensor_parallel_degree == 0:
                final_actions.pop("lm_head.weight")
                final_actions.pop("embed_tokens.weight")

            for layer_index in range(config.num_hidden_layers):
                layer_type_name = config.layers_block_type[layer_index]
                if layer_type_name == "mamba":
                    # NO TP
                    pass
                elif layer_type_name == "attention":
                    # Column Linear
                    final_actions[f"layers.{layer_index}.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                    # if we have enough num_key_value_heads to split, then split it.
                    if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                        final_actions[f"layers.{layer_index}.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                        final_actions[f"layers.{layer_index}.self_attn.v_proj.weight"] = partial(fn, is_column=True)

                    # Row Linear
                    final_actions[f"layers.{layer_index}.self_attn.o_proj.weight"] = partial(fn, is_column=False)
                else:
                    raise ValueError(f"{layer_type_name} is not a valid layer type.")

                num_experts = config.layers_num_experts[layer_index]
                for expert_idx in range(num_experts):
                    expert_tag = f"experts.{expert_idx}." if num_experts > 1 else ""
                    # Column Linear
                    final_actions[f"layers.{layer_index}.feed_forward.{expert_tag}gate_proj.weight"] = partial(
                        fn, is_column=True
                    )
                    final_actions[f"layers.{layer_index}.feed_forward.{expert_tag}up_proj.weight"] = partial(
                        fn, is_column=True
                    )
                    # Row Linear
                    final_actions[f"layers.{layer_index}.feed_forward.{expert_tag}down_proj.weight"] = partial(
                        fn, is_column=False
                    )

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config)
        return mappings

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()

    @paddle.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        if isinstance(
            module,
            (
                nn.Linear,
                nn.Conv1D,
                nn.Embedding,
                mpu.VocabParallelEmbedding,
                mpu.ColumnParallelLinear,
                mpu.RowParallelLinear,
            ),
        ):
            if isinstance(module.weight, paddle.Tensor):
                if module.weight.is_distributed:
                    with rng_tracker():
                        normal_(module.weight, mean=0.0, std=std)
                else:
                    normal_(module.weight, mean=0.0, std=std)

            if isinstance(module, (nn.Linear, nn.Conv1D)):
                if module.bias is not None:
                    zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and hasattr(module, "padding_idx"):
                module.weight[module.padding_idx] = 0.0


ALL_DECODER_LAYER_TYPES = {"attention": JambaAttentionDecoderLayer, "mamba": JambaMambaDecoderLayer}


# Adapted from transformers.models.mistral.modeling_mistral.MistralModel with MISTRAL->JAMBA, Mistral->Jamba
class JambaModel(JambaPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`JambaDecoderLayer`]

    Args:
        config: JambaConfig
    """

    def __init__(self, config: JambaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # new added
        if config.tensor_parallel_degree > 1 and config.sequence_parallel:
            logger.warning_once("Currently we donot support sequence parallelism yet!")
        self.recompute_granularity = config.recompute_granularity
        self.no_recompute_layers = config.no_recompute_layers if config.no_recompute_layers is not None else []

        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            self.embed_tokens = mpu.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
            )

        self.embed_tokens.padding_idx = self.padding_idx

        decoder_layers = []
        for i in range(config.num_hidden_layers):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[i]]
            decoder_layers.append(layer_class(config, layer_idx=i))
        self.layers = nn.LayerList(decoder_layers)

        self.final_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.enable_recompute = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(expanded_attn_mask, 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids: paddle.Tensor = None,
        past_key_values: HybridMambaAttentionDynamicCache = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        use_cache: bool = False,
        cache_position: paddle.Tensor = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            output_router_logits,
            use_cache,
            cache_position,
            use_reentrant=self.config.recompute_use_reentrant,
        )

        return hidden_states

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, MoEModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.enable_recompute and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with recompute. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            logger.warning_once(
                "Jamba requires an initialized `HybridMambaAttentionDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )
        if not use_cache and past_key_values is not None:
            past_key_values = None

        batch_size, seq_length = inputs_embeds.shape[:2]
        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values is not None:
            cache_length = past_key_values.get_seq_length()
            seq_length_with_past += cache_length

        # embed positions
        if attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
        )  # [bs, 1, seq_len, seq_len]
        if self.config.use_flash_attention:
            is_casual = is_casual_mask(attention_mask)
            if is_casual:
                attention_mask = None

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            has_gradient = not hidden_states.stop_gradient
            if (
                self.enable_recompute
                and idx not in self.no_recompute_layers
                and has_gradient
                and self.recompute_granularity == "full"
            ):
                layer_outputs = self.recompute_training_full(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                if layer_outputs[1] is not None:
                    # append attentions only of attention layers. Mamba layers return `None` as the attention weights
                    all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                if layer_outputs[-1] is not None:
                    # append router logits only of expert layers. Regular MLP layers return `None` as the router logits
                    all_router_logits += (layer_outputs[-1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None if not use_cache else past_key_values

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoEModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class JambaPretrainingCriterion(nn.Layer):
    """
    Criterion for Jamba.
    It calculates the final loss.
    """

    def __init__(self, config: JambaConfig):

        super().__init__()
        self.ignore_index = getattr(config, "ignore_index", -100)
        self.config = config
        self.enable_parallel_cross_entropy = (
            config.tensor_parallel_degree > 1
            and config.vocab_size % config.tensor_parallel_degree == 0
            and config.tensor_parallel_output
        )

        if self.enable_parallel_cross_entropy:  # and False: # and lm_head is distributed
            self.loss_func = mpu.ParallelCrossEntropy(ignore_index=self.ignore_index)
        else:
            self.loss_func = nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

    def forward(self, prediction_scores, masked_lm_labels):
        if self.enable_parallel_cross_entropy:
            if prediction_scores.shape[-1] == self.config.vocab_size:
                logger.warning_once(
                    f"enable_parallel_cross_entropy, the vocab_size should be splited: {prediction_scores.shape[-1]}, {self.config.vocab_size}"
                )
                self.loss_func = nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)

        with paddle.amp.auto_cast(False):
            masked_lm_loss = self.loss_func(prediction_scores.astype("float32"), masked_lm_labels.unsqueeze(2))
            # skip ignore_index which loss == 0
            # masked_lm_loss = masked_lm_loss[masked_lm_loss > 0]
            # loss = paddle.mean(masked_lm_loss)
            binary_sequence = paddle.where(
                masked_lm_loss > 0, paddle.ones_like(masked_lm_loss), paddle.zeros_like(masked_lm_loss)
            )
            count = paddle.sum(binary_sequence)
            if count == 0:
                loss = paddle.sum(masked_lm_loss * binary_sequence)
            else:
                loss = paddle.sum(masked_lm_loss * binary_sequence) / count

        return loss


class JambaLMHead(nn.Layer):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.config = config
        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        if vocab_size != config.vocab_size:
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[config.hidden_size, vocab_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:
            self.weight = self.create_parameter(
                shape=[config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
            )
        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
        if self.weight.is_distributed:
            self.weight.split_axis = 1

    def forward(self, hidden_states, tensor_parallel_output=None):
        # if self.config.sequence_parallel:
        #     hidden_states = GatherOp.apply(hidden_states)
        #     seq_length = self.config.seq_length
        #     hidden_states = paddle.reshape_(hidden_states, [-1, seq_length, self.config.hidden_size])

        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1

        logits = parallel_matmul(hidden_states, self.weight, tensor_parallel_output=tensor_parallel_output)
        return logits


# Adapted from transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM with MIXTRAL->JAMBA, Mixtral->Jamba
class JambaForCausalLM(JambaPretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: JambaConfig):
        super().__init__(config)

        self.jamba = JambaModel(config)
        assert not config.tie_word_embeddings, "Tied word embeddings are not supported in JambaForCausalLM"
        self.lm_head = JambaLMHead(config)
        self.criterion = JambaPretrainingCriterion(config)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.jamba.embed_tokens

    def set_input_embeddings(self, value):
        self.jamba.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.jamba = decoder

    def get_decoder(self):
        return self.jamba

    # Ignore copy
    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        num_logits_to_keep: Optional[Union[int, None]] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int` or `None`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `None`, calculate logits for all
                `input_ids`. Only last token logits are needed for generation, and calculating them only for that token
                can save memory, which becomes pretty significant for long sequences.

        Returns:

        Example:

        ```python
        >>> from paddlenlp.transformers import JambaTokenizer, JambaForCausalLM

        >>> model = JambaForCausalLM.from_pretrained("ai21labs/Jamba-v0.1")
        >>> tokenizer = JambaTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pd")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.jamba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # [bs, seq_len, dim]

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is togather with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output and labels is not None and self.config.tensor_parallel_degree > 1
        )

        if num_logits_to_keep is None:
            logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)
        else:
            logits = self.lm_head(
                hidden_states[..., -num_logits_to_keep:, :], tensor_parallel_output=tensor_parallel_output
            )

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        empty_past_kv = past_key_values is None

        # Omit tokens covered by past_key_values
        if not empty_past_kv:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config,
                input_ids.shape[0],
                self.get_input_embeddings().weight.dtype,
            )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases
        model_inputs.update(
            {
                "position_ids": None,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                "num_logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": None,
            }
        )
        return model_inputs

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, MoECausalLMOutputWithPast) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.concat(
                [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)], axis=-1
            )

        return model_kwargs

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype(paddle.int64)
        else:
            attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)
        return attention_mask


# Copied from transformers.models.mixtral.modeling_mixtral.MixtralForSequenceClassification with Mixtral->Jamba, MIXTRAL->JAMBA
# class JambaForSequenceClassification(JambaPretrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.jamba = JambaModel(config)
#         self.score = nn.Linear(config.hidden_size, self.num_labels, bias_attr=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.jamba.embed_tokens

#     def set_input_embeddings(self, value):
#         self.jamba.embed_tokens = value

#     def forward(
#         self,
#         input_ids: paddle.Tensor = None,
#         attention_mask: Optional[paddle.Tensor] = None,
#         position_ids: Optional[paddle.Tensor] = None,
#         past_key_values: Optional[Union[HybridMambaAttentionDynamicCache, List[paddle.Tensor]]] = None,
#         inputs_embeds: Optional[paddle.Tensor] = None,
#         labels: Optional[paddle.Tensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
#         r"""
#         labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.jamba(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size = input_ids.shape[0]
#         else:
#             batch_size = inputs_embeds.shape[0]

#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#         if self.config.pad_token_id is None:
#             sequence_lengths = -1
#         else:
#             if input_ids is not None:
#                 # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
#                 sequence_lengths = paddle.equal(input_ids, self.config.pad_token_id).cast("int32").argmax(-1) - 1
#                 sequence_lengths = sequence_lengths % input_ids.shape[-1]
#             else:
#                 sequence_lengths = -1

#         pooled_logits = logits[paddle.arange(batch_size), sequence_lengths]

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(pooled_logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(
#                     pooled_logits.reshape([-1, self.num_labels]),
#                     labels.reshape(
#                         [
#                             -1,
#                         ]
#                     ),
#                 )
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(pooled_logits, labels)
#         if not return_dict:
#             output = (pooled_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )
