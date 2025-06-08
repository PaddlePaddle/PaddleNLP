# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Paddle implementation of the Phi3 model."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        GatherOp,
        ScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except ImportError:
    # For older Paddle versions that might not have these
    class ScatterOp:
        @staticmethod
        def apply(x):
            return x

    class GatherOp:
        @staticmethod
        def apply(x):
            return x

    def mark_as_sequence_parallel_parameter(x):
        pass


from paddlenlp.transformers import linear_utils  # Added
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from .configuration import Phi3Config

# from paddlenlp.utils.log import logger # Removed logger import


__all__ = [
    "Phi3Model",
    "Phi3PreTrainedModel",
    "Phi3ForCausalLM",
]


def _make_causal_mask(
    input_ids_shape: Tuple[int, int],
    past_key_values_length: int,
) -> paddle.Tensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = paddle.full((target_length, target_length + past_key_values_length), float("-inf"))
    mask_cond = paddle.arange(mask.shape[-1])
    mask.stop_gradient = True
    mask[paddle.arange(mask.shape[0]).unsqueeze(1) >= mask_cond] = 0

    if batch_size > 1:
        mask = mask.unsqueeze(0).expand([batch_size, target_length, target_length + past_key_values_length])

    return mask


def _expand_mask(mask: paddle.Tensor, tgt_length: int) -> paddle.Tensor:
    """
    Expands attention_mask from [batch_size, src_length] to [batch_size, 1, tgt_length, src_length].
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = mask.unsqueeze(1).expand([batch_size, 1, tgt_length, src_length])
    return expanded_mask


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    position_ids: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Apply rotary position embeddings to the query and key tensors."""
    # Handle a subset of position indices
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Phi3RMSNorm(nn.Layer):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class Phi3RotaryEmbedding(nn.Layer):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Create position embeddings
        inv_freq = 1.0 / (self.base ** (paddle.arange(0, self.dim, 2).astype("float32") / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

        # Create cos and sin embeddings
        t = paddle.arange(self.max_position_embeddings, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.register_buffer("cos_cached", paddle.cos(emb), persistable=False)
        self.register_buffer("sin_cached", paddle.sin(emb), persistable=False)

    def forward(self, x: paddle.Tensor, seq_len: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        return (
            self.cos_cached[:seq_len].astype(x.dtype),
            self.sin_cached[:seq_len].astype(x.dtype),
        )


class Phi3Attention(nn.Layer):
    """Multi-head attention with rotary position embeddings."""

    def __init__(self, config: Phi3Config, layerwise_recompute: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.tensor_parallel_degree = getattr(config, "tensor_parallel_degree", 1)
        # num_attention_heads and num_key_value_heads in config are global/total heads
        # We need to divide them by TP degree for local use in linear layers if TP is active
        self.num_attention_heads = config.num_attention_heads // self.tensor_parallel_degree
        self.num_key_value_heads = config.num_key_value_heads // self.tensor_parallel_degree

        self.head_dim = self.hidden_size // config.num_attention_heads  # head_dim uses global num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        self.layerwise_recompute = layerwise_recompute
        self.recompute_granularity = getattr(config, "recompute_granularity", "full")
        self.enable_recompute = False  # Controlled by DecoderLayer/Model
        self.recompute_use_reentrant = getattr(config, "recompute_use_reentrant", True)

        if (self.head_dim * config.num_attention_heads) != self.hidden_size:  # Check with global num_attention_heads
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size} "
                f"and `num_attention_heads`: {config.num_attention_heads})."
            )

        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads  # Global ratio

        # Determine correct Linear layer type
        if self.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        elif self.tensor_parallel_degree > 1:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear
        else:
            ColumnParallelLinear = nn.Linear
            RowParallelLinear = nn.Linear

        # Linear layers use local head counts (already divided by TP degree)
        q_inner_dim = self.num_attention_heads * self.head_dim
        kv_inner_dim = self.num_key_value_heads * self.head_dim

        self.q_proj = ColumnParallelLinear(self.hidden_size, q_inner_dim, bias_attr=True)
        self.k_proj = ColumnParallelLinear(self.hidden_size, kv_inner_dim, bias_attr=True)
        self.v_proj = ColumnParallelLinear(self.hidden_size, kv_inner_dim, bias_attr=True)
        # o_proj input dim is also based on local num_attention_heads
        self.o_proj = RowParallelLinear(
            q_inner_dim,
            self.hidden_size,
            bias_attr=True,
            input_is_parallel=(self.tensor_parallel_degree > 1 or self.sequence_parallel),
        )

        self.rotary_emb = Phi3RotaryEmbedding(
            self.head_dim,  # head_dim is universal
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta,
        )

        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        # batch_size might be needed if SP and hidden_states is [bs*seq/TP, dim]
        # For now, assume hidden_states is [bs, seq_dim_rank, dim] if SP, or [bs, seq_dim, dim] if not.
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:

        bsz, q_len, _ = hidden_states.shape  # If SP, q_len is seq_len_this_rank

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention. self.num_attention_heads and self.num_key_value_heads are TP-local.
        query_states = query_states.reshape([bsz, q_len, self.num_attention_heads, self.head_dim])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim])

        kv_seq_len = key_states.shape[1]  # This is q_len for current states
        if past_key_value is not None:
            # past_key_value[0] shape is [bsz, num_key_value_heads_local, past_kv_seq_len, head_dim]
            kv_seq_len += past_key_value[0].shape[2]

        # Compute rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Transpose for attention computation
        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        # Concatenate with past key/value states if provided
        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)

        # Save key/value states for future use if caching is enabled
        if use_cache:
            past_key_value = (key_states, value_states)

        # Handle Grouped Query Attention (GQA)
        # self.num_key_value_groups is global Q_heads / global_KV_heads
        # For local TP rank, if local_Q_heads / local_KV_heads > 1, then repeat.
        # This is equivalent to self.num_key_value_groups if Q and KV heads are divided by TP proportionally.
        # Or, more simply, local_q_heads / local_kv_heads.
        local_num_kv_groups = self.num_attention_heads // self.num_key_value_heads
        if local_num_kv_groups > 1:
            key_states = paddle.repeat_interleave(
                key_states, repeats=local_num_kv_groups, axis=1
            )  # Repeat along head dimension
            value_states = paddle.repeat_interleave(value_states, repeats=local_num_kv_groups, axis=1)

        # Core attention computation logic including recompute for "core_attn"
        has_gradient = not query_states.stop_gradient or not key_states.stop_gradient or not value_states.stop_gradient
        should_recompute_core_attn = (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "core_attn"
        )

        attn_output = None
        attn_weights_for_output = None  # Separate variable for clarity

        if should_recompute_core_attn:

            def core_attention_computation(q, k, v, mask, inv_norm_factor, dropout_fn):
                attn_weights_core = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * inv_norm_factor
                if mask is not None:
                    attn_weights_core = attn_weights_core + mask
                attn_weights_core = F.softmax(attn_weights_core, axis=-1).astype(v.dtype)
                attn_weights_core = dropout_fn(attn_weights_core)
                return paddle.matmul(attn_weights_core, v), attn_weights_core  # Return weights for output if needed

            attn_output, attn_weights_for_output = recompute(
                core_attention_computation,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.inv_norm_factor,
                self.attention_dropout,  # Pass the dropout layer itself
                use_reentrant=self.recompute_use_reentrant,
            )
        else:
            # Compute attention scores
            attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.inv_norm_factor
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            # Convert scores to probabilities
            attn_weights = F.softmax(attn_weights, axis=-1).astype(value_states.dtype)
            attn_weights_for_output = attn_weights  # Save for potential output
            attn_weights = self.attention_dropout(attn_weights)
            # Compute attention output
            attn_output = paddle.matmul(attn_weights, value_states)

        # Transpose and reshape output
        attn_output = attn_output.transpose([0, 2, 1, 3])  # [bsz, q_len, num_attention_heads_local, head_dim]
        # Reshape to [bsz, q_len, num_attention_heads_local * head_dim]
        # This is the input to o_proj, which is hidden_size / TP_degree if TP > 1
        attn_output = attn_output.reshape([bsz, q_len, self.num_attention_heads * self.head_dim])
        attn_output = self.o_proj(attn_output)  # o_proj is RowParallelLinear, gathers if TP/SP

        outputs = (attn_output,)
        if use_cache:
            outputs += (past_key_value,)
        if output_attentions:
            outputs += (attn_weights_for_output,)  # Use the saved attn_weights before dropout

        return outputs


class Phi3MLP(nn.Layer):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        self.tensor_parallel_degree = getattr(config, "tensor_parallel_degree", 1)

        # intermediate_size in config is global. For TP, it needs to be divided.
        intermediate_size = config.intermediate_size
        if self.tensor_parallel_degree > 1 and not self.sequence_parallel:  # TP only
            intermediate_size = intermediate_size // self.tensor_parallel_degree
        # For SP, ColumnSequenceParallelLinear handles splitting internally if tensor_parallel_degree > 1.
        # So, pass the global intermediate_size if SP is on and TP > 1, or just intermediate_size if TP=1.
        # If SP and TP > 1, the nn.Linear inside ColumnSequenceParallelLinear will use intermediate_size / TP.
        # If only SP (TP=1), it uses intermediate_size.

        # Determine correct Linear layer type
        if self.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear
        elif self.tensor_parallel_degree > 1:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear
        else:
            ColumnParallelLinear = nn.Linear
            RowParallelLinear = nn.Linear

        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias_attr=True)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias_attr=True)
        # For RowParallelLinear, input_dim is the TP-split intermediate_size
        # For RowSequenceParallelLinear, input_dim is also TP-split if TP > 1
        # So, if TP > 1, down_proj input is intermediate_size / TP degree.
        # Otherwise, it's the full intermediate_size.
        down_proj_input_dim = (
            config.intermediate_size // self.tensor_parallel_degree
            if self.tensor_parallel_degree > 1
            else config.intermediate_size
        )
        self.down_proj = RowParallelLinear(
            down_proj_input_dim,
            hidden_size,
            bias_attr=True,
            input_is_parallel=(self.tensor_parallel_degree > 1 or self.sequence_parallel),
        )
        self.act_fn = F.gelu

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # x is [bs, seq_len, hidden_size] or [bs*seq_len/TP, hidden_size] if SP
        gate_out = self.gate_proj(x)  # Output is [..., intermediate_size/TP] if TP
        up_out = self.up_proj(x)  # Output is [..., intermediate_size/TP] if TP

        activated_gate = self.act_fn(gate_out)
        fused_out = activated_gate * up_out  # Element-wise on TP-split dimension

        # down_proj takes [..., intermediate_size/TP] and outputs [..., hidden_size] (gathered)
        return self.down_proj(fused_out)


class Phi3DecoderLayer(nn.Layer):
    def __init__(self, config: Phi3Config, layerwise_recompute: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layerwise_recompute = layerwise_recompute
        # Default recompute_granularity to "full" if not in config, layer might override with "full_attn"
        self.recompute_granularity = getattr(config, "recompute_granularity", "full")
        self.enable_recompute = False  # Controlled by Phi3Model, which is controlled by Trainer
        # Default recompute_use_reentrant to True if not in config
        self.recompute_use_reentrant = getattr(config, "recompute_use_reentrant", True)

        self.self_attn = Phi3Attention(config, layerwise_recompute=self.layerwise_recompute)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        if getattr(config, "sequence_parallel", False):  # Check sequence_parallel from config
            mark_as_sequence_parallel_parameter(self.input_layernorm.weight)
            mark_as_sequence_parallel_parameter(self.post_attention_layernorm.weight)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        # batch_size is not explicitly taken by Phi3's original DecoderLayer/Attention/MLP forward
        # If SP is used, hidden_states might be [token_num_this_rank, dim]
        # The attention_mask and position_ids must align with this.
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]], Optional[paddle.Tensor]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        has_gradient = not hidden_states.stop_gradient

        # Set recompute flag for self_attn module if it has one
        if hasattr(self.self_attn, "enable_recompute"):
            self.self_attn.enable_recompute = self.enable_recompute
            if hasattr(self.self_attn, "recompute_use_reentrant"):
                self.self_attn.recompute_use_reentrant = self.recompute_use_reentrant

        # Recompute for 'full_attn' granularity, if enabled at this layer
        # The 'full' layer recompute is handled by Phi3Model
        should_recompute_attn = (
            self.enable_recompute
            and self.layerwise_recompute
            and has_gradient
            and self.recompute_granularity == "full_attn"
        )
        if should_recompute_attn:
            self_attn_outputs = recompute(
                self.self_attn,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                use_reentrant=self.recompute_use_reentrant,
            )
        else:
            self_attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        attn_output = self_attn_outputs[0]
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Pass recompute flags to MLP if it supports internal recompute
        if hasattr(self.mlp, "enable_recompute"):
            self.mlp.enable_recompute = self.enable_recompute
            if hasattr(self.mlp, "recompute_use_reentrant"):
                self.mlp.recompute_use_reentrant = self.recompute_use_reentrant

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        # Correctly append past_key_value and attentions based on what self_attn_outputs contains
        next_output_idx = 1
        if use_cache:
            if len(self_attn_outputs) > next_output_idx:
                outputs += (self_attn_outputs[next_output_idx],)
                next_output_idx += 1
            # else: outputs += (None,) # Or handle error: cache expected but not received

        if output_attentions:
            if len(self_attn_outputs) > next_output_idx:
                outputs += (self_attn_outputs[next_output_idx],)
            # else: outputs += (None,) # Or handle error: attention expected but not received

        return outputs


class Phi3PreTrainedModel(PretrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models."""

    config_class = Phi3Config
    base_model_prefix = "phi3"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"h\.[\d+]\.self_attention.rotary_emb.inv_freq"]

    def _init_weights(self, layer):
        """Initialize the weights."""
        # Consider tensor parallel degree from config, defaulting to 1 if not present
        tensor_parallel_degree = getattr(self.config, "tensor_parallel_degree", 1)
        if tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state

        # Check for specific linear layer types once they are integrated
        # For now, using nn.Linear and nn.Embedding as placeholders
        if isinstance(
            layer,
            (
                nn.Linear,
                nn.Embedding,
                linear_utils.ColumnParallelLinear,
                linear_utils.RowParallelLinear,
                linear_utils.ColumnSequenceParallelLinear,
                linear_utils.RowSequenceParallelLinear,
            ),
        ):
            if isinstance(layer.weight, paddle.Tensor):
                is_distributed = getattr(layer.weight, "is_distributed", False)
                if is_distributed and tensor_parallel_degree > 1:  # Check is_distributed only if TP > 1
                    with rng_tracker():
                        layer.weight.set_value(
                            paddle.tensor.normal(
                                mean=0.0,
                                std=self.config.initializer_range,
                                shape=layer.weight.shape,
                            )
                        )
                else:  # Not distributed or TP degree is 1
                    layer.weight.set_value(
                        paddle.tensor.normal(
                            mean=0.0,
                            std=self.config.initializer_range,
                            shape=layer.weight.shape,
                        )
                    )
            if hasattr(layer, "bias") and isinstance(layer.bias, paddle.Tensor) and layer.bias is not None:
                # Check if bias is distributed (e.g. ColumnParallelLinear bias)
                is_bias_distributed = getattr(layer.bias, "is_distributed", False)
                if is_bias_distributed and tensor_parallel_degree > 1:
                    with rng_tracker():  # Should biases be initialized differently for TP? Usually zeros.
                        layer.bias.set_value(paddle.zeros_like(layer.bias))
                else:
                    layer.bias.set_value(paddle.zeros_like(layer.bias))

        # TODO: Add scaling for RowParallelLinear weights like in Qwen2 if applicable,
        # after sequence parallel linear layers are integrated.
        # Example:
        # with paddle.no_grad():
        #     if isinstance(layer, Phi3MLP): # or specific linear layer type
        #         factor = 1 / math.sqrt(2 * self.config.num_hidden_layers) # Example factor
        #         if hasattr(layer, "down_proj"): # Or specific weight
        #             layer.down_proj.weight.scale_(factor)
        #     if isinstance(layer, Phi3Attention):
        #         factor = 1 / math.sqrt(2 * self.config.num_hidden_layers) # Example factor
        #         if hasattr(layer, "o_proj"):
        #             layer.o_proj.weight.scale_(factor)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Phi3Model):  # Or relevant submodules like Phi3DecoderLayer
            module.gradient_checkpointing = value  # Keep for backward compatibility
            module.enable_recompute = value


class Phi3Model(Phi3PreTrainedModel):
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        # Default recompute_granularity to "full" if not specified
        self.recompute_granularity = getattr(config, "recompute_granularity", "full")
        self.no_recompute_layers = getattr(config, "no_recompute_layers", [])
        # Ensure recompute_use_reentrant defaults to True if not in config
        self.recompute_use_reentrant = getattr(config, "recompute_use_reentrant", True)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.LayerList(
            [
                Phi3DecoderLayer(config, layerwise_recompute=(idx not in self.no_recompute_layers))
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        if self.sequence_parallel:
            mark_as_sequence_parallel_parameter(self.norm.weight)

        self.enable_recompute = False  # To be controlled by Trainer
        self.gradient_checkpointing = False  # Kept for backward compatibility, prefer enable_recompute

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[Tuple[paddle.Tensor, paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[paddle.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
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
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = paddle.arange(past_key_values_length, seq_length + past_key_values_length)
            position_ids = position_ids.expand([batch_size, seq_length])

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                expanded_mask = _expand_mask(attention_mask, seq_length)
                expanded_mask = expanded_mask.astype(paddle.get_default_dtype())
            else:
                expanded_mask = attention_mask

        else:
            if past_key_values_length > 0:
                expanded_mask = _make_causal_mask(
                    [batch_size, seq_length],
                    past_key_values_length,
                )
            else:
                expanded_mask = None

        # Embed positions
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Scatter inputs_embeds if sequence_parallel is enabled
        if self.sequence_parallel:
            # inputs_embeds: [bs, seq_len, dim] -> [bs * seq_len, dim]
            # Need to store original batch_size and seq_length for GatherOp later
            # This was done near the start of the original forward method
            # batch_size, seq_length are already defined if input_ids is not None
            # If only inputs_embeds is provided, batch_size, seq_length are derived from its shape
            # This logic should be fine.
            current_batch_size, current_seq_length, _ = inputs_embeds.shape
            inputs_embeds_reshaped = paddle.reshape_(
                inputs_embeds, [current_batch_size * current_seq_length, inputs_embeds.shape[-1]]
            )
            inputs_embeds = ScatterOp.apply(inputs_embeds_reshaped)
        else:
            # Store batch_size and seq_length if not SP for potential GatherOp if only norm is SP (not typical)
            current_batch_size, current_seq_length = batch_size, seq_length

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                # If SP, hidden_states is [bs*seq/TP, dim]. Need to gather before appending if want full hs.
                # Qwen2 appends the scattered hidden_states. Let's follow that for now.
                all_hidden_states += (hidden_states,)

            if hasattr(decoder_layer, "enable_recompute"):  # Set recompute flag for the layer
                decoder_layer.enable_recompute = self.enable_recompute
                # Also pass recompute_use_reentrant if the layer uses it
                if hasattr(decoder_layer, "recompute_use_reentrant"):
                    decoder_layer.recompute_use_reentrant = self.recompute_use_reentrant

            has_gradient = not hidden_states.stop_gradient

            # Check for recompute conditions
            should_recompute = (
                self.enable_recompute
                and (idx not in self.no_recompute_layers)
                and has_gradient
                and self.recompute_granularity == "full"
            )

            if should_recompute:
                layer_outputs = recompute(
                    decoder_layer,  # The function to recompute
                    hidden_states,  # Input hidden_states
                    expanded_mask,  # attention_mask
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    # Phi3DecoderLayer.forward doesn't take batch_size, so not passing it here
                    use_reentrant=self.recompute_use_reentrant,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=expanded_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                attn_output_idx = 2 if use_cache else 1
                if len(layer_outputs) > attn_output_idx:
                    all_self_attns += (layer_outputs[attn_output_idx],)

        hidden_states = self.norm(hidden_states)  # norm is applied on [bs*seq_len/TP, dim] if SP

        if self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
            # Reshape back to [bs, seq_len, dim] using stored original batch_size and seq_length
            # batch_size, seq_length were defined at the start of the forward method
            hidden_states = paddle.reshape_(hidden_states, [batch_size, seq_length, self.hidden_size])

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi3ForCausalLM(Phi3PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.phi3 = Phi3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.phi3.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.phi3.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: paddle.Tensor,
        past_key_values: Optional[List[Tuple[paddle.Tensor, paddle.Tensor]]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

            if position_ids is not None:
                position_ids = position_ids[:, remove_prefix_length:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[Tuple[paddle.Tensor, paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[paddle.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            past_key_values (`tuple(tuple(paddle.Tensor))`, *optional*, returned when `use_cache=True` is passed):
                Contains pre-computed key and value hidden states of the attention blocks.
            inputs_embeds (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`CausalLMOutputWithCrossAttentions`] instead of a tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.phi3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.reshape([-1, self.config.vocab_size]), shift_labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
