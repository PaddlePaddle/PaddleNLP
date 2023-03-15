# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle LLaMA model"""

import math
from typing import Optional, Tuple

import paddle
import paddle.nn.functional as F
from configuration import LLaMAConfig
from paddle import nn
from paddle.distributed import fleet
from paddle.nn import CrossEntropyLoss

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

__all__ = [
    "LLaMAModel",
    "LLaMAPretrainedModel",
    "LLaMAForCausalLM",
]


class RMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size], dtype="float32", default_initializer=nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        if self.config.use_pure_fp16:
            with paddle.amp.auto_cast(False):
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states.astype("float32") * paddle.rsqrt(variance + self.variance_epsilon)

                output = self.weight * hidden_states
                output = output.astype("float16")
        else:
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states.astype("float32") * paddle.rsqrt(variance + self.variance_epsilon)

            output = self.weight * hidden_states
            output = output.astype(paddle.get_default_dtype())
        return output


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2) / dim))
        # self.register_buffer("inv_freq", inv_freq)
        self.inv_freq = inv_freq

        self.max_seq_len_cached = max_position_embeddings
        t = paddle.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = paddle.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = paddle.concat([freqs, freqs], axis=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LLaMAMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if config.mp_degree > 1:
            self.gate_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.up_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)

        if config.mp_degree > 1:
            self.down_proj = fleet.meta_parallel.RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMAAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.num_heads})."
        #     )

        assert self.num_heads % config.mp_degree == 0
        self.num_heads = self.num_heads // config.mp_degree
        self.head_dim = self.hidden_size // self.num_heads

        if config.mp_degree > 1:
            self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim * config.mp_degree,
                has_bias=False,
                gather_output=False,
            )
            self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim * config.mp_degree,
                has_bias=False,
                gather_output=False,
            )
            self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim * config.mp_degree,
                has_bias=False,
                gather_output=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias_attr=False,
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias_attr=False,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias_attr=False,
            )

        if config.mp_degree > 1:
            self.o_proj = fleet.meta_parallel.RowParallelLinear(
                self.num_heads * self.head_dim * config.mp_degree,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim * config.mp_degree,
                self.hidden_size,
                bias_attr=False,
            )
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.config = config

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.shape

        query_states = (
            self.q_proj(hidden_states)
            .reshape(shape=[bsz, q_len, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        key_states = (
            self.k_proj(hidden_states)
            .reshape(shape=[bsz, q_len, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )
        value_states = (
            self.v_proj(hidden_states)
            .reshape(shape=[bsz, q_len, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3])
        )

        kv_seq_len = key_states.shape[-2]
        offset = 0
        if past_key_value:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states)

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)

        if attn_weights.shape != [bsz, self.num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        attention_mask = attention_mask.expand([bsz, 1, q_len, kv_seq_len])

        if attention_mask is not None:
            if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
                raise ValueError(
                    f"Attention mask should be of shape {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        if self.config.use_pure_fp16:
            with paddle.amp.auto_cast(False):
                attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)
        else:
            attn_weights = F.softmax(attn_weights, axis=-1, dtype="float32").astype(query_states.dtype)

        attn_output = paddle.matmul(attn_weights, value_states)

        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of shape {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LLaMADecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLaMAAttention(config)
        self.mlp = LLaMAMLP(config)
        self.input_layernorm = RMSNorm(config)
        self.post_attention_layernorm = RMSNorm(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LLaMAPretrainedModel(PretrainedModel):
    config_class = LLaMAConfig
    base_model_prefix = "llama"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                # TODO(linjieccc): enable after normal support fp16
                if paddle.get_default_dtype() not in ["float16"]:
                    layer.weight.set_value(
                        paddle.tensor.normal(
                            mean=0.0,
                            std=self.config.initializer_range
                            if hasattr(self.config, "initializer_range")
                            else self.llama.config.initializer_range,
                            shape=layer.weight.shape,
                        )
                    )


@register_base_model
class LLaMAModel(LLaMAPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LLaMADecoderLayer`]
    Args:
        config: LLaMAConfig
    """

    def __init__(self, config: LLaMAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.bias = paddle.tril(
            paddle.ones([1, 1, config.max_position_embeddings, config.max_position_embeddings], dtype="int64")
        )

        if config.mp_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range)
                ),
            )
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)

        self.layers = nn.LayerList([LLaMADecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):

        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape(shape=(-1, input_shape[-1]))
        else:
            raise ValueError("You have to specify input_ids")

        # past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        inputs_embeds = self.embed_tokens(input_ids)

        length = input_shape[-1]
        causal_mask = self.bias[:, :, :length, :length]

        attention_mask = (1.0 - causal_mask) * -1e4
        # The tensor returned by triu not in static graph.
        attention_mask.stop_gradient = True

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
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
            cross_attentions=None,
        )


class LLaMAForCausalLM(LLaMAPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.llama = LLaMAModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.llama(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape([-1, self.config.vocab_size]),
                shift_labels.reshape(
                    [
                        -1,
                    ]
                ),
            )

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
