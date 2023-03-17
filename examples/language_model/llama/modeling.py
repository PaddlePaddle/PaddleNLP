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

import numpy as np
import paddle
import paddle.nn.functional as F
from configuration import LLaMAConfig
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from processor import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model
from paddlenlp.utils.log import logger

__all__ = [
    "LLaMAModel",
    "LLaMAPretrainedModel",
    "LLaMAForCausalLM",
    "LLaMAForGeneration",
]


def finfo(dtype):
    if dtype == "float32":
        return np.finfo(np.float32)
    if dtype == "float16":
        return np.finfo(np.float16)
    if dtype == "float64":
        return np.finfo(np.float64)


def _make_causal_mask(input_ids_shape, past_key_values_length, dtype):
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape

    mask = paddle.full((target_length, target_length), paddle.to_tensor(finfo(paddle.get_default_dtype()).min))
    mask_cond = paddle.arange(mask.shape[-1])
    mask_cond = mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1])
    mask = paddle.where(mask_cond, paddle.full(mask_cond.shape, 0), mask)

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask.unsqueeze(0).expand([batch_size, target_length, target_length + past_key_values_length])
    return expanded_mask


def _expand_mask(mask, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(paddle.cast(mask[:, None, :], "bool"))
    expanded_mask = paddle.cast(expanded_mask, dtype=paddle.float32)

    return expanded_mask.expand([batch_size, tgt_length, src_length])


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
        self.register_buffer("inv_freq", inv_freq)
        # self.inv_freq = inv_freq

        # self.max_seq_len_cached = max_position_embeddings
        # t = paddle.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = paddle.arange(2048, dtype=self.inv_freq.dtype)
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        # if seq_len > self.max_seq_len_cached:
        #     self.max_seq_len_cached = seq_len
        #     t = paddle.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if seq_len > 2048:
            t = paddle.arange(seq_len, dtype=self.inv_freq.dtype)
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
        use_cache: bool = False,
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

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)

        if attn_weights.shape != [bsz, self.num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of shape {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        attention_mask = attention_mask.reshape([bsz, 1, q_len, kv_seq_len])

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
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
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
            use_cache=use_cache,
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

        if config.mp_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=config.initializer_range)
                ),
            )
        else:
            # self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.LayerList([LLaMADecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length, dtype=attention_mask.dtype
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, tgt_length=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        **kwargs,
    ):
        past_key_values = kwargs.get("cache", past_key_values)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = recompute(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    past_key_value,
                    output_attentions,
                    use_cache,
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


class LLaMAPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for LLaMA.
    It calculates the final loss.
    """

    def __init__(self, pad_token_id=None, mp_degree=1):
        super(LLaMAPretrainingCriterion, self).__init__()
        if mp_degree > 1:
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        self.pad_token_id = pad_token_id

    def forward(self, prediction_scores, masked_lm_labels, loss_mask=None):
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


class LLaMAForCausalLM(LLaMAPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.llama = LLaMAModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.criterion = LLaMAPretrainingCriterion(pad_token_id=config.pad_token_id, mp_degree=config.mp_degree)

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
            loss = self.criterion(shift_logits, shift_labels)

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

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(tuple(past_state.index_select(0, beam_idx) for past_state in layer_past) for layer_past in past)


def TopKProcess(probs, top_k, min_tokens_to_keep):
    top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
    # Remove all tokens with a probability less than the last token of the top-k
    topk_probs, _ = paddle.topk(probs, k=top_k)
    probs = paddle.where(probs >= topk_probs[:, -1:], probs, paddle.full_like(probs, 0.0))
    return probs


class LLaMAForGeneration(LLaMAPretrainedModel):
    """
    LLaMA Model with pretraining tasks on top.

    Args:
        llama (:class:`LLaMAModel`):
            An instance of :class:`LLaMAModel`.

    """

    def __init__(self, config):
        super(LLaMAForGeneration, self).__init__(config)
        self.llama = LLaMAModel(config)
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
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="int64")], axis=-1
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
            return self.llama(**model_inputs, **immutable)

        def _post_process_(outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs):

            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            with paddle.fluid.name_scope("skip_quant"):
                logits = paddle.matmul(logits, self.llama.embed_tokens.weight, transpose_y=True)

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
