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
"""Paddle Llama model"""

import math
from functools import partial
from typing import Optional, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

from .configuration import LlamaConfig

LLAMA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebookresearch/tiny-random-llama",
]

__all__ = [
    "LlamaModel",
    "LlamaPretrainedModel",
    "LlamaForCausalLM",
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

    mask = paddle.full((target_length, target_length), float(finfo(paddle.get_default_dtype()).min))

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
        self.inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2) / dim))

        t = paddle.arange(max_position_embeddings, dtype=self.inv_freq.dtype)
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
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


class LlamaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if config.tensor_parallel_degree > 1:
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

        if config.tensor_parallel_degree > 1:
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


class LlamaAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.num_heads % config.tensor_parallel_degree == 0
        self.head_dim = self.hidden_size // self.num_heads
        self.num_heads = self.num_heads // config.tensor_parallel_degree

        if config.tensor_parallel_degree > 1:
            self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                gather_output=False,
            )
            self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                gather_output=False,
            )
            self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                gather_output=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=False,
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=False,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias_attr=False,
            )

        if config.tensor_parallel_degree > 1:
            self.o_proj = fleet.meta_parallel.RowParallelLinear(
                self.hidden_size,
                self.hidden_size,
                has_bias=False,
                input_is_parallel=True,
            )
        else:
            self.o_proj = nn.Linear(
                self.hidden_size,
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

        if past_key_value is not None:
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
        attn_output = attn_output.reshape([bsz, q_len, self.head_dim * self.num_heads])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config)
        self.post_attention_layernorm = RMSNorm(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
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
                If set to `True`, `cache` key value states are returned and can be used to speed up decoding
                (see `cache`).
            cache (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
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


class LlamaPretrainedModel(PretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    config_class = LlamaConfig
    base_model_prefix = "llama"

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
                "layers.0.self_attn.q_proj.weight": partial(fn, is_column=True),
                "layers.0.self_attn.k_proj.weight": partial(fn, is_column=True),
                "layers.0.self_attn.v_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.gate_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.up_proj.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

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
class LlamaModel(LlamaPretrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            # self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
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

    @paddle.jit.not_to_static
    def recompute_training(self, layer_module, hidden_states, attention_mask, output_attentions):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_attentions, None)

            return custom_forward

        hidden_states = recompute(create_custom_forward(layer_module), hidden_states, attention_mask)
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
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

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[2]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), cache_length)
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            has_gradient = not hidden_states.stop_gradient
            if self.config.use_recompute and has_gradient:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    past_key_value,
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


class LlamaPretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for Llama.
    It calculates the final loss.
    """

    def __init__(self, tensor_parallel_degree=1, tensor_parallel_output=False):
        super(LlamaPretrainingCriterion, self).__init__()
        if tensor_parallel_degree > 1 and tensor_parallel_output:
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_scores, masked_lm_labels):
        masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))
        with paddle.amp.auto_cast(False):
            masked_lm_loss = masked_lm_loss.astype("float32")
            masked_lm_loss = masked_lm_loss[masked_lm_labels != -100]
            loss = paddle.mean(masked_lm_loss)
        return loss


class LlamaForCausalLM(LlamaPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.llama = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.criterion = LlamaPretrainingCriterion(
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_output=config.tensor_parallel_output,
        )

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.llama.embed_tokens

    def set_input_embeddings(self, value):
        self.llama.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llama = decoder

    def get_decoder(self):
        return self.llama

    def prepare_inputs_for_generation(
        self, input_ids, use_cache=False, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
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
            past_key_values=past_key_values,
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
