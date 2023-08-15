# Copyright (c) 2023 Alibaba Cloud and PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import List, Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.log import logger

from ...utils.converter import StateDictNameMapping, init_name_mappings
from .configuration import QWenConfig


class QWenAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            paddle.tril(paddle.ones((max_positions, max_positions), dtype=paddle.bool)).reshape(
                [1, 1, max_positions, max_positions]
            ),
            persistable=False,
        )
        self.register_buffer("masked_bias", paddle.to_tensor(-1e4), persistable=False)
        self.seq_length = config.seq_length

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self.scale_attn_weights = True

        self.projection_size = config.kv_channels * config.num_attention_heads

        assert self.projection_size % config.num_attention_heads == 0
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads

        self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)

        self.c_proj = nn.Linear(config.hidden_size, self.projection_size, bias_attr=not config.no_bias)

        self.is_fp32 = not (config.bf16 or config.fp16)

        self.bf16 = config.bf16

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(self.hidden_size_per_attention_head * config.rotary_pct)
        dim = self.rotary_ndims if self.rotary_ndims is not None else self.hidden_size_per_attention_head
        self.rotary_emb = RotaryEmbedding(dim, base=config.rotary_emb_base)

        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn

        logn_list = [math.log(i, self.seq_length) if i > self.seq_length else 1 for i in range(1, 32768)]
        self.logn_tensor = paddle.to_tensor(logn_list)[None, :, None, None]
        self._ntk_cached = 1.0

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2]))

        if self.scale_attn_weights:
            attn_weights = attn_weights * self.inv_norm_factor

        query_length, key_length = query.shape[-2], key.shape[-2]
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = paddle.finfo(attn_weights.dtype).min
        mask_value = paddle.full_like(attn_weights, mask_value, dtype=attn_weights.dtype)
        attn_weights = paddle.where(causal_mask, attn_weights, mask_value)

        attn_weights = nn.functional.softmax(attn_weights, axis=-1)

        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = paddle.matmul(attn_weights, value)
        attn_output = attn_output.transpose([0, 2, 1, 3])

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.shape[:-2] + [
            num_heads * attn_head_size,
        ]
        return tensor.reshape(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        use_cache=False,
    ):

        mixed_x_layer = self.c_attn(hidden_states)
        query, key, value = paddle.split(mixed_x_layer, num_or_sections=3, axis=-1)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        kv_seq_len = hidden_states.shape[1]
        if layer_past:
            # layer past[0] shape: bs * seq_len * head_num * dim
            kv_seq_len += layer_past[0].shape[1]
        if self.use_dynamic_ntk and kv_seq_len == hidden_states.shape[1] and not self.training:
            context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
            ntk_alpha = 2 ** math.ceil(context_value) - 1
            ntk_alpha = max(ntk_alpha, 1)
            self._ntk_cached = ntk_alpha
        else:
            ntk_alpha = self._ntk_cached
        rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # Slice the pos emb for current inference
            cur_len = query.shape[1]
            q_pos_emb = q_pos_emb[:, -cur_len:, :, :]
            k_pos_emb = k_pos_emb[:, -cur_len:, :, :]
            query = apply_rotary_pos_emb(query, q_pos_emb)
            key = apply_rotary_pos_emb(key, k_pos_emb)

        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            key = paddle.concat([past_key, key], axis=1)
            value = paddle.concat([past_value, value], axis=1)

        if use_cache:
            present = (key, value)
        else:
            present = None

        if self.use_logn_attn and not self.training:
            if self.logn_tensor.dtype != query.dtype:
                self.logn_tensor = self.logn_tensor.astype(query.dtype)
            seq_start = key.shape[1] - query.shape[1]
            seq_end = key.shape[1]
            logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
            query = query * logn_tensor.expand_as(query)

        query = query.transpose([0, 2, 1, 3])
        key = key.transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])
        attn_output, attn_weight = self._attn(query, key, value, attention_mask, head_mask)
        context_layer = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(context_layer)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight,)

        return outputs


class QWenMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias_attr=not config.no_bias)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias_attr=not config.no_bias)
        ff_dim_in = config.intermediate_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias_attr=not config.no_bias)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output


class QWenBlock(nn.Layer):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.bf16 = config.bf16

        self.ln_1 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attn = QWenAttention(config)
        self.ln_2 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.mlp = QWenMLP(config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            layernorm_output,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        residual = hidden_states
        layernorm_input = attn_output + residual

        layernorm_output = self.ln_2(layernorm_input)

        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class QWenPreTrainedModel(PretrainedModel):
    config_class = QWenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["QWenBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @classmethod
    def _get_name_mappings(cls, config: QWenConfig) -> List[StateDictNameMapping]:
        mappings = [
            "wte.weight",
            "ln_f.weight",
        ]

        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"h.{layer_index}.ln_1.weight",
                    f"h.{layer_index}.ln_1.weight",
                ],
                [
                    f"h.{layer_index}.attn.c_attn.weight",
                    f"h.{layer_index}.attn.c_attn.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.attn.c_attn.bias",
                    f"h.{layer_index}.attn.c_attn.bias",
                ],
                [
                    f"h.{layer_index}.attn.c_proj.weight",
                    f"h.{layer_index}.attn.c_proj.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.ln_2.weight",
                    f"h.{layer_index}.ln_2.weight",
                ],
                [
                    f"h.{layer_index}.mlp.w1.weight",
                    f"h.{layer_index}.mlp.w1.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.mlp.w2.weight",
                    f"h.{layer_index}.mlp.w2.weight",
                    "transpose",
                ],
                [
                    f"h.{layer_index}.mlp.c_proj.weight",
                    f"h.{layer_index}.mlp.c_proj.weight",
                    "transpose",
                ],
            ]
            mappings.extend(layer_mappings)

        init_name_mappings(mappings)
        for mapping in mappings:
            mapping[0] = "transformer." + mapping[0]
            if len(mapping) > 1 and mapping[1] is not None:
                mapping[1] = "transformer." + mapping[1]

        if config.architectures is not None:
            if "QWenLMHeadModel" in config.architectures:
                mappings.extend(
                    [
                        [
                            "lm_head.weight",
                            "lm_head.weight",
                            "transpose",
                        ]
                    ]
                )

        init_name_mappings(mappings)
        return [StateDictNameMapping(*mapping) for mapping in mappings]

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.set_value(
                paddle.tensor.normal(mean=0.0, std=self.config.initializer_range, shape=module.weight.shape)
            )
            if getattr(module, "bias", None) is not None:
                module.weight.set_value(paddle.zeros(shape=module.weight.shape, dtype=paddle.get_default_dtype()))

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers),
                        shape=p.shape,
                    )
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, QWenModel):
            module.enable_recompute = value


class QWenModel(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size

        self.enable_recompute = False

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

        self.drop = nn.Dropout(config.emb_dropout_prob)
        self.h = nn.LayerList(
            [
                QWenBlock(
                    config,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(
            self.embed_dim,
            eps=config.layer_norm_epsilon,
        )

        # self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            axis = paddle.to_tensor([0, 1, 3, 4])
            head_mask = paddle.unsqueeze(head_mask, axis=axis)
            head_mask = head_mask.expand(shape=(num_hidden_layers, -1, -1, -1, -1))
        elif head_mask.dim() == 2:
            axis = paddle.to_tensor([1, 3, 4])
            head_mask = paddle.unsqueeze(head_mask, axis=axis)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"

        head_mask = paddle.cast(head_mask, dtype=self.config.dtype)
        return head_mask

    def get_head_mask(self, head_mask: Optional[Tensor], num_hidden_layers: int) -> Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`paddle.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
        Returns:
            `paddle.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        block,
        hidden_states,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
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
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            use_reentrant=False,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.reshape([-1, input_shape[-1]])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1, input_shape[-1]])
        if position_ids is not None:
            position_ids = position_ids.reshape([-1, input_shape[-1]])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = paddle.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=paddle.int64,
            )
            position_ids = position_ids.unsqueeze(0).reshape([-1, input_shape[-1]])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.reshape([batch_size, -1])
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.astype(paddle.bfloat16)
            attention_mask = (1.0 - attention_mask) * paddle.finfo(paddle.bfloat16).min

        encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + [
            hidden_states.shape[-1],
        ]

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.enable_recompute and self.training:
                outputs = self.recompute_training(
                    block,
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.reshape(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QWenLMHeadModel(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = QWenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            paddle.where(attention_mask == 0, paddle.ones_like(position_ids, dtype=position_ids.dtype), position_ids)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype(paddle.int64)
        else:
            import pdb

            pdb.set_trace()
            attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)
        return attention_mask

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # TODO shift_label if
            # labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape([-1, shift_logits.shape[-1]]), shift_labels.reshape([-1]))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))
        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (paddle.arange(0, self.dim, 2, dtype=paddle.float32) / self.dim))
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = paddle.arange(self._seq_len_cached)
            freqs = paddle.outer(seq.astype(self.inv_freq.dtype), self.inv_freq)
            emb = paddle.concat([freqs, freqs], axis=-1)

            self._rotary_pos_emb_cache = emb.unsqueeze([0, 2])

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[:, offset : offset + max_seq_len]


def _rotate_half(x):

    x = x.reshape(x.shape[:-1] + [2, -1])
    x1, x2 = x.unbind(axis=-2)
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(t, freqs):
    rot_dim = freqs.shape[-1]
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    t_ = t_.astype(paddle.float32)
    t_pass_ = t_pass_.astype(paddle.float32)
    t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
    return paddle.concat([t_, t_pass_], axis=-1).astype(t.dtype)


class RMSNorm(nn.Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.astype(paddle.float32)).astype(x.dtype)
        return output * self.weight
