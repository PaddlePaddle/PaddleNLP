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
from functools import partial
from typing import List

import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.log import logger

from ...utils.converter import StateDictNameMapping, init_name_mappings
from ..model_outputs import ModelOutput
from .configuration import QWenConfig

MAX_NTK_SEQ_LENGTH = 32768


def parallel_matmul(x: Tensor, y: Tensor, tensor_parallel_output=True):
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
        logits = paddle.matmul(input_parallel, y, transpose_y=False)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)

    else:
        logits = paddle.matmul(x, y, transpose_y=False)
        return logits


class QWenAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()

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

        if config.tensor_parallel_degree > 1:
            if config.num_attention_heads % config.tensor_parallel_degree != 0:
                raise ValueError("num_attention_heads has to be divisible by tensor_parallel_degree")
            self.num_heads = config.num_attention_heads // config.tensor_parallel_degree
            self.c_attn = mpu.ColumnParallelLinear(
                config.hidden_size,
                3 * self.projection_size,
                has_bias=True,
                gather_output=False,
            )
            self.c_proj = mpu.RowParallelLinear(
                config.hidden_size,
                self.projection_size,
                has_bias=not config.no_bias,
                input_is_parallel=True,
            )
        else:
            self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size, bias_attr=True)
            self.c_proj = nn.Linear(config.hidden_size, self.projection_size, bias_attr=not config.no_bias)

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(self.hidden_size_per_attention_head * config.rotary_pct)
        dim = self.rotary_ndims if self.rotary_ndims is not None else self.hidden_size_per_attention_head
        self.rotary_emb = RotaryEmbedding(dim, base=config.rotary_emb_base)

        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn

        logn_list = [math.log(i, self.seq_length) if i > self.seq_length else 1 for i in range(1, MAX_NTK_SEQ_LENGTH)]
        self.logn_tensor = paddle.to_tensor(logn_list)[None, :, None, None]
        self._ntk_cached = 1.0

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2]))

        if self.scale_attn_weights:
            attn_weights = attn_weights * self.inv_norm_factor

        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, axis=-1)

        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

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
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        use_cache=False,
    ):
        # [bz, sql, hid] ==> [bz, sql, 3*hid]
        mixed_x_layer = self.c_attn(hidden_states)
        # [bz, sql, 3*hid] ==> [bz, sql, hid]
        query, key, value = paddle.split(mixed_x_layer, num_or_sections=3, axis=-1)

        # [bz, sql, hid] ==> [bz, sql, nh, hdim]
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
            query = query * logn_tensor.expand(query.shape)

        query = query.transpose([0, 2, 1, 3])
        key = key.transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])
        attn_output, attn_weight = self._attn(query, key, value, attention_mask)
        context_layer = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.c_proj(context_layer)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weight,)

        return outputs


class QWenMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        ff_dim_in = config.intermediate_size // 2
        if config.tensor_parallel_degree > 1:
            self.w1 = mpu.ColumnParallelLinear(
                config.hidden_size,
                ff_dim_in,
                gather_output=False,
                has_bias=False,
            )
            self.w2 = mpu.ColumnParallelLinear(
                config.hidden_size,
                ff_dim_in,
                gather_output=False,
                has_bias=False,
            )
            self.c_proj = mpu.RowParallelLinear(
                ff_dim_in,
                config.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.w1 = nn.Linear(config.hidden_size, ff_dim_in, bias_attr=not config.no_bias)
            self.w2 = nn.Linear(config.hidden_size, ff_dim_in, bias_attr=not config.no_bias)
            self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias_attr=not config.no_bias)

    def forward(self, hidden_states):
        # up
        a1 = self.w1(hidden_states)
        # gate
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        # down
        output = self.c_proj(intermediate_parallel)
        return output


class QWenBlock(nn.Layer):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

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
    base_model_prefix = "qwen"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_hidden_layers):
            final_actions = {}
            base_actions = {
                # Column Linear
                "lm_head.weight": partial(fn, is_column=True),
                "qwen.h.0.mlp.w2.weight": partial(fn, is_column=True),
                "qwen.h.0.mlp.w1.weight": partial(fn, is_column=True),
                "qwen.h.0.attn.c_attn.weight": partial(fn, is_column=True, is_naive_3fuse=True),
                "qwen.h.0.attn.c_attn.bias": partial(fn, is_column=True, is_naive_3fuse=True),
                # Row Linear
                "qwen.wte.weight": partial(fn, is_column=False),
                "qwen.h.0.mlp.c_proj.weight": partial(fn, is_column=False),
                "qwen.h.0.attn.c_proj.weight": partial(fn, is_column=False),
            }
            for key, action in base_actions.items():
                if "h.0." in key:
                    for i in range(num_hidden_layers):
                        final_actions[key.replace("h.0.", f"h.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

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
                mapping[1] = "qwen." + mapping[1]

        if config.architectures is not None:
            if "QWenForCausalLM" in config.architectures:
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
        if isinstance(
            module,
            (
                nn.Linear,
                nn.Embedding,
                mpu.ColumnParallelLinear,
                mpu.RowParallelLinear,
                mpu.VocabParallelEmbedding,
                QWenLMHead,
            ),
        ):
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


class QWenModel(QWenPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.enable_recompute = False

        if config.tensor_parallel_degree > 1:
            self.wte = mpu.VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
            )
        else:
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

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        block,
        hidden_states,
        layer_past,
        attention_mask,
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
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            use_reentrant=False,
        )
        return hidden_states

    def get_masks(self, batch_size, seq_length, past_length, padding_mask=None):
        # casual mask
        casual_mask = paddle.tril(paddle.ones([batch_size, 1, seq_length, seq_length], dtype="bool"))
        if past_length > 0:
            casual_mask = paddle.concat(
                [paddle.ones([batch_size, 1, seq_length, past_length], dtype="bool"), casual_mask], axis=-1
            )

        # seq_mask
        if padding_mask is None:
            padding_mask = paddle.ones((batch_size, 1, seq_length, seq_length + past_length), dtype="bool")
        if len(padding_mask.shape) == 2:
            # from Tokenizer
            padding_mask = (
                padding_mask.unsqueeze(axis=[1, 2])
                .expand([batch_size, 1, seq_length, seq_length + past_length])
                .astype("bool")
            )
        elif len(padding_mask.shape) == 3:
            # [batch_size,tgt_length, src_length] -> [batch_size, 1, tgt_length, src_length]
            padding_mask = padding_mask.unsqueeze(1).astype("bool")
        elif len(padding_mask.shape) == 4:
            padding_mask = padding_mask.astype("bool")

        casual_mask = casual_mask & padding_mask

        return casual_mask

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
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
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].shape[1]

        encoder_attention_mask = None
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds

        # bool 4D mask
        attention_mask = self.get_masks(input_shape[0], input_shape[1], past_length, padding_mask=attention_mask)
        zero = paddle.zeros(attention_mask.shape, dtype=hidden_states.dtype)
        neg_inf = paddle.full_like(attention_mask, paddle.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        # dtype 4D mask
        attention_mask = paddle.where(attention_mask, zero, neg_inf)

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + [
            hidden_states.shape[-1],
        ]

        if self.enable_recompute and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with recompute")
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


class QWenLMHead(nn.Layer):
    def __init__(self, config: QWenConfig):
        super(QWenLMHead, self).__init__()
        self.config = config
        if config.tensor_parallel_degree > 1:
            vocab_size = config.vocab_size // config.tensor_parallel_degree
        else:
            vocab_size = config.vocab_size

        self.weight = self.create_parameter(
            shape=[config.hidden_size, vocab_size],
            dtype=paddle.get_default_dtype(),
        )
        # Must set distributed attr for Tensor Parallel !
        self.weight.is_distributed = True if (vocab_size != config.vocab_size) else False
        if self.weight.is_distributed:
            self.weight.split_axis = 1

    def forward(self, hidden_states, tensor_parallel_output=None):
        if tensor_parallel_output is None:
            tensor_parallel_output = self.config.tensor_parallel_output

        logits = parallel_matmul(hidden_states, self.weight, tensor_parallel_output=tensor_parallel_output)
        return logits


class QWenForCausalLM(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]

    def __init__(self, config):
        super().__init__(config)
        self.qwen = QWenModel(config)
        self.lm_head = QWenLMHead(config)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["cache"] = outputs[1]
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, ModelOutput) and "past_key_values" in outputs:
            model_kwargs["cache"] = outputs.past_key_values
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = None

        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
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
            attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)
        return attention_mask

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
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

        transformer_outputs = self.qwen(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits, labels)

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
