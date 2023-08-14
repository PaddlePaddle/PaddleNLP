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
from __future__ import annotations

from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from custom_setup_ops import fused_get_rotary_embedding, get_padding_offset
from paddle import Tensor, nn
from paddle.distributed import fleet

from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)
from paddlenlp.transformers.fused_multi_transformer_dyquant_fine_grained import (
    FusedMultiTransformerDyquant,
)
from paddlenlp.transformers.llama.configuration import LlamaConfig
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

LLAMA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/tiny-random-llama",
    "facebook/llama-7b",
    "facebook/llama-13b",
    "facebook/llama-65b",
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
        logits = paddle.matmul(input_parallel, y, transpose_y=False)
        if parallel_output:
            return logits
        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(x, y, transpose_y=False)
        return logits


class FusedLlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        return paddle.incubate.nn.functional.rms_norm(
            hidden_states, self.weight, None, self.variance_epsilon, begin_norm_axis=1
        )


class LlamaPretrainedModel(PretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    config_class = LlamaConfig
    base_model_prefix = "llama"

    @classmethod
    def _get_name_mappings(cls, config: LlamaConfig) -> list[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [f"layers.{layer_index}.self_attn.q_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.k_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.v_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.o_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                [f"layers.{layer_index}.input_layernorm.weight"],
                [f"layers.{layer_index}.post_attention_layernorm.weight"],
            ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)
        # base-model prefix "LlamaModel"
        if "LlamaModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "llama." + mapping[1]
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

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
                "lm_head.weight": partial(fn, is_column=True),
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

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range
                        if hasattr(self.config, "initializer_range")
                        else self.model.config.initializer_range,
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
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.epsilon = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.quant_bits = config.quant_bits

        print("Quant bits is: ", self.quant_bits)

        if config.tensor_parallel_degree > 1:
            self.embed_tokens = fleet.meta_parallel.VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        # get ring_id
        ring_id = -1
        try:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
        except:
            pass

        ln_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ln_scale".format(i))
            for i in range(self.num_layers)
        ]
        qkv_weight_attrs = [
            paddle.ParamAttr(name="fusellama.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0))
            for i in range(self.num_layers)
        ]
        qkv_weight_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.qkv_weight_scale".format(i))
            for i in range(self.num_layers)
        ]
        out_proj_weight_attrs = [
            paddle.ParamAttr(name="fusellama.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0))
            for i in range(self.num_layers)
        ]
        out_proj_weight_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.out_proj_weight_scale".format(i))
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn_ln_scale".format(i))
            for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0))
            for i in range(self.num_layers)
        ]
        ffn1_weight_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn1_weight_scale".format(i))
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0))
            for i in range(self.num_layers)
        ]
        ffn2_weight_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn2_weight_scale".format(i))
            for i in range(self.num_layers)
        ]

        self.transformer_block = FusedMultiTransformerDyquant(
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size,
            quant_bits=self.quant_bits, 
            activation="swiglu",
            num_layers=config.num_hidden_layers,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_weight_scale_attrs=qkv_weight_scale_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_weight_scale_attrs=out_proj_weight_scale_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            epsilon=self.epsilon,
            norm_type="rmsnorm",
            use_neox_rotary_style=True,
        )
        self.norm = FusedLlamaRMSNorm(config)

        self.cache_kvs = None
        self.head_dim_shape_tensor = paddle.ones((self.hidden_size // self.num_attention_heads), dtype="int8")

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        past_key_values = kwargs.get("cache", None)
        is_decoder = past_key_values is not None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        cache_kvs = cache_kvs if cache_kvs is not None else self.cache_kvs

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ids_remove_padding)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder

        new_rope = fused_get_rotary_embedding(input_ids, position_ids, self.head_dim_shape_tensor, 0, True)

        hidden_states, _ = self.transformer_block(
            input_ids,
            hidden_states,
            cum_offsets=cum_offsets,
            padding_offset=padding_offset,
            attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
            caches=cache_kvs,
            seq_lens=seq_lens,
            rotary_embs=new_rope,
            rotary_emb_dims=1,
            time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None,
        )
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        quant_bits = self.quant_bits

        unfused_state_dict = {}
        head_size = self.hidden_size // self.num_attention_heads

        self.embed_tokens.weight.set_value(
            paddle.to_tensor(state_dict["embed_tokens.weight"])
        )
        self.norm.weight.set_value(paddle.to_tensor(state_dict["norm.weight"]))

        for idx in range(self.config.num_hidden_layers):
            unfused_state_dict = {}
            unfused_state_dict["self_attn.q_proj.weight"] = state_dict[
                "layers.{}.self_attn.q_proj.weight".format(idx)
            ]
            unfused_state_dict["self_attn.k_proj.weight"] = state_dict[
                "layers.{}.self_attn.k_proj.weight".format(idx)
            ]
            unfused_state_dict["self_attn.v_proj.weight"] = state_dict[
                "layers.{}.self_attn.v_proj.weight".format(idx)
            ]

            concated_qkv_weight = (
                np.concatenate(
                    [
                        unfused_state_dict["self_attn.q_proj.weight"],
                        unfused_state_dict["self_attn.k_proj.weight"],
                        unfused_state_dict["self_attn.v_proj.weight"],
                    ],
                    axis=-1,
                )
                .transpose(1, 0)
                .reshape(
                    3 * self.num_attention_heads // self.config.tensor_parallel_degree * head_size,
                    self.hidden_size,
                )
            )

            qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
            qkv_weight_tensor = paddle.transpose(qkv_weight_tensor, perm=[1, 0])
            qkv_quanted_weight_tensor, qkv_weight_scale_tensor = F.quant_for_compress(qkv_weight_tensor, bits=self.quant_bits)
            self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight_tensor)
            self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale_tensor)


            linear_weight_tensor = paddle.to_tensor(state_dict["layers.{}.self_attn.o_proj.weight".format(idx)])
            linear_quanted_weight_tensor, linear_weight_scale_tensor = F.quant_for_compress(linear_weight_tensor, bits=self.quant_bits)
            self.transformer_block.linear_weights[idx].set_value(
                linear_quanted_weight_tensor
            )
            self.transformer_block.linear_weights_scale[idx].set_value(
                linear_weight_scale_tensor
            )

            unfused_state_dict["mlp.gate_proj.weight"] = state_dict[
                "layers.{}.mlp.gate_proj.weight".format(idx)
            ]
            unfused_state_dict["mlp.up_proj.weight"] = state_dict[
                "layers.{}.mlp.up_proj.weight".format(idx)
            ]

            concated_ffn1_weight = np.concatenate(
                [
                    unfused_state_dict["mlp.gate_proj.weight"],
                    unfused_state_dict["mlp.up_proj.weight"],
                ],
                axis=-1,
            )

            ffn1_weight_tensor = paddle.to_tensor(concated_ffn1_weight)
            ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = F.quant_for_compress(ffn1_weight_tensor, bits=self.quant_bits)
            self.transformer_block.ffn1_weights[idx].set_value(
                ffn1_quanted_weight_tensor
            )
            self.transformer_block.ffn1_weights_scale[idx].set_value(
                ffn1_weight_scale_tensor
            )

            ffn2_weight_tensor = paddle.to_tensor(state_dict["layers.{}.mlp.down_proj.weight".format(idx)])
            ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = F.quant_for_compress(ffn2_weight_tensor, bits=self.quant_bits)
            self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
            self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)

            self.transformer_block.ln_scales[idx].set_value(
                paddle.to_tensor(
                    state_dict["layers.{}.input_layernorm.weight".format(idx)]
                )
            )

            self.transformer_block.ffn_ln_scales[idx].set_value(
                paddle.to_tensor(
                    state_dict[
                        "layers.{}.post_attention_layernorm.weight".format(idx)
                    ]
                )
            )


class LlamaLMHead(nn.Layer):
    def __init__(self, config: LlamaConfig):
        super(LlamaLMHead, self).__init__()
        shard_num = config.tensor_parallel_degree if config.tensor_parallel_degree > 1 else 1
        self.weight = self.create_parameter(
            shape=[config.hidden_size, config.vocab_size // shard_num],
            dtype=paddle.get_default_dtype(),
        )
        self.config = config

    def forward(self, hidden_states, parallel_output=False):
        logits = parallel_matmul(hidden_states, self.weight, parallel_output=parallel_output)
        return logits


class LlamaForCausalLMDyBatch(LlamaPretrainedModel):
    """
    Dynamic Batching for LLaMA Model with pretraining tasks on top.

    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = LlamaLMHead(config)

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        attention_mask = paddle.ones_like(input_ids, dtype="int64")
        attention_mask = (input_ids != pad_token_id).astype("int64")
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_kvs,
        seq_len_encoder,
        seq_len_decoder,
        tgt_ids,
        tgt_pos,
        tgt_generation_mask,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        cache = kwargs.get("cache", None)
        if cache is not None:
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (tgt_generation_mask - 1) * 1e4
        else:
            attention_mask = (attention_mask - 1) * 1e4
        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
        }
        return model_inputs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
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

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            cache_kvs=cache_kvs,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        parallel_output = False
        logits = self.lm_head(
            hidden_states,
            parallel_output=parallel_output,
        )

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

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.lm_head.weight.set_value(state_dict["lm_head.weight"])
        self.model.set_state_dict({k: state_dict[k] for k in state_dict.keys()})
