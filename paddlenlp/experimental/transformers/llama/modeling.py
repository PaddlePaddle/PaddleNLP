# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
from functools import partial

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.nn.quant import weight_quantize

from paddlenlp.experimental.model_utils import (
    ActScalesLoader,
    CacheScaleLoader,
    PerTensorWeightScalesLoader,
    WeightScalesLoader,
)
from paddlenlp.experimental.transformers.fused_transformer_layers import (
    AvxConfig,
    FusedBlockMultiTransformer,
    FusedBlockMultiTransformerA8W8,
    FusedBlockMultiTransformerFP8,
    FusedBlockMultiTransformerWeightOnly,
    FusedMultiTransformerA8W8,
    FusedMultiTransformerAvx,
    FusedMultiTransformerBase,
    FusedMultiTransformerConfig,
    FusedMultiTransformerWeightOnly,
    SpeculateConfig,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationAvxInferenceModel,
    GenerationBlockInferenceModel,
    GenerationInferenceModel,
)
from paddlenlp.experimental.transformers.utils import (
    EmptyActScale,
    EmptyCacheScale,
    EmptyWeightScale,
    infererence_model_from_pretrained,
)
from paddlenlp.transformers import LlamaConfig, LlamaPretrainedModel
from paddlenlp.transformers.conversion_utils import split_param_func
from paddlenlp.transformers.llama.modeling import LlamaLMHead
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)
from paddlenlp.utils.download import resolve_file_path
from paddlenlp.utils.log import logger

__all__ = [
    "LlamaInferenceModel",
    "LlamaForCausalLMInferenceModel",
    "LlamaForCausalLMAvxInferenceModel",
    "LlamaForCausalLMBlockInferenceModel",
    "LlamaForMiniGPT4InferenceModel",
]


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
        return paddle.incubate.nn.functional.fused_rms_norm(
            hidden_states, self.weight, None, self.variance_epsilon, begin_norm_axis=1
        )[0]


class LLamaAvxLMHead(nn.Layer):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.avx_type = config.avx_type
        self.alog = "fp16"
        if self.avx_type == "fp16":
            self.alog = "fp16"
        elif self.avx_type == "fp16_int8" or self.avx_type == "bf16_int8":
            self.alog = "int8"
        self.weight = self.create_parameter(
            shape=[self.config.hidden_size, self.config.vocab_size],
            attr=None,
            dtype=paddle.get_default_dtype(),
            is_bias=False,
        )

    def forward(self, input):
        from paddlenlp_ops import avx_weight_only

        return avx_weight_only(input, self.weight, self.alog, trans=False)


@register_base_model
class LlamaAvxInferenceModel(LlamaPretrainedModel):
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
        self.quant_type = config.quant_type
        self.dtype = config.dtype
        self.rope_theta = config.rope_theta
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )
        self.compute_type = config.avx_type
        self.cachekv_type = config.avx_cachekv_type
        ln_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.ln_scale".format(i)) for i in range(self.num_layers)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        gate_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.gate_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        up_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.up_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        avx_config = AvxConfig(
            max_position_embeddings=self.max_position_embeddings,
            compute_type=self.compute_type,
            cache_dtype=self.cachekv_type,
        )

        transformer_config = FusedMultiTransformerConfig(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            kv_num_heads=self.num_layers,
            dim_feedforward=self.intermediate_size,
            activation="silu",
            num_layers=self.num_layers,
            ln_scale_attrs=ln_scale_attrs,
            ln_bias_attrs=None,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_bias_attrs=None,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_bias_attrs=None,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn_ln_bias_attrs=None,
            gate_weight_attrs=gate_weight_attrs,
            gate_bias_attrs=None,
            up_weight_attrs=up_weight_attrs,
            up_bias_attrs=None,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_bias_attrs=None,
            norm_type="rmsnorm",
            epsilon=self.epsilon,
            rope_theta=self.rope_theta,
            nranks=config.tensor_parallel_degree,
            avx_config=avx_config,
        )

        self.set_transformer_block(transformer_config)
        self.norm = FusedLlamaRMSNorm(config)

    def set_transformer_block(self, transformer_config):
        self.transformer_block = FusedMultiTransformerAvx(transformer_config)

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_seq_len=None,
        cur_seq_len=None,
        step_idx=None,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # genereate a fake input_ids according to inputs_embeds
        if input_ids is None and inputs_embeds is not None:
            input_ids = self.prepare_input_ids_for_generation(self.config.bos_token_id, inputs_embeds)
        if inputs_embeds is not None:
            batch, seq_len, hidden_dim = inputs_embeds.shape
            # merge batch and seq_len dimension.
            inputs_embeds = inputs_embeds.reshape([batch * seq_len, hidden_dim])

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        with dy2st_nocheck_guard_context():
            hidden_states = self.transformer_block(
                input_ids=input_ids,
                src=hidden_states,
                past_seq_len=past_seq_len,
                cur_seq_len=cur_seq_len,
                step_idx=step_idx,
            )
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)

    @paddle.no_grad()
    # avx
    def set_state_dict(self, state_dict):
        unfused_state_dict = {}
        head_size = self.hidden_size // self.num_attention_heads
        split_fn = split_param_func()

        self.embed_tokens.weight.set_value(
            paddle.to_tensor(state_dict["llama.embed_tokens.weight"]).cast(self.embed_tokens.weight.dtype)
        )
        self.norm.weight.set_value(paddle.to_tensor(state_dict["llama.norm.weight"]).cast(self.norm.weight.dtype))

        for idx in range(self.config.num_hidden_layers):
            logger.info(f"set state for layer {idx}")
            self.transformer_block.ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.input_layernorm.weight".format(idx)]).cast(
                    self.transformer_block.ln_scales[idx].dtype
                )
            )
            if "llama.layers.{}.self_attn.qkv_proj.weight".format(idx) in state_dict.keys():
                concated_qkv_weight = np.concatenate(
                    split_fn(
                        state_dict["llama.layers.{}.self_attn.qkv_proj.weight".format(idx)],
                        is_qkv=True,
                        num_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                        num_key_value_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                    ),
                    axis=-1,
                )
            else:
                unfused_state_dict = {}
                unfused_state_dict["self_attn.q_proj.weight"] = state_dict[
                    "llama.layers.{}.self_attn.q_proj.weight".format(idx)
                ]
                unfused_state_dict["self_attn.k_proj.weight"] = state_dict[
                    "llama.layers.{}.self_attn.k_proj.weight".format(idx)
                ]
                unfused_state_dict["self_attn.v_proj.weight"] = state_dict[
                    "llama.layers.{}.self_attn.v_proj.weight".format(idx)
                ]
                concated_qkv_weight = np.concatenate(
                    [
                        unfused_state_dict["self_attn.q_proj.weight"],
                        unfused_state_dict["self_attn.k_proj.weight"],
                        unfused_state_dict["self_attn.v_proj.weight"],
                    ],
                    axis=-1,
                ).reshape(
                    self.hidden_size,
                    3 * (self.num_attention_heads // self.config.tensor_parallel_degree) * (head_size),
                )  # reshape(3, self.num_attention_heself.hidden_sizeads // self.config.tensor_parallel_degree, head_size, )
            if "llama.layers.{}.mlp.gate_up_fused_proj.weight".format(idx) in state_dict.keys():
                concated_ffn1_weight = np.concatenate(
                    split_fn(state_dict["llama.layers.{}.mlp.gate_up_fused_proj.weight".format(idx)]), axis=-1
                )
            else:
                unfused_state_dict["mlp.gate_proj.weight"] = state_dict[
                    "llama.layers.{}.mlp.gate_proj.weight".format(idx)
                ]
                unfused_state_dict["mlp.up_proj.weight"] = state_dict["llama.layers.{}.mlp.up_proj.weight".format(idx)]
                concated_ffn1_weight = np.concatenate(
                    [unfused_state_dict["mlp.gate_proj.weight"], unfused_state_dict["mlp.up_proj.weight"]], axis=-1
                )
            gate_up_list = split_fn(concated_ffn1_weight)
            gate_weight_tensor = paddle.to_tensor(gate_up_list[0])
            up_weight_tensor = paddle.to_tensor(gate_up_list[1])

            qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight)
            self.transformer_block.qkv_weights[idx].set_value(
                qkv_weight_tensor.cast(self.transformer_block.qkv_weights[idx].dtype)
            )

            linear_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)])
            self.transformer_block.linear_weights[idx].set_value(
                linear_weight_tensor.cast(self.transformer_block.linear_weights[idx].dtype)
            )
            # mlp
            self.transformer_block.ffn_ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.post_attention_layernorm.weight".format(idx)]).cast(
                    self.transformer_block.ffn_ln_scales[idx].dtype
                )
            )
            self.transformer_block.gate_weights[idx].set_value(
                gate_weight_tensor.cast(self.transformer_block.gate_weights[idx].dtype)
            )
            self.transformer_block.up_weights[idx].set_value(
                up_weight_tensor.cast(self.transformer_block.up_weights[idx].dtype)
            )

            ffn2_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)])
            self.transformer_block.ffn2_weights[idx].set_value(
                ffn2_weight_tensor.cast(self.transformer_block.ffn2_weights[idx].dtype)
            )


@register_base_model
class LlamaInferenceModel(LlamaPretrainedModel):
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
        self.num_key_value_heads = config.num_key_value_heads
        self.head_size = self.hidden_size // self.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.epsilon = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.quant_type = config.get("quant_type", "")

        self.rope_theta = config.rope_theta
        self.use_neox = True

        self.use_fake_parameter = config.get("use_fake_parameter", False)

        self.use_weight_only = False
        if config.quant_type == "weight_only_int8":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int8"
        elif config.quant_type == "weight_only_int4":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int4"
        elif "a8w8" in config.quant_type:
            self.quant_model_path = config.model_name_or_path
            self.shift = config.quantization_config.shift
            self.smooth = config.quantization_config.smooth
            self.shift_smooth_all_linears = config.quantization_config.shift_smooth_all_linears
            if self.use_fake_parameter:
                self.shift_smooth_all_linears = True
                config.quantization_config.shift_smooth_all_linears = True

        if self.use_weight_only:
            assert (
                self.quant_type == "weight_only_int8" or self.quant_type == "weight_only_int4"
            ), "Expected quant_type equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_type
            )

        if config.tensor_parallel_degree > 1 and config.vocab_size % config.tensor_parallel_degree == 0:
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

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None

        qkv_out_scale_attrs = None
        linear_out_scale_attrs = None
        ffn1_out_scale_attrs = None
        ffn2_out_scale_attrs = None
        linear_shift_attrs = None
        linear_smooth_attrs = None
        ffn2_shift_attrs = None
        ffn2_smooth_attrs = None

        ln_bias_attrs = None
        qkv_bias_attrs = None
        out_proj_bias_attrs = None
        ffn_ln_bias_attrs = None
        ffn1_bias_attrs = None
        ffn2_bias_attrs = None

        ffn1_0_weight_attrs = None
        ffn1_1_weight_attrs = None
        ffn1_0_bias_attrs = None
        ffn1_1_bias_attrs = None

        ffn1_weight_attrs = None
        ffn2_weight_attrs = None

        ln_scale_attrs = [paddle.ParamAttr(name="fusellama.{}.ln_scale".format(i)) for i in range(self.num_layers)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fusellama.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        if "fp8" in self.quant_type:
            ffn1_0_weight_attrs = [
                paddle.ParamAttr(
                    name="fusellama.{}.ffn1_0_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
                )
                for i in range(self.num_layers)
            ]
            ffn1_1_weight_attrs = [
                paddle.ParamAttr(
                    name="fusellama.{}.ffn1_1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
                )
                for i in range(self.num_layers)
            ]
        else:
            ffn1_weight_attrs = [
                paddle.ParamAttr(
                    name="fusellama.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
                )
                for i in range(self.num_layers)
            ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fusellama.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        if "a8w8" in self.quant_type:
            qkv_out_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.qkv_out_scale".format(i)) for i in range(self.num_layers)
            ]
            linear_out_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.linear_out_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn1_out_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn1_out_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn2_out_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn2_out_scale".format(i)) for i in range(self.num_layers)
            ]

            if self.shift_smooth_all_linears:
                linear_shift_attrs = [
                    paddle.ParamAttr(name="fusellama.{}.linear_shift".format(i)) for i in range(self.num_layers)
                ]
                linear_smooth_attrs = [
                    paddle.ParamAttr(name="fusellama.{}.linear_smooth".format(i)) for i in range(self.num_layers)
                ]
                ffn2_shift_attrs = [
                    paddle.ParamAttr(name="fusellama.{}.ffn2_shift".format(i)) for i in range(self.num_layers)
                ]
                ffn2_smooth_attrs = [
                    paddle.ParamAttr(name="fusellama.{}.ffn2_smooth".format(i)) for i in range(self.num_layers)
                ]

            if self.shift:
                ln_bias_attrs = [
                    paddle.ParamAttr(name="fusellama.{}.ln_bias".format(i)) for i in range(self.num_layers)
                ]
                ffn_ln_bias_attrs = [
                    paddle.ParamAttr(name="fusellama.{}.ffn_ln_bias".format(i)) for i in range(self.num_layers)
                ]
                qkv_bias_attrs = [
                    paddle.ParamAttr(name="fusellama.{}.qkv_bias".format(i)) for i in range(self.num_layers)
                ]
                ffn1_bias_attrs = [
                    paddle.ParamAttr(name="fusellama.{}.ffn1_bias".format(i)) for i in range(self.num_layers)
                ]
                if self.shift_smooth_all_linears:
                    out_proj_bias_attrs = [
                        paddle.ParamAttr(name="fusellama.{}.out_proj_bias".format(i)) for i in range(self.num_layers)
                    ]
                    ffn2_bias_attrs = [
                        paddle.ParamAttr(name="fusellama.{}.ffn2_bias".format(i)) for i in range(self.num_layers)
                    ]

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None

        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.qkv_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            out_proj_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.out_proj_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn1_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.ffn2_weight_scale".format(i)) for i in range(self.num_layers)
            ]

        cache_k_scale_attrs = None
        cache_v_scale_attrs = None
        cache_k_out_scale_attrs = None
        cache_v_out_scale_attrs = None

        if config.cachekv_int8_type == "static":
            cache_k_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.cache_k_scale".format(i)) for i in range(self.num_layers)
            ]
            cache_v_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.cache_v_scale".format(i)) for i in range(self.num_layers)
            ]
            cache_k_out_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.cache_k_out_scale".format(i)) for i in range(self.num_layers)
            ]
            cache_v_out_scale_attrs = [
                paddle.ParamAttr(name="fusellama.{}.cache_v_out_scale".format(i)) for i in range(self.num_layers)
            ]

        speculate_config = SpeculateConfig(
            speculate_method=config.get("speculate_method", None),
            speculate_max_draft_token_num=config.get("speculate_max_draft_token_num", 5),
        )
        transformer_config = FusedMultiTransformerConfig(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            dim_feedforward=self.intermediate_size,
            quant_type=self.quant_type,
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
            ffn1_0_weight_attrs=ffn1_0_weight_attrs,
            ffn1_1_weight_attrs=ffn1_1_weight_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            qkv_out_scale_attrs=qkv_out_scale_attrs,
            linear_out_scale_attrs=linear_out_scale_attrs,
            ffn1_out_scale_attrs=ffn1_out_scale_attrs,
            ffn2_out_scale_attrs=ffn2_out_scale_attrs,
            linear_shift_attrs=linear_shift_attrs,
            linear_smooth_attrs=linear_smooth_attrs,
            ffn2_shift_attrs=ffn2_shift_attrs,
            ffn2_smooth_attrs=ffn2_smooth_attrs,
            ln_bias_attrs=ln_bias_attrs,
            qkv_bias_attrs=qkv_bias_attrs,
            linear_bias_attrs=out_proj_bias_attrs,
            ffn_ln_bias_attrs=ffn_ln_bias_attrs,
            ffn1_bias_attrs=ffn1_bias_attrs,
            ffn1_0_bias_attrs=ffn1_0_bias_attrs,
            ffn1_1_bias_attrs=ffn1_1_bias_attrs,
            ffn2_bias_attrs=ffn2_bias_attrs,
            cache_k_scale_attrs=cache_k_scale_attrs,
            cache_v_scale_attrs=cache_v_scale_attrs,
            cache_k_out_scale_attrs=cache_k_out_scale_attrs,
            cache_v_out_scale_attrs=cache_v_out_scale_attrs,
            epsilon=self.epsilon,
            norm_type="rmsnorm",
            use_neox_rotary_style=self.use_neox,
            cachekv_int8_type=config.cachekv_int8_type,
            rank_id=config.tensor_parallel_rank,
            trans_qkvw=(False if paddle.is_compiled_with_rocm() and "a8w8" in self.quant_type else True),
            append_attn=config.append_attn,
            speculate_config=speculate_config,
        )

        self.set_transformer_block(transformer_config)
        self.norm = FusedLlamaRMSNorm(config)

        self.cache_kvs = None
        self.head_dim_shape_tensor = paddle.ones((self.hidden_size // self.num_attention_heads), dtype="int8")

        self.gradient_checkpointing = False

    def set_transformer_block(self, transformer_config):
        if self.use_weight_only:
            self.transformer_block = FusedMultiTransformerWeightOnly(transformer_config)
        elif "a8w8" in self.quant_type:
            self.transformer_block = FusedMultiTransformerA8W8(transformer_config)
        else:
            self.transformer_block = FusedMultiTransformerBase(transformer_config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        from paddlenlp_ops import get_padding_offset

        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    # This function is a little different from prepare_input_ids_for_generation in paddlenlp/transformers/generation/utils.py
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.ones([batch_size, seq_len], dtype="int64") * bos_token_id

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        cache_kvs=None,
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):
        # kwargs["cache"] is used used to distinguish between encoder and decoder phase.
        past_key_values = kwargs.get("cache", None)
        is_decoder = past_key_values is not None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # genereate a fake input_ids according to inputs_embeds
        # this is usually occurred in img2txt multimodal model when first enter into this forward function.
        if input_ids is None and inputs_embeds is not None:
            input_ids = self.prepare_input_ids_for_generation(self.config.bos_token_id, inputs_embeds)
        if inputs_embeds is not None:
            batch, seq_len, hidden_dim = inputs_embeds.shape
            # merge batch and seq_len dimension.
            inputs_embeds = inputs_embeds.reshape([batch * seq_len, hidden_dim])

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
            ids_remove_padding = input_ids.squeeze(axis=1)
            padding_offset = None
            cum_offsets = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ids_remove_padding)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder

        position_offset = 0
        if not is_decoder and pre_caches is not None:
            position_offset = 128
        from paddlenlp_ops import fused_get_rotary_embedding

        new_rope = fused_get_rotary_embedding(
            input_ids, position_ids, self.head_dim_shape_tensor, position_offset, self.rope_theta, self.use_neox
        )

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids,
                hidden_states,
                cum_offsets=cum_offsets,
                padding_offset=padding_offset,
                attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                caches=cache_kvs,
                pre_caches=pre_caches,
                pre_caches_length=position_offset,
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
    def set_quant_scale(self):
        current_work_dir = os.path.dirname(__file__)
        if "fp8" in self.quant_type:
            scale_map_file = f"{current_work_dir}/ptq_fp8_scales_map.json"
            with open(scale_map_file) as json_file:
                scale_map_dict = json.load(json_file)
                act_scale_map_dict = scale_map_dict["act_scale"]
                weight_scale_map_dict = scale_map_dict["weight_scale"]
                cache_scale_map_dict = scale_map_dict["cachekv_scale"]
                act_scale_json_path = resolve_file_path(self.quant_model_path, "act_scales.json")
                weight_scale_json_path = resolve_file_path(self.quant_model_path, "weight_scales.json")
                if self.config.tensor_parallel_degree > 1 and not self.config.single_card_ptq:
                    act_scale_json_path = resolve_file_path(
                        self.quant_model_path, f"act_scales_{self.config.tensor_parallel_rank}.json"
                    )
                    weight_scale_json_path = resolve_file_path(
                        self.quant_model_path, f"weight_scales_{self.config.tensor_parallel_rank}.json"
                    )

                act_scales = ActScalesLoader(
                    act_scale_json_path, act_scale_map_dict, num_of_layers=self.config.num_hidden_layers
                )

                weight_scales = PerTensorWeightScalesLoader(
                    weight_scale_json_path,
                    weight_scale_map_dict,
                    num_of_layers=self.config.num_hidden_layers,
                )

                for weight_name in weight_scales.scale:
                    weight_scales.scale[weight_name] = weight_scales.scale[weight_name].astype(np.float32)
                for act_name in act_scales.scale:
                    act_scales.scale[act_name] = act_scales.scale[act_name].astype(np.float32)
                self.transformer_block.weight_scales = weight_scales.scale
                self.transformer_block.act_scales = act_scales.scale
        elif "a8w8" in self.quant_type:
            scale_map_file = (
                f"{current_work_dir}/ptq_scales_map.json"
                if not self.shift_smooth_all_linears
                else f"{current_work_dir}/ptq_scales_map_shift_smooth.json"
            )
            with open(scale_map_file) as json_file:
                scale_map_dict = json.load(json_file)
                act_scale_map_dict = scale_map_dict["act_scale"]
                weight_scale_map_dict = scale_map_dict["weight_scale"]
                cache_scale_map_dict = scale_map_dict["cachekv_scale"]
                if not self.use_fake_parameter:
                    act_scale_json_path = resolve_file_path(self.quant_model_path, "act_scales.json")
                    weight_scale_json_path = resolve_file_path(self.quant_model_path, "weight_scales.json")
                    if self.config.tensor_parallel_degree > 1 and not self.config.single_card_ptq:
                        act_scale_json_path = resolve_file_path(
                            self.quant_model_path, f"act_scales_{self.config.tensor_parallel_rank}.json"
                        )
                        weight_scale_json_path = resolve_file_path(
                            self.quant_model_path, f"weight_scales_{self.config.tensor_parallel_rank}.json"
                        )
                    act_scale_loader = ActScalesLoader(
                        act_scale_json_path, act_scale_map_dict, num_of_layers=self.config.num_hidden_layers
                    )
                    weight_scales_loader = WeightScalesLoader(
                        weight_scale_json_path,
                        weight_scale_map_dict,
                        num_of_layers=self.config.num_hidden_layers,
                        concat_qkv=True,
                        concat_ffn1=True,
                    )
                else:
                    act_scale_loader = EmptyActScale(act_scale_map_dict, num_of_layers=self.config.num_hidden_layers)
                    weight_scales_loader = EmptyWeightScale(
                        weight_scale_map_dict,
                        num_of_layers=self.config.num_hidden_layers,
                        num_heads=self.num_attention_heads,
                        dim_head=self.hidden_size // self.num_attention_heads,
                        ffn_hidden_size=self.intermediate_size,
                        num_key_value_heads=self.num_key_value_heads,
                        mp_size=self.config.tensor_parallel_degree,
                        concat_qkv=True,
                        concat_ffn1=True,
                    )
                self.transformer_block.weight_scales = weight_scales_loader.scale
                self.transformer_block.act_scales = act_scale_loader.scale

            for k, v in weight_scales_loader.scale.items():
                if "qkv_" in k:
                    for i_layer, weight_scale in enumerate(v):
                        if not np.all(weight_scale == -1):
                            tmp = paddle.to_tensor(
                                weight_scale
                                / (
                                    127.0 * 127.0 * act_scale_loader.scale["qkv_in_scale"][i_layer]
                                )  # [3 * num_head * dim_head]
                            ).reshape([-1])
                            if self.config.tensor_parallel_degree > 1 and self.config.single_card_ptq:
                                tmp = (
                                    tmp.reshape([3, self.num_attention_heads, self.head_size])
                                    .split(self.config.tensor_parallel_degree, axis=1)[
                                        self.config.tensor_parallel_rank
                                    ]
                                    .reshape([-1])
                                )
                            self.transformer_block.qkv_out_scales[i_layer].set_value(tmp)
                elif "out_linear_" in k:
                    for i_layer, weight_scale in enumerate(v):
                        if not np.all(weight_scale == -1):
                            tmp = paddle.to_tensor(
                                weight_scale / (127.0 * 127.0 * act_scale_loader.scale["out_linear_in_scale"][i_layer])
                            )
                            self.transformer_block.linear_out_scales[i_layer].set_value(tmp)
                elif "ffn1_weight_scale" in k:
                    for i_layer, weight_scale in enumerate(v):
                        if not np.all(weight_scale == -1):
                            tmp = paddle.to_tensor(
                                weight_scale / (127.0 * 127.0 * act_scale_loader.scale["ffn1_in_scale"][i_layer])
                            )
                            if self.config.tensor_parallel_degree > 1 and self.config.single_card_ptq:
                                tmp = paddle.split(tmp, self.config.tensor_parallel_degree * 2)
                                tmp = paddle.concat(
                                    [
                                        tmp[self.config.tensor_parallel_rank],
                                        tmp[self.config.tensor_parallel_rank + self.config.tensor_parallel_degree],
                                    ],
                                    axis=0,
                                )
                            self.transformer_block.ffn1_out_scales[i_layer].set_value(tmp)
                elif "ffn2" in k:
                    for i_layer, weight_scale in enumerate(v):
                        if not np.all(weight_scale == -1):
                            self.transformer_block.ffn2_out_scales[i_layer].set_value(
                                paddle.to_tensor(
                                    weight_scale / (127.0 * 127.0 * act_scale_loader.scale["ffn2_in_scale"][i_layer])
                                )
                            )

        if self.config.cachekv_int8_type == "static":
            if not self.use_fake_parameter:
                cache_scale_json_path = resolve_file_path(self.quant_model_path, "cachekv_scales.json")
                if self.config.tensor_parallel_degree > 1 and not self.config.single_card_ptq:
                    cache_scale_json_path = resolve_file_path(
                        self.quant_model_path, f"cachekv_scales_{self.config.tensor_parallel_rank}.json"
                    )
                cache_scales_loader = CacheScaleLoader(
                    cache_scale_json_path,
                    cache_scale_map_dict,
                    num_of_layers=self.config.num_hidden_layers,
                    num_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                    num_key_value_heads=self.num_key_value_heads // self.config.tensor_parallel_degree,
                )
            else:
                cache_scales_loader = EmptyCacheScale(
                    cache_scale_map_dict,
                    num_of_layers=self.config.num_hidden_layers,
                    num_heads=self.num_attention_heads,
                    dim_heads=self.hidden_size // self.num_attention_heads,
                    is_channel_wise=False,
                    num_key_value_heads=self.num_key_value_heads,
                    mp_size=self.config.tensor_parallel_degree,
                )

            for k, v in cache_scales_loader.scale.items():
                for i_layer, weight_scale in enumerate(v):
                    if self.config.append_attn:
                        weight_scale = paddle.to_tensor(weight_scale).cast(paddle.get_default_dtype())
                    else:
                        weight_scale = weight_scale.astype("float32")
                    if k == "cache_k_scale":
                        self.transformer_block.cache_k_scales[i_layer].set_value(weight_scale)
                    elif k == "cache_v_scale":
                        self.transformer_block.cache_v_scales[i_layer].set_value(weight_scale)
                    elif k == "cache_k_out_scale":
                        self.transformer_block.cache_k_out_scales[i_layer].set_value(weight_scale)
                    else:
                        self.transformer_block.cache_v_out_scales[i_layer].set_value(weight_scale)

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.set_quant_scale()
        self.transformer_block.init_weight()
        split_fn = split_param_func()
        self.embed_tokens.weight.set_value(
            paddle.to_tensor(state_dict["llama.embed_tokens.weight"]).cast(self.embed_tokens.weight.dtype)
        )
        self.norm.weight.set_value(paddle.to_tensor(state_dict["llama.norm.weight"]).cast(self.norm.weight.dtype))
        if self.use_weight_only:
            logger.info("weight only is enabled")
        for idx in range(self.config.num_hidden_layers):
            logger.info(f"set state for layer {idx}")

            self.transformer_block.ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.input_layernorm.weight".format(idx)]).cast(
                    self.transformer_block.ln_scales[idx].dtype
                )
            )
            if "llama.layers.{}.self_attn.qkv_proj.weight".format(idx) in state_dict.keys():
                concated_qkv_weight = paddle.to_tensor(
                    np.concatenate(
                        split_fn(
                            state_dict["llama.layers.{}.self_attn.qkv_proj.weight".format(idx)],
                            is_qkv=True,
                            num_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                            num_key_value_heads=self.num_key_value_heads // self.config.tensor_parallel_degree,
                        ),
                        axis=-1,
                    ).transpose(1, 0)
                )
            else:
                unfused_state_dict = {}
                unfused_state_dict["self_attn.q_proj.weight"] = paddle.to_tensor(
                    state_dict["llama.layers.{}.self_attn.q_proj.weight".format(idx)]
                )
                unfused_state_dict["self_attn.k_proj.weight"] = paddle.to_tensor(
                    state_dict["llama.layers.{}.self_attn.k_proj.weight".format(idx)]
                )
                unfused_state_dict["self_attn.v_proj.weight"] = paddle.to_tensor(
                    state_dict["llama.layers.{}.self_attn.v_proj.weight".format(idx)]
                )
                if "fp8" in self.quant_type:
                    q_wgt_scale = self.transformer_block.weight_scales["q_weight_scale"][idx]
                    k_wgt_scale = self.transformer_block.weight_scales["k_weight_scale"][idx]
                    v_wgt_scale = self.transformer_block.weight_scales["v_weight_scale"][idx]
                    qkv_wgt_scale = self.transformer_block.weight_scales["qkv_weight_scale"][idx]
                    unfused_state_dict["self_attn.q_proj.weight"] = (
                        unfused_state_dict["self_attn.q_proj.weight"].cast("float32") * q_wgt_scale / qkv_wgt_scale
                    )
                    unfused_state_dict["self_attn.k_proj.weight"] = (
                        unfused_state_dict["self_attn.k_proj.weight"].cast("float32") * k_wgt_scale / qkv_wgt_scale
                    )
                    unfused_state_dict["self_attn.v_proj.weight"] = (
                        unfused_state_dict["self_attn.v_proj.weight"].cast("float32") * v_wgt_scale / qkv_wgt_scale
                    )
                if paddle.is_compiled_with_rocm() and "a8w8" in self.quant_type and "fp8" not in self.quant_type:
                    concated_qkv_weight = paddle.concat(
                        [
                            unfused_state_dict["self_attn.q_proj.weight"],
                            unfused_state_dict["self_attn.k_proj.weight"],
                            unfused_state_dict["self_attn.v_proj.weight"],
                        ],
                        axis=-1,
                    ).reshape(
                        [
                            self.hidden_size,
                            (
                                self.num_attention_heads // self.config.tensor_parallel_degree
                                + 2 * self.num_key_value_heads // self.config.tensor_parallel_degree
                            )
                            * (self.head_size),
                        ]
                    )
                else:
                    concated_qkv_weight = (
                        paddle.concat(
                            [
                                unfused_state_dict["self_attn.q_proj.weight"],
                                unfused_state_dict["self_attn.k_proj.weight"],
                                unfused_state_dict["self_attn.v_proj.weight"],
                            ],
                            axis=-1,
                        )
                        .transpose([1, 0])
                        .reshape(
                            [
                                (
                                    self.num_attention_heads // self.config.tensor_parallel_degree
                                    + 2 * self.num_key_value_heads // self.config.tensor_parallel_degree
                                )
                                * (self.head_size),
                                self.hidden_size,
                            ]
                        )
                    )
            qkv_weight_tensor = paddle.to_tensor(concated_qkv_weight).cast(paddle.get_default_dtype())
            if self.use_weight_only:
                qkv_weight_tensor = paddle.transpose(qkv_weight_tensor, perm=[1, 0])
                qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                    qkv_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight_tensor)
                self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale_tensor)
            elif "fp8" in self.quant_type:
                self.transformer_block.qkv_weights[idx].copy_(paddle.cast(concated_qkv_weight, "float8_e4m3fn"), False)
            elif "a8w8" in self.quant_type and not self.transformer_block.skip_quant("qkv_weight_scale", idx):
                self.transformer_block.qkv_weights[idx].set_value(
                    paddle.cast(paddle.to_tensor(concated_qkv_weight), "int8")
                )
            else:
                self.transformer_block.qkv_weights[idx].set_value(qkv_weight_tensor)

            linear_weight_tensor = paddle.to_tensor(
                state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)]
            ).cast(paddle.get_default_dtype())
            if self.use_weight_only:
                linear_quanted_weight_tensor, linear_weight_scale_tensor = weight_quantize(
                    linear_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight_tensor)
                self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale_tensor)
            elif "fp8" in self.quant_type:
                self.transformer_block.linear_weights[idx].copy_(
                    paddle.cast(
                        paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)]).transpose(
                            (1, 0)
                        ),
                        "float8_e4m3fn",
                    ),
                    False,
                )
            elif "a8w8" in self.quant_type:
                w_dtype = (
                    paddle.get_default_dtype()
                    if self.transformer_block.skip_quant("out_linear_weight_scale", idx)
                    else "int8"
                )
                if paddle.is_compiled_with_rocm():
                    self.transformer_block.linear_weights[idx].set_value(
                        paddle.cast(
                            paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)]),
                            w_dtype,
                        )
                    )
                else:
                    self.transformer_block.linear_weights[idx].set_value(
                        paddle.cast(
                            paddle.to_tensor(
                                state_dict["llama.layers.{}.self_attn.o_proj.weight".format(idx)]
                            ).transpose((1, 0)),
                            w_dtype,
                        )
                    )
            else:
                self.transformer_block.linear_weights[idx].set_value(linear_weight_tensor)

            self.transformer_block.ffn_ln_scales[idx].set_value(
                paddle.to_tensor(state_dict["llama.layers.{}.post_attention_layernorm.weight".format(idx)]).cast(
                    self.transformer_block.ffn_ln_scales[idx].dtype
                )
            )

            if "llama.layers.{}.mlp.gate_up_fused_proj.weight".format(idx) in state_dict.keys():
                concated_ffn1_weight = np.concatenate(
                    split_fn(state_dict["llama.layers.{}.mlp.gate_up_fused_proj.weight".format(idx)]), axis=-1
                )
            else:
                unfused_state_dict["mlp.gate_proj.weight"] = state_dict[
                    "llama.layers.{}.mlp.gate_proj.weight".format(idx)
                ]
                unfused_state_dict["mlp.up_proj.weight"] = state_dict["llama.layers.{}.mlp.up_proj.weight".format(idx)]
                concated_ffn1_weight = np.concatenate(
                    [unfused_state_dict["mlp.gate_proj.weight"], unfused_state_dict["mlp.up_proj.weight"]], axis=-1
                )
            ffn1_weight_tensor = paddle.to_tensor(concated_ffn1_weight).cast(paddle.get_default_dtype())

            if self.use_weight_only:
                ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                    ffn1_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight_tensor)
                self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale_tensor)
            elif "fp8" in self.quant_type:
                self.transformer_block.ffn1_0_weights[idx].copy_(
                    paddle.to_tensor(unfused_state_dict["mlp.gate_proj.weight"])
                    .transpose((1, 0))
                    .cast("float8_e4m3fn"),
                    False,
                )
                self.transformer_block.ffn1_1_weights[idx].copy_(
                    paddle.to_tensor(unfused_state_dict["mlp.up_proj.weight"]).transpose((1, 0)).cast("float8_e4m3fn"),
                    False,
                )
            elif "a8w8" in self.quant_type:
                w_dtype = (
                    paddle.get_default_dtype()
                    if self.transformer_block.skip_quant("ffn1_weight_scale", idx)
                    else "int8"
                )
                if paddle.is_compiled_with_rocm():
                    self.transformer_block.ffn1_weights[idx].set_value(
                        paddle.cast(paddle.to_tensor(concated_ffn1_weight), w_dtype)
                    )
                else:
                    self.transformer_block.ffn1_weights[idx].set_value(
                        paddle.cast(paddle.to_tensor(concated_ffn1_weight).transpose((1, 0)), w_dtype)
                    )
            else:
                self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight_tensor)

            ffn2_weight_tensor = paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)]).cast(
                paddle.get_default_dtype()
            )
            if self.use_weight_only:
                ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                    ffn2_weight_tensor, algo=self.quant_algo
                )
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
                self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)
            elif "fp8" in self.quant_type:
                self.transformer_block.ffn2_weights[idx].copy_(
                    paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)])
                    .transpose([1, 0])
                    .cast("float8_e4m3fn"),
                    False,
                )
            elif "a8w8" in self.quant_type:
                w_dtype = (
                    paddle.get_default_dtype()
                    if self.transformer_block.skip_quant("ffn2_weight_scale", idx)
                    else "int8"
                )
                if paddle.is_compiled_with_rocm():
                    self.transformer_block.ffn2_weights[idx].set_value(
                        paddle.cast(
                            paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)]), w_dtype
                        )
                    )
                else:
                    self.transformer_block.ffn2_weights[idx].set_value(
                        paddle.cast(
                            paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.weight".format(idx)]).transpose(
                                (1, 0)
                            ),
                            w_dtype,
                        )
                    )
            else:
                self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight_tensor)

            if "fp8" not in self.quant_type and "a8w8" in self.quant_type:
                if self.shift_smooth_all_linears:
                    if self.use_fake_parameter:
                        if "llama.layers.{}.self_attn.o_proj.shift_bias".format(idx) not in state_dict:
                            state_dict["llama.layers.{}.self_attn.o_proj.shift_bias".format(idx)] = paddle.zeros(
                                shape=[
                                    (self.num_attention_heads // self.config.tensor_parallel_degree)
                                    * (self.hidden_size // self.num_attention_heads)
                                ],
                                dtype=paddle.get_default_dtype(),
                            )
                            state_dict["llama.layers.{}.self_attn.o_proj.smooth_weight".format(idx)] = paddle.ones(
                                shape=[
                                    (self.num_attention_heads // self.config.tensor_parallel_degree)
                                    * (self.hidden_size // self.num_attention_heads)
                                ],
                                dtype=paddle.get_default_dtype(),
                            )
                            state_dict["llama.layers.{}.mlp.down_proj.shift_bias".format(idx)] = paddle.zeros(
                                shape=[self.intermediate_size // self.config.tensor_parallel_degree],
                                dtype=paddle.get_default_dtype(),
                            )
                            state_dict["llama.layers.{}.mlp.down_proj.smooth_weight".format(idx)] = paddle.ones(
                                shape=[self.intermediate_size // self.config.tensor_parallel_degree],
                                dtype=paddle.get_default_dtype(),
                            )
                    self.transformer_block.linear_shifts[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.shift_bias".format(idx)]).astype(
                            paddle.get_default_dtype()
                        )
                    )
                    self.transformer_block.linear_smooths[idx].set_value(
                        paddle.to_tensor(
                            state_dict["llama.layers.{}.self_attn.o_proj.smooth_weight".format(idx)]
                        ).astype(paddle.get_default_dtype())
                    )
                    self.transformer_block.ffn2_shifts[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.shift_bias".format(idx)]).astype(
                            paddle.get_default_dtype()
                        )
                    )
                    self.transformer_block.ffn2_smooths[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.smooth_weight".format(idx)]).astype(
                            paddle.get_default_dtype()
                        )
                    )

                if self.shift:
                    if self.use_fake_parameter:
                        if "llama.layers.{}.input_layernorm.bias".format(idx) not in state_dict:
                            state_dict["llama.layers.{}.input_layernorm.bias".format(idx)] = paddle.zeros(
                                shape=[self.hidden_size], dtype=paddle.get_default_dtype()
                            )
                            state_dict["llama.layers.{}.post_attention_layernorm.bias".format(idx)] = paddle.zeros(
                                [self.hidden_size], dtype=paddle.get_default_dtype()
                            )
                            unfused_state_dict["self_attn.q_proj.bias"] = paddle.zeros(
                                shape=[self.num_attention_heads * (self.hidden_size // self.num_attention_heads)],
                                dtype=paddle.get_default_dtype(),
                            )
                            unfused_state_dict["self_attn.k_proj.bias"] = paddle.zeros(
                                shape=[self.num_key_value_heads * (self.hidden_size // self.num_attention_heads)],
                                dtype=paddle.get_default_dtype(),
                            )
                            unfused_state_dict["self_attn.v_proj.bias"] = paddle.zeros(
                                shape=[self.num_key_value_heads * (self.hidden_size // self.num_attention_heads)],
                                dtype=paddle.get_default_dtype(),
                            )
                            unfused_state_dict["mlp.gate_proj.bias"] = paddle.zeros(
                                shape=[self.intermediate_size], dtype=paddle.get_default_dtype()
                            )
                            unfused_state_dict["mlp.up_proj.bias"] = paddle.zeros(
                                shape=[self.intermediate_size], dtype=paddle.get_default_dtype()
                            )
                    else:
                        unfused_state_dict["self_attn.q_proj.bias"] = state_dict[
                            "llama.layers.{}.self_attn.q_proj.bias".format(idx)
                        ]
                        unfused_state_dict["self_attn.k_proj.bias"] = state_dict[
                            "llama.layers.{}.self_attn.k_proj.bias".format(idx)
                        ]
                        unfused_state_dict["self_attn.v_proj.bias"] = state_dict[
                            "llama.layers.{}.self_attn.v_proj.bias".format(idx)
                        ]
                        unfused_state_dict["mlp.gate_proj.bias"] = state_dict[
                            "llama.layers.{}.mlp.gate_proj.bias".format(idx)
                        ]
                        unfused_state_dict["mlp.up_proj.bias"] = state_dict[
                            "llama.layers.{}.mlp.up_proj.bias".format(idx)
                        ]

                    self.transformer_block.ln_biases[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.input_layernorm.bias".format(idx)])
                    )
                    self.transformer_block.ffn_ln_biases[idx].set_value(
                        paddle.to_tensor(state_dict["llama.layers.{}.post_attention_layernorm.bias".format(idx)])
                    )

                    concated_qkv_biases = np.concatenate(
                        [
                            unfused_state_dict["self_attn.q_proj.bias"],
                            unfused_state_dict["self_attn.k_proj.bias"],
                            unfused_state_dict["self_attn.v_proj.bias"],
                        ],
                        axis=-1,
                    )

                    self.transformer_block.qkv_biases[idx].set_value(paddle.to_tensor(concated_qkv_biases))

                    concated_ffn1_bias = np.concatenate(
                        [unfused_state_dict["mlp.gate_proj.bias"], unfused_state_dict["mlp.up_proj.bias"]], axis=-1
                    )

                    self.transformer_block.ffn1_biases[idx].set_value(paddle.to_tensor(concated_ffn1_bias))

                    if self.shift_smooth_all_linears:
                        if self.use_fake_parameter:
                            if "llama.layers.{}.self_attn.o_proj.bias".format(idx) not in state_dict:
                                state_dict["llama.layers.{}.self_attn.o_proj.bias".format(idx)] = paddle.zeros(
                                    [self.hidden_size], dtype=paddle.get_default_dtype()
                                )
                                state_dict["llama.layers.{}.mlp.down_proj.layer.bias".format(idx)] = paddle.zeros(
                                    [self.hidden_size], dtype=paddle.get_default_dtype()
                                )
                        self.transformer_block.linear_biases[idx].set_value(
                            paddle.to_tensor(state_dict["llama.layers.{}.self_attn.o_proj.bias".format(idx)])
                        )
                        self.transformer_block.ffn2_biases[idx].set_value(
                            paddle.to_tensor(state_dict["llama.layers.{}.mlp.down_proj.layer.bias".format(idx)])
                        )


@register_base_model
class LlamaBlockInferenceModel(LlamaInferenceModel):
    def __init__(self, config: LlamaConfig):
        self.append_attn = config.append_attn
        super().__init__(config)
        self.max_seq_len = config.max_seq_len
        self.block_size = config.block_size

    def set_transformer_block(self, transformer_config):
        if self.use_weight_only:
            self.transformer_block = FusedBlockMultiTransformerWeightOnly(transformer_config)
        elif self.quant_type == "a8w8" or self.quant_type == "a8w8c8":
            self.transformer_block = FusedBlockMultiTransformerA8W8(transformer_config)
        elif "fp8" in self.quant_type:
            self.transformer_block = FusedBlockMultiTransformerFP8(transformer_config)
        else:
            self.transformer_block = FusedBlockMultiTransformer(transformer_config)

    def remove_padding(self, input_ids, seq_lens_this_time, draft_tokens=None, seq_lens_encoder=None):
        cum_offsets_now = paddle.cumsum(self.max_seq_len - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        from paddlenlp_ops import get_padding_offset_v2

        ids_remove_padding, cum_offsets, padding_offset, cu_seqlens_q, cu_seqlens_k = get_padding_offset_v2(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time, draft_tokens, seq_lens_encoder
        )
        return ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        caches=None,
        pre_caches=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):

        seq_lens_this_time = kwargs.get("seq_lens_this_time", None)
        rope_emb = kwargs.get("rope_emb", None)
        draft_tokens = kwargs.get("draft_tokens", None)
        seq_lens_encoder = kwargs.get("seq_lens_encoder", None)

        # whether speculative decoding or not
        if draft_tokens is None:
            ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k = self.remove_padding(
                input_ids, seq_lens_this_time
            )
        else:
            ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k = self.remove_padding(
                input_ids, seq_lens_this_time, draft_tokens, seq_lens_encoder
            )

        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_k"] = cu_seqlens_k
        kwargs["padding_offsets"] = padding_offset
        kwargs["max_input_length"] = self.max_seq_len

        inputs_embeds = self.embed_tokens(ids_remove_padding)

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids=input_ids,
                src=inputs_embeds,
                cum_offsets=cum_offsets,
                attn_mask=attention_mask,
                caches=caches,
                pre_caches=pre_caches,
                rotary_embs=rope_emb,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class LlamaForCausalLMAvxInferenceModel(GenerationAvxInferenceModel, LlamaPretrainedModel):

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.llama = LlamaAvxInferenceModel(config)
        self.lm_head = LLamaAvxLMHead(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: LlamaConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        return []

    def prepare_inputs_for_generation(
        self,
        input_ids,
        **kwargs,
    ):
        seq_len_encoder = kwargs.get("seq_len_encoder", None)
        seq_len_decoder = kwargs.get("seq_len_decoder", None)
        tgt_ids = kwargs.get("tgt_ids", None)
        cache = kwargs.get("cache", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        step_idx = kwargs.get("step_idx", None)
        if cache is None:
            # encoder
            past_seq_len = paddle.zeros_like(seq_len_decoder - seq_len_encoder, dtype="int64")
        else:
            # decoer
            past_seq_len = paddle.cast(seq_len_decoder, "int64")
            input_ids = tgt_ids
            inputs_embeds = None
        cur_seq_len = (paddle.to_tensor(input_ids.shape[1], dtype="int64"),)

        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "past_seq_len": past_seq_len,
            "cur_seq_len": cur_seq_len,
            "step_idx": step_idx,
        }
        return model_inputs

    def forward(
        self,
        input_ids,
        inputs_embeds=None,
        past_seq_len=None,
        cur_seq_len=None,
        step_idx=None,
        return_dict=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.llama(
            input_ids,
            inputs_embeds=inputs_embeds,
            past_seq_len=past_seq_len,
            cur_seq_len=paddle.to_tensor(input_ids.shape[1], dtype="int64"),
            step_idx=step_idx,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(
                paddle.to_tensor(state_dict["lm_head.weight"]).cast(paddle.get_default_dtype())
            )
        self.llama.set_state_dict({k: state_dict[k] for k in state_dict.keys()})

    @classmethod
    def set_inference_config(cls, config, predictor_args, **kwargs):
        super().set_inference_config(config, predictor_args, **kwargs)
        config.avx_type = predictor_args.avx_type
        config.avx_cachekv_type = predictor_args.avx_cachekv_type


class LlamaForCausalLMInferenceModel(GenerationInferenceModel, LlamaPretrainedModel):
    """
    Dynamic Batching for LLaMA Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.llama = LlamaInferenceModel(config)
        if config.tie_word_embeddings:
            self.lm_head = LlamaLMHead(config, embedding_weights=self.llama.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = LlamaLMHead(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: LlamaConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for llama model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """
        if max_length is None:
            max_length = config.max_position_embeddings

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kvs.append(
                [
                    2,
                    max_batch_size,
                    config.num_key_value_heads // max(config.tensor_parallel_degree, 1),
                    max_length,
                    config.hidden_size // config.num_attention_heads,
                ]
            )
        return cache_kvs

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
        pre_caches = kwargs.get("pre_caches", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if cache is not None:
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (tgt_generation_mask - 1) * 1e4
            # make inputs_embeds be none in decoder phase.
            # in forward function, it will be assigned according to input_ids.
            inputs_embeds = None
        else:
            attention_mask = (attention_mask - 1) * 1e4
        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
            "pre_caches": pre_caches,
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
        pre_caches=None,
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

        outputs = self.llama(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            cache_kvs=cache_kvs,
            pre_caches=pre_caches,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
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
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(
                paddle.to_tensor(state_dict["lm_head.weight"]).cast(self.lm_head.weight.dtype)
            )
        self.llama.set_state_dict({k: state_dict[k] for k in state_dict.keys()})

    @classmethod
    def confirm_inference_model(cls, predictor_args, **kwargs):
        if predictor_args.device == "xpu":
            raise ValueError(
                "you should run xpu dynamic model with --block_attn flag"
                "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/inference.md"
            )
        elif predictor_args.device == "cpu" and predictor_args.avx_model:
            import importlib

            import_class = importlib.import_module("paddlenlp.experimental.transformers.llama.modeling")
            model_class = getattr(import_class, "LlamaForCausalLMAvxInferenceModel")
            return model_class
        return cls


class LlamaForCausalLMBlockInferenceModel(GenerationBlockInferenceModel, LlamaPretrainedModel):
    """
    Dynamic Batching for LLaMA Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.max_candidate_len = config.get("speculate_max_candidate_len", 5)
        self.verify_window = config.get("speculate_verify_window", 2)
        self.max_seq_len = config.max_seq_len

        self.llama = LlamaBlockInferenceModel(config)
        if config.tie_word_embeddings:
            self.lm_head = LlamaLMHead(config, embedding_weights=self.llama.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = LlamaLMHead(config)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: LlamaConfig, is_split=True):

        logger.info("llama inference model _get_tensor_parallel_mappings")

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
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            if "a8w8" in config.quant_type:
                if config.quantization_config.shift_smooth_all_linears:
                    base_actions["layers.0.self_attn.o_proj.shift_bias"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.o_proj.smooth_weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.mlp.down_proj.shift_bias"] = partial(fn, is_column=True)
                    base_actions["layers.0.mlp.down_proj.smooth_weight"] = partial(fn, is_column=True)

                if config.quantization_config.shift:
                    if config.fuse_attention_qkv:
                        base_actions["layers.0.self_attn.qkv_proj.bias"] = partial(fn, is_column=True)
                    else:
                        base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
                        # if we have enough num_key_value_heads to split, then split it.
                        if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                            base_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
                            base_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)

                    if config.fuse_attention_ffn:
                        base_actions["layers.0.mlp.gate_up_fused_proj.bias"] = partial(
                            fn, is_column=True, is_naive_2fuse=True
                        )
                    else:
                        base_actions["layers.0.mlp.gate_proj.bias"] = partial(fn, is_column=True)
                        base_actions["layers.0.mlp.up_proj.bias"] = partial(fn, is_column=True)

            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
            else:
                base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                # if we have enough num_key_value_heads to split, then split it.
                if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                    base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            if config.fuse_attention_ffn:
                base_actions["layers.0.mlp.gate_up_fused_proj.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
                base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: LlamaConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for llama model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """
        max_block_per_seq = (config.max_seq_len + config.block_size - 1) // config.block_size
        if max_batch_size == -1:
            max_block_nums = None
        else:
            max_block_nums = max_batch_size * max_block_per_seq

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kv_shape = [
                max_block_nums,
                config.num_key_value_heads // max(config.tensor_parallel_degree, 1),
                config.block_size,
                config.hidden_size // config.num_attention_heads,
            ]
            cache_kvs.append(cache_kv_shape)
            cache_kvs.append(cache_kv_shape)
        return cache_kvs

    def prepare_inputs_for_generation(self, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        src_mask = kwargs.get("src_mask", None)
        block_tables = kwargs.get("block_tables", None)

        pre_caches = kwargs.get("pre_caches", None)
        caches = kwargs.get("caches", None)

        rope_emb = kwargs["rope_emb"]
        seq_lens_this_time = kwargs["seq_lens_this_time"]
        seq_lens_encoder = kwargs["seq_lens_encoder"]
        seq_lens_decoder = kwargs["seq_lens_decoder"]
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)

        # speculative decoding related parameters
        draft_tokens = kwargs.get("draft_tokens", None)
        output_padding_offset = kwargs.get("output_padding_offset", None)

        model_inputs = {
            "input_ids": input_ids,
            "src_mask": src_mask,
            "rope_emb": rope_emb,
            "pre_caches": pre_caches,
            "caches": caches,
            "seq_lens_this_time": seq_lens_this_time,
            "seq_lens_encoder": seq_lens_encoder,
            "seq_lens_decoder": seq_lens_decoder,
            "block_tables": block_tables,
            "k_quant_scales": k_quant_scales,
            "v_quant_scales": v_quant_scales,
            "k_dequant_scales": k_dequant_scales,
            "v_dequant_scales": v_dequant_scales,
            "draft_tokens": draft_tokens,
            "output_padding_offset": output_padding_offset,
        }
        return model_inputs

    def forward(
        self,
        input_ids,
        src_mask=None,
        pre_caches=None,
        caches=None,
        seq_lens_this_time=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        rope_emb=None,
        block_tables=None,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
        draft_tokens=None,
        output_padding_offset=None,
    ):
        outputs = self.llama(
            input_ids,
            src_mask=src_mask,
            caches=caches,
            rope_emb=rope_emb,
            block_tables=block_tables,
            pre_caches=pre_caches,
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            k_quant_scales=k_quant_scales,
            v_quant_scales=v_quant_scales,
            k_dequant_scales=k_dequant_scales,
            v_dequant_scales=v_dequant_scales,
            draft_tokens=draft_tokens,
            output_padding_offset=output_padding_offset,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        )

        return logits

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(
                paddle.to_tensor(state_dict["lm_head.weight"]).cast(self.lm_head.weight.dtype)
            )
        self.llama.set_state_dict({k: state_dict[k] for k in state_dict.keys()})


class LlamaForMiniGPT4InferenceModel(LlamaForCausalLMInferenceModel):
    """
    This class is 99% like LlamaForCausalLMInferenceModel.
    Used only for miniGPT4's second part.
    """

    # This function corresponds to miniGPT4's second part, only used in miniGPT4.
    @paddle.no_grad()
    def generate_text_with_image_features(
        self,
        image_features: paddle.Tensor,
        first_input_ids: paddle.Tensor,
        second_input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        inputs_embeds=None,
        **generate_kwargs
    ) -> paddle.Tensor:

        first_embeds = self.llama.embed_tokens(first_input_ids)
        second_embeds = self.llama.embed_tokens(second_input_ids)
        image_features = paddle.cast(image_features, dtype=first_embeds.dtype)
        inputs_embeds = paddle.concat([first_embeds, image_features, second_embeds], axis=1)

        outputs = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            penalty_score=penalty_score,
            frequency_score=frequency_score,
            presence_score=presence_score,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            step_idx=step_idx,
            stop_flags=stop_flags,
            tgt_ids=tgt_ids,
            tgt_pos=tgt_pos,
            tgt_generation_mask=tgt_generation_mask,
            pre_ids=pre_ids,
            stop_nums=stop_nums,
            cache_kvs=cache_kvs,
        )
        return outputs

    # rewrite to_static function in generation_utils.py
    def to_static(self, output_path: str, config: dict):
        dtype = config.get("dtype", paddle.get_default_dtype())
        cache_kvs_shapes = self.get_cache_kvs_shape(self.config, max_length=config.get("max_length", None))
        input_spec = [
            paddle.static.InputSpec(
                shape=[None, None, None], dtype="float32", name="image_features"
            ),  # image_features
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="first_input_ids"),  # first_input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="second_input_ids"),  # second_input_ids
            paddle.static.InputSpec(shape=[None, None], dtype=dtype, name="attention_mask"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),  # position_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_pos"),  # tgt_pos
            paddle.static.InputSpec(
                shape=[None, 1, 1, None], dtype=dtype, name="tgt_generation_mask"
            ),  # tgt_generation_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
            [
                paddle.static.InputSpec(
                    shape=shape,
                    dtype=dtype,
                    name="cache_kvs_{}".format(i),
                )
                for i, shape in enumerate(cache_kvs_shapes)
            ],  # cache_kvs
        ]

        model = paddle.jit.to_static(self.generate_text_with_image_features, input_spec=input_spec)
        paddle.jit.save(model, output_path)
