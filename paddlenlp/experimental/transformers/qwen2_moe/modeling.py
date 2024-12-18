# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.nn.quant import weight_quantize

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedBlockMultiTransformer,
    FusedBlockMultiTransformerWeightOnly,
    FusedMultiTransformerBase,
    FusedMultiTransformerConfig,
    FusedMultiTransformerWeightOnly,
    MoeConfig,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationBlockInferenceModel,
    GenerationInferenceModel,
)
from paddlenlp.experimental.transformers.utils import infererence_model_from_pretrained
from paddlenlp.transformers import Qwen2MoeConfig, Qwen2MoePretrainedModel
from paddlenlp.transformers.conversion_utils import split_param_func
from paddlenlp.transformers.model_outputs import (  # CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)
from paddlenlp.transformers.qwen2_moe.modeling import (
    Qwen2MoeLMHead,
    Qwen2MoePretrainingCriterion,
)
from paddlenlp.utils.log import logger

__all__ = ["Qwen2MoeForCausalLMInferenceModel", "Qwen2MoeForCausalLMBlockInferenceModel"]


class FusedQwen2MoeRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = paddle.create_parameter(
            shape=[config.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        return paddle.incubate.nn.functional.fused_rms_norm(x, self.weight, None, self.eps, begin_norm_axis=1)[0]


@register_base_model
class Qwen2MoeInferenceModel(Qwen2MoePretrainedModel):
    def __init__(self, config: Qwen2MoeConfig):
        super(Qwen2MoePretrainedModel, self).__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings
        self.quant_type = config.quant_type
        self.rope_theta = config.rope_theta

        self.use_neox = True

        self.use_weight_only = False
        if config.quant_type == "weight_only_int8":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int8"
        elif config.quant_type == "weight_only_int4":
            self.use_weight_only = True
            self.quant_algo = "weight_only_int4"

        if self.use_weight_only:
            assert (
                self.quant_type == "weight_only_int8" or self.quant_type == "weight_only_int4"
            ), "Expected quant_type equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_type
            )

        self.num_experts = config.num_experts
        self.moe_topk = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.moe_intermediate_size = config.moe_intermediate_size
        self.shared_expert_intermediate_size = config.shared_expert_intermediate_size

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

        ln_scale_attrs = [paddle.ParamAttr(name="fuseqwen2_moe.{}.ln_scale".format(i)) for i in range(self.num_layers)]
        qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen2_moe.{}.qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        qkv_bias_attrs = [paddle.ParamAttr(name="fuseqwen2_moe.{}.qkv_bias".format(i)) for i in range(self.num_layers)]
        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen2_moe.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fuseqwen2_moe.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen2_moe.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen2_moe.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        gate_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen2_moe.{}.gate_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        shared_expert_ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen2_moe.{}.shared_expert_ffn1_weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        shared_expert_ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen2_moe.{}.shared_expert_ffn2_weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]
        shared_expert_gate_weight_attrs = [
            paddle.ParamAttr(
                name="fuseqwen2_moe.{}.shared_expert_gate_weight".format(i),
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for i in range(self.num_layers)
        ]

        qkv_weight_scale_attrs = None
        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None
        shared_expert_ffn1_weight_scale_attrs = None
        shared_expert_ffn2_weight_scale_attrs = None

        if self.use_weight_only:
            qkv_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen2_moe.{}.qkv_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            out_proj_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen2_moe.{}.out_proj_weight_scale".format(i))
                for i in range(self.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen2_moe.{}.ffn1_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen2_moe.{}.ffn2_weight_scale".format(i)) for i in range(self.num_layers)
            ]
            shared_expert_ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen2_moe.{}.shared_expert_ffn1_weight_scale".format(i))
                for i in range(self.num_layers)
            ]
            shared_expert_ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name="fuseqwen2_moe.{}.shared_expert_ffn2_weight_scale".format(i))
                for i in range(self.num_layers)
            ]

        moe_config = MoeConfig(
            num_experts=self.num_experts,
            top_k=self.moe_topk,
            norm_topk_prob=self.norm_topk_prob,
            moe_every2=False,
            shared_expert_intermediate_size=self.shared_expert_intermediate_size,
            shared_expert_ffn1_weight_attrs=shared_expert_ffn1_weight_attrs,
            shared_expert_ffn1_weight_scale_attrs=shared_expert_ffn1_weight_scale_attrs,
            shared_expert_ffn2_weight_attrs=shared_expert_ffn2_weight_attrs,
            shared_expert_ffn2_weight_scale_attrs=shared_expert_ffn2_weight_scale_attrs,
            shared_expert_gate_weight_attrs=shared_expert_gate_weight_attrs,
        )

        transformer_config = FusedMultiTransformerConfig(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            dim_feedforward=self.moe_intermediate_size,
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
            gate_weight_attrs=gate_weight_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            qkv_bias_attrs=qkv_bias_attrs,
            epsilon=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            norm_type="rmsnorm",
            use_neox_rotary_style=self.use_neox,
            rank_id=config.tensor_parallel_rank,
            moe_config=moe_config,
            append_attn=config.append_attn,
        )

        self.set_transformer_block(transformer_config)

        self.norm = FusedQwen2MoeRMSNorm(config)

        self.cache_kvs = None
        self.head_dim_shape_tensor = paddle.ones((self.hidden_size // self.num_attention_heads), dtype="int8")

    def set_transformer_block(self, transformer_config):
        if self.use_weight_only:
            self.transformer_block = FusedMultiTransformerWeightOnly(transformer_config)
        else:
            self.transformer_block = FusedMultiTransformerBase(transformer_config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.transformer_block.init_weight()
        head_size = self.hidden_size // self.num_attention_heads
        split_fn = split_param_func()
        dtype = paddle.get_default_dtype()
        embed_tokens_weight = paddle.to_tensor(state_dict["qwen2_moe.embed_tokens.weight"]).cast(
            self.embed_tokens.weight.dtype
        )
        norm_weight = paddle.to_tensor(state_dict["qwen2_moe.norm.weight"]).cast(self.norm.weight.dtype)
        self.embed_tokens.weight.set_value(embed_tokens_weight)
        self.norm.weight.set_value(norm_weight)

        if self.use_weight_only:
            logger.info("weight only is enabled")
        for idx in range(self.num_layers):
            logger.info(f"set state for layer {idx}")
            unfused_state_dict = {}
            ln_scale = paddle.to_tensor(state_dict["qwen2_moe.layers.{}.input_layernorm.weight".format(idx)]).cast(
                self.transformer_block.ln_scales[idx].dtype
            )
            self.transformer_block.ln_scales[idx].set_value(ln_scale)
            if "qwen2_moe.layers.{}.self_attn.qkv_proj.weight".format(idx) in state_dict.keys():
                concated_qkv_weight = np.concatenate(
                    split_fn(
                        state_dict["qwen2_moe.layers.{}.self_attn.qkv_proj.weight".format(idx)],
                        is_qkv=True,
                        num_heads=self.num_attention_heads // self.config.tensor_parallel_degree,
                        num_key_value_heads=self.num_key_value_heads // self.config.tensor_parallel_degree,
                    ),
                    axis=-1,
                ).transpose(1, 0)
            else:
                unfused_state_dict = {}
                unfused_state_dict["qwen2_moe.self_attn.q_proj.weight"] = state_dict[
                    "qwen2_moe.layers.{}.self_attn.q_proj.weight".format(idx)
                ]
                unfused_state_dict["qwen2_moe.self_attn.k_proj.weight"] = state_dict[
                    "qwen2_moe.layers.{}.self_attn.k_proj.weight".format(idx)
                ]
                unfused_state_dict["qwen2_moe.self_attn.v_proj.weight"] = state_dict[
                    "qwen2_moe.layers.{}.self_attn.v_proj.weight".format(idx)
                ]

                concated_qkv_weight = (
                    np.concatenate(
                        [
                            unfused_state_dict["qwen2_moe.self_attn.q_proj.weight"],
                            unfused_state_dict["qwen2_moe.self_attn.k_proj.weight"],
                            unfused_state_dict["qwen2_moe.self_attn.v_proj.weight"],
                        ],
                        axis=-1,
                    )
                    .transpose(1, 0)
                    .reshape(
                        (
                            self.num_attention_heads // self.config.tensor_parallel_degree
                            + 2 * self.num_key_value_heads // self.config.tensor_parallel_degree
                        )
                        * (head_size),
                        self.hidden_size,
                    )
                )

            qkv_weight = paddle.to_tensor(concated_qkv_weight).cast(dtype)

            if self.use_weight_only:
                qkv_weight = paddle.transpose(qkv_weight, perm=[1, 0])
                qkv_quanted_weight, qkv_weight_scale = weight_quantize(qkv_weight, algo=self.quant_algo)
                self.transformer_block.qkv_weights[idx].set_value(qkv_quanted_weight)
                self.transformer_block.qkv_weights_scale[idx].set_value(qkv_weight_scale)
            else:
                self.transformer_block.qkv_weights[idx].set_value(qkv_weight)

            unfused_state_dict["qwen2_moe.self_attn.q_proj.bias"] = state_dict[
                "qwen2_moe.layers.{}.self_attn.q_proj.bias".format(idx)
            ]
            unfused_state_dict["qwen2_moe.self_attn.k_proj.bias"] = state_dict[
                "qwen2_moe.layers.{}.self_attn.k_proj.bias".format(idx)
            ]
            unfused_state_dict["qwen2_moe.self_attn.v_proj.bias"] = state_dict[
                "qwen2_moe.layers.{}.self_attn.v_proj.bias".format(idx)
            ]

            concated_qkv_biases = np.concatenate(
                [
                    unfused_state_dict["qwen2_moe.self_attn.q_proj.bias"],
                    unfused_state_dict["qwen2_moe.self_attn.k_proj.bias"],
                    unfused_state_dict["qwen2_moe.self_attn.v_proj.bias"],
                ],
                axis=-1,
            )
            qkv_bias = paddle.to_tensor(concated_qkv_biases).cast(self.transformer_block.qkv_biases[idx].dtype)
            self.transformer_block.qkv_biases[idx].set_value(qkv_bias)

            linear_weight = paddle.to_tensor(
                state_dict["qwen2_moe.layers.{}.self_attn.o_proj.weight".format(idx)]
            ).cast(dtype)

            if self.use_weight_only:
                linear_quanted_weight, linear_weight_scale = weight_quantize(linear_weight, algo=self.quant_algo)
                self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight)
                self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale)
            else:
                self.transformer_block.linear_weights[idx].set_value(linear_weight)

            ffn_ln_scale = paddle.to_tensor(
                state_dict["qwen2_moe.layers.{}.post_attention_layernorm.weight".format(idx)],
            ).cast(
                self.transformer_block.ffn_ln_scales[idx].dtype,
            )
            self.transformer_block.ffn_ln_scales[idx].set_value(ffn_ln_scale)

            ffn1_weights = []
            ffn2_weights = []
            ffn1_scales = []
            ffn2_scales = []
            for expert_idx in range(self.num_experts):
                up_weight = paddle.to_tensor(
                    state_dict["qwen2_moe.layers.{0}.mlp.experts.{1}.up_proj.weight".format(idx, expert_idx)]
                ).cast(dtype)
                gate_weight = paddle.to_tensor(
                    state_dict["qwen2_moe.layers.{0}.mlp.experts.{1}.gate_proj.weight".format(idx, expert_idx)]
                ).cast(dtype)
                down_weight = paddle.to_tensor(
                    state_dict["qwen2_moe.layers.{0}.mlp.experts.{1}.down_proj.weight".format(idx, expert_idx)]
                ).cast(dtype)

                if self.use_weight_only:
                    ffn1_weight = paddle.concat(x=[gate_weight, up_weight], axis=-1)
                    ffn1_quanted_weight, ffn1_weight_scale = weight_quantize(ffn1_weight, algo=self.quant_algo)
                    ffn2_quanted_weight, ffn2_weight_scale = weight_quantize(down_weight, algo=self.quant_algo)
                    ffn1_weights.append(ffn1_quanted_weight.reshape([self.transformer_block.config.embed_dim, -1]))
                    ffn2_weights.append(ffn2_quanted_weight.reshape([-1, self.transformer_block.config.embed_dim]))
                    ffn1_scales.append(ffn1_weight_scale)
                    ffn2_scales.append(ffn2_weight_scale)
                else:
                    ffn1_weights.append(paddle.concat(x=[gate_weight, up_weight], axis=-1))
                    ffn2_weights.append(down_weight)

            fused_moe_ffn1_weight = paddle.to_tensor(ffn1_weights)
            fused_moe_ffn2_weight = paddle.to_tensor(ffn2_weights)
            fused_moe_ffn1_weight_scale = paddle.to_tensor(ffn1_scales)
            fused_moe_ffn2_weight_scale = paddle.to_tensor(ffn2_scales)
            gate_weight = paddle.to_tensor(state_dict["qwen2_moe.layers.{}.mlp.gate.weight".format(idx)]).cast(
                "float32"
            )

            self.transformer_block.ffn1_weights[idx].set_value(fused_moe_ffn1_weight)
            self.transformer_block.ffn2_weights[idx].set_value(fused_moe_ffn2_weight)
            self.transformer_block.gate_weights[idx].set_value(gate_weight)

            if self.use_weight_only:
                self.transformer_block.ffn1_weights_scale[idx].set_value(fused_moe_ffn1_weight_scale)
                self.transformer_block.ffn2_weights_scale[idx].set_value(fused_moe_ffn2_weight_scale)

            shared_expert_ffn1gate_weight = paddle.to_tensor(
                state_dict["qwen2_moe.layers.{}.mlp.shared_expert.gate_proj.weight".format(idx)]
            ).cast(dtype)
            shared_expert_ffn1up_weight = paddle.to_tensor(
                state_dict["qwen2_moe.layers.{}.mlp.shared_expert.up_proj.weight".format(idx)]
            ).cast(dtype)
            shared_expert_ffn1_weight = paddle.concat(
                x=[shared_expert_ffn1gate_weight, shared_expert_ffn1up_weight], axis=-1
            )

            shared_expert_ffn2_weight = paddle.to_tensor(
                state_dict["qwen2_moe.layers.{}.mlp.shared_expert.down_proj.weight".format(idx)]
            ).cast(dtype)
            shared_expert_gate_weight = paddle.to_tensor(
                state_dict["qwen2_moe.layers.{}.mlp.shared_expert_gate.weight".format(idx)]
            ).cast(dtype)

            if self.use_weight_only:
                shared_expert_ffn1_quanted_weight, shared_expert_ffn1_weight_scale = weight_quantize(
                    shared_expert_ffn1_weight, algo=self.quant_algo
                )
                self.transformer_block.shared_expert_ffn1_weights[idx].set_value(shared_expert_ffn1_quanted_weight)
                self.transformer_block.shared_expert_ffn1_weights_scale[idx].set_value(shared_expert_ffn1_weight_scale)

                shared_expert_ffn2_quanted_weight, shared_expert_ffn2_weight_scale = weight_quantize(
                    shared_expert_ffn2_weight, algo=self.quant_algo
                )
                self.transformer_block.shared_expert_ffn2_weights[idx].set_value(shared_expert_ffn2_quanted_weight)
                self.transformer_block.shared_expert_ffn2_weights_scale[idx].set_value(shared_expert_ffn2_weight_scale)
                # shared_expert_gate_weight is not quanted
                self.transformer_block.shared_expert_gate_weights[idx].set_value(shared_expert_gate_weight)

            else:
                self.transformer_block.shared_expert_ffn1_weights[idx].set_value(shared_expert_ffn1_weight)
                self.transformer_block.shared_expert_ffn2_weights[idx].set_value(shared_expert_ffn2_weight)
                self.transformer_block.shared_expert_gate_weights[idx].set_value(shared_expert_gate_weight)

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        from paddlenlp_ops import get_padding_offset

        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    # This function is a little different from prepare_input_ids_for_generation in paddlenlp/transformers/generation/utils.py,
    # it is used to generate fake input_ids according to inputs_embeds length.
    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        seq_len = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
            seq_len = encoder_output.shape[1]
        return paddle.full([batch_size, seq_len], bos_token_id, dtype="int64")

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

        # generate a fake input_ids according to inputs_embeds
        # this is usually occurred in img2txt multimodal model when first enter into this forward function.
        if input_ids is None and inputs_embeds is not None:
            input_ids = self.prepare_input_ids_for_generation(self.config.bos_token_id, inputs_embeds)
        if inputs_embeds is not None:
            batch, seq_len, hidden_dim = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape([batch * seq_len, hidden_dim])

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
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
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

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
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Qwen2MoeForCausalLMInferenceModel(GenerationInferenceModel, Qwen2MoePretrainedModel):
    def __init__(self, config: Qwen2MoeConfig, **kwargs):
        super(Qwen2MoeForCausalLMInferenceModel, self).__init__(config)
        self.qwen2_moe = Qwen2MoeInferenceModel(config)
        if config.tie_word_embeddings:
            self.lm_head = Qwen2MoeLMHead(
                config, embedding_weights=self.qwen2_moe.embed_tokens.weight, transpose_y=True
            )
            self.tie_weights()
        else:
            self.lm_head = Qwen2MoeLMHead(config)
        self.criterion = Qwen2MoePretrainingCriterion(config)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return infererence_model_from_pretrained(cls, pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def get_cache_kvs_shape(
        cls, config: Qwen2MoeConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for qwen model

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
        use_cache=False,
        cache=None,
        cache_kvs=None,
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.qwen2_moe(
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

        # if labels is None，means we need full output, instead of tensor_parallel_output
        # tensor_parallel_output is together with ParallelCrossEntropy
        tensor_parallel_output = (
            self.config.tensor_parallel_output and labels is not None and self.config.tensor_parallel_degree > 1
        )
        lm_logits = self.lm_head(hidden_states, tensor_parallel_output=tensor_parallel_output)

        loss = None
        if labels is not None:
            loss = self.criterion(lm_logits, labels)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            lm_head_weight = paddle.to_tensor(state_dict["lm_head.weight"]).cast(self.lm_head.weight.dtype)
            self.lm_head.weight.set_value(lm_head_weight)
        self.qwen2_moe.set_state_dict({k: state_dict[k] for k in state_dict.keys()})


@register_base_model
class Qwen2MoeBlockInferenceModel(Qwen2MoeInferenceModel):
    def __init__(self, config: Qwen2MoeConfig):
        self.append_attn = config.append_attn
        super().__init__(config)
        self.max_seq_len = config.max_seq_len
        self.block_size = config.block_size

    def set_transformer_block(self, transformer_config):
        if self.use_weight_only:
            self.transformer_block = FusedBlockMultiTransformerWeightOnly(transformer_config)
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


class Qwen2MoeForCausalLMBlockInferenceModel(GenerationBlockInferenceModel, Qwen2MoePretrainedModel):
    """
    Dynamic Batching for Qwen2Moe Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.max_candidate_len = config.get("speculate_max_candidate_len", 5)
        self.verify_window = config.get("speculate_verify_window", 2)
        self.max_seq_len = config.max_seq_len

        self.qwen2_moe = Qwen2MoeBlockInferenceModel(config)
        if config.tie_word_embeddings:
            self.lm_head = Qwen2MoeLMHead(
                config, embedding_weights=self.qwen2_moe.embed_tokens.weight, transpose_y=True
            )
            self.tie_weights()
        else:
            self.lm_head = Qwen2MoeLMHead(config)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: Qwen2MoeConfig, is_split=True):

        logger.info("Qwen2Moe inference model _get_tensor_parallel_mappings")

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
            }

            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
            else:
                base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
                # if we have enough num_key_value_heads to split, then split it.
                if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                    base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)

            if config.fuse_attention_ffn:
                base_actions["layers.0.mlp.gate_up_fused_proj.weight"] = partial(
                    fn, is_column=True, is_naive_2fuse=True
                )
            else:
                for expert_idx in range(config.num_experts):
                    base_actions[f"layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = partial(fn, is_column=True)
                    base_actions[f"layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = partial(fn, is_column=True)
                    base_actions[f"layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = partial(fn, is_column=False)
            base_actions["layers.0.mlp.shared_expert.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_expert.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_expert.down_proj.weight"] = partial(fn, is_column=False)

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
        cls, config: Qwen2MoeConfig, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for Qwen2Moe model

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
        outputs = self.qwen2_moe(
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
        self.qwen2_moe.set_state_dict({k: state_dict[k] for k in state_dict.keys()})
