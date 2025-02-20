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
from typing import Tuple

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.nn.quant import weight_quantize

from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedBlockMultiTransformer,
    FusedBlockMultiTransformerWeightOnly,
    FusedMultiTransformerConfig,
    MLAConfig,
    MoeConfig,
    SpeculateConfig,
)
from paddlenlp.experimental.transformers.generation_utils import (
    GenerationBlockInferenceModel,
)
from paddlenlp.experimental.transformers.utils import infererence_model_from_pretrained
from paddlenlp.transformers import DeepseekV2Config, DeepseekV2PretrainedModel
from paddlenlp.transformers.deepseek_v2.modeling import (
    DeepseekV2LMHead,
    yarn_find_correction_range,
    yarn_get_mscale,
    yarn_linear_ramp_mask,
)
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)
from paddlenlp.utils.log import logger

__all__ = ["DeepseekV2ForCausalLMBlockInferenceModel"]


class DeepseekScalingRotaryEmbedding(nn.Layer):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        scaling_factor: float,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        super().__init__()
        self._dtype = paddle.get_default_dtype()

        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )

        cos_cache, sin_cache = self._compute_cos_sin_cache()

        self.cos_cache: paddle.Tensor
        self.register_buffer("cos_cache", cos_cache, persistable=True)
        self.sin_cache: paddle.Tensor
        self.register_buffer("sin_cache", sin_cache, persistable=True)

    def _compute_inv_freq(self, scaling_factor: float) -> paddle.Tensor:
        pos_freqs = self.base ** (paddle.arange(0, self.rotary_dim, 2, dtype=paddle.float32) / self.rotary_dim)

        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.rotary_dim, self.base, self.max_position_embeddings
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> paddle.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = paddle.arange(self.max_position_embeddings * self.scaling_factor, dtype=paddle.float32)

        freqs = paddle.outer(t, inv_freq)
        emb = paddle.concat((freqs, freqs), axis=-1)
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale

        return cos.cast(self._dtype), sin.cast(self._dtype)

    def forward(
        self,
        position_ids: paddle.Tensor,
        query: paddle.Tensor,
        key: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        cos = self.cos_cache[position_ids].unsqueeze(1)
        sin = self.sin_cache[position_ids].unsqueeze(1)

        def rotate_half(x):
            """Rotates half the hidden axiss of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x

        s, h, d = query.shape
        query = query.reshape([s, h, d // 2, 2]).transpose([0, 1, 3, 2]).reshape([s, h, d])

        s, h, d = key.shape
        key = key.reshape([s, h, d // 2, 2]).transpose([0, 1, 3, 2]).reshape([s, h, d])

        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        return query, key


class DeepseekV2RMSNorm(nn.Layer):
    def __init__(self, config: DeepseekV2Config):
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
class DeepseekV2BlockInferenceModel(DeepseekV2PretrainedModel):
    def __init__(self, config: DeepseekV2Config, base_model_prefix: str):
        super(DeepseekV2PretrainedModel, self).__init__(config)
        self.base_model_prefix = base_model_prefix

        self.config = config

        self.max_seq_len = config.max_seq_len

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.quant_type = config.quant_type
        self.rope_theta = config.rope_theta
        self.return_full_hidden_states = config.get("return_full_hidden_states", False)

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
            ), f"Expected quant_type equal to 'weight_only_int8' or 'weight_only_int4', but received {self.quant_type}"

        self.first_k_dense_replace = config.first_k_dense_replace
        self.n_routed_experts = config.n_routed_experts

        if config.tensor_parallel_degree > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {config.tensor_parallel_degree} is greater than "
                f"the number of experts {config.n_routed_experts}."
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

        self.norm = DeepseekV2RMSNorm(config)

        scaling_factor = config.rope_scaling.get("factor", 1)
        original_max_position = config.rope_scaling.get("original_max_position_embeddings", 4096)
        extra_kwargs = {
            k: v
            for k, v in config.rope_scaling.items()
            if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim")
        }
        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            config.qk_rope_head_dim,
            original_max_position,
            config.rope_theta,
            scaling_factor,
            **extra_kwargs,
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
            paddle.ParamAttr(name=f"fuse{self.base_model_prefix}.{idx}.ln_scale") for idx in range(self.num_layers)
        ]

        q_a_proj_weight_attrs = None
        q_a_layernorm_weight_attrs = None
        q_b_proj_weight_attrs = None
        q_proj_weight_attrs = None

        if self.config.q_lora_rank is not None:
            q_a_proj_weight_attrs = [
                paddle.ParamAttr(
                    name=f"fuse{self.base_model_prefix}.{idx}.q_a_proj_weight",
                    initializer=paddle.nn.initializer.Constant(value=0),
                )
                for idx in range(self.num_layers)
            ]
            q_a_layernorm_weight_attrs = [
                paddle.ParamAttr(
                    name=f"fuse{self.base_model_prefix}.{idx}.q_a_layernorm_weight",
                    initializer=paddle.nn.initializer.Constant(value=1.0),
                )
                for idx in range(self.num_layers)
            ]
            q_b_proj_weight_attrs = [
                paddle.ParamAttr(
                    name=f"fuse{self.base_model_prefix}.{idx}.q_b_proj_weight",
                    initializer=paddle.nn.initializer.Constant(value=0),
                )
                for idx in range(self.num_layers)
            ]
        else:
            q_proj_weight_attrs = [
                paddle.ParamAttr(
                    name=f"fuse{self.base_model_prefix}.{idx}.q_proj_weight",
                    initializer=paddle.nn.initializer.Constant(value=0),
                )
                for idx in range(self.num_layers)
            ]

        kv_a_proj_with_mqa_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.kv_a_proj_with_mqa_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for idx in range(self.num_layers)
        ]
        kv_a_layernorm_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.kv_a_layernorm_weight",
                initializer=paddle.nn.initializer.Constant(value=1.0),
            )
            for idx in range(self.num_layers)
        ]
        kv_b_proj_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.kv_b_proj_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for idx in range(self.num_layers)
        ]

        out_proj_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.out_proj_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for idx in range(self.num_layers)
        ]
        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name=f"fuse{self.base_model_prefix}.{idx}.ffn_ln_scale") for idx in range(self.num_layers)
        ]
        ffn1_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.ffn1_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for idx in range(self.num_layers)
        ]
        ffn2_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.ffn2_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for idx in range(self.num_layers)
        ]
        gate_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.gate_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for idx in range(self.num_layers)
        ]

        e_score_correction_bias_attrs = None
        if self.base_model_prefix.startswith("deepseek_v3"):
            e_score_correction_bias_attrs = [
                paddle.ParamAttr(
                    name=f"fuse{self.base_model_prefix}.{idx}.e_score_correction_bias",
                    initializer=paddle.nn.initializer.Constant(value=0),
                )
                if idx >= self.config.first_k_dense_replace
                else None
                for idx in range(self.num_layers)
            ]

        shared_expert_ffn1_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.shared_expert_ffn1_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for idx in range(self.num_layers)
        ]
        shared_expert_ffn2_weight_attrs = [
            paddle.ParamAttr(
                name=f"fuse{self.base_model_prefix}.{idx}.shared_expert_ffn2_weight",
                initializer=paddle.nn.initializer.Constant(value=0),
            )
            for idx in range(self.num_layers)
        ]

        q_proj_weight_scale_attrs = None
        q_a_proj_weight_scale_attrs = None
        q_b_proj_weight_scale_attrs = None
        kv_a_proj_with_mqa_weight_scale_attrs = None
        kv_b_proj_weight_scale_attrs = None

        out_proj_weight_scale_attrs = None
        ffn1_weight_scale_attrs = None
        ffn2_weight_scale_attrs = None
        shared_expert_ffn1_weight_scale_attrs = None
        shared_expert_ffn2_weight_scale_attrs = None

        if self.use_weight_only:
            if self.config.q_lora_rank is not None:
                q_proj_weight_scale_attrs = [
                    paddle.ParamAttr(
                        name=f"fuse{self.base_model_prefix}.{idx}.q_a_proj_weight_scale",
                    )
                    for idx in range(self.num_layers)
                ]
                q_b_proj_weight_scale_attrs = [
                    paddle.ParamAttr(
                        name=f"fuse{self.base_model_prefix}.{idx}.q_b_proj_weight_scale",
                    )
                    for idx in range(self.num_layers)
                ]
            else:
                q_proj_weight_scale_attrs = [
                    paddle.ParamAttr(
                        name=f"fuse{self.base_model_prefix}.{idx}.q_proj_weight_scale",
                    )
                    for idx in range(self.num_layers)
                ]

            kv_a_proj_with_mqa_weight_scale_attrs = [
                paddle.ParamAttr(
                    name=f"fuse{self.base_model_prefix}.{idx}.kv_a_proj_with_mqa_weight_scale",
                )
                for idx in range(self.num_layers)
            ]
            kv_b_proj_weight_scale_attrs = [
                paddle.ParamAttr(
                    name=f"fuse{self.base_model_prefix}.{idx}.kv_b_proj_weight_scale",
                )
                for idx in range(self.num_layers)
            ]

            out_proj_weight_scale_attrs = [
                paddle.ParamAttr(name=f"fuse{self.base_model_prefix}.{idx}.out_proj_weight_scale")
                for idx in range(self.num_layers)
            ]
            ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name=f"fuse{self.base_model_prefix}.{idx}.ffn1_weight_scale")
                for idx in range(self.num_layers)
            ]
            ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name=f"fuse{self.base_model_prefix}.{idx}.ffn2_weight_scale")
                for idx in range(self.num_layers)
            ]
            shared_expert_ffn1_weight_scale_attrs = [
                paddle.ParamAttr(name=f"fuse{self.base_model_prefix}.{idx}.shared_expert_ffn1_weight_scale")
                for idx in range(self.num_layers)
            ]
            shared_expert_ffn2_weight_scale_attrs = [
                paddle.ParamAttr(name=f"fuse{self.base_model_prefix}.{idx}.shared_expert_ffn2_weight_scale")
                for idx in range(self.num_layers)
            ]

        mla_config = MLAConfig(
            q_lora_rank=self.config.q_lora_rank,
            kv_lora_rank=self.config.kv_lora_rank,
            qk_nope_head_dim=self.config.qk_nope_head_dim,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            v_head_dim=self.config.v_head_dim,
            mscale=yarn_get_mscale(scaling_factor, float(config.rope_scaling.get("mscale_all_dim", 1.0))),
            q_proj_weight_attrs=q_proj_weight_attrs,
            q_proj_weight_scale_attrs=q_proj_weight_scale_attrs,
            q_a_proj_weight_attrs=q_a_proj_weight_attrs,
            q_a_proj_weight_scale_attrs=q_a_proj_weight_scale_attrs,
            q_a_layernorm_weight_attrs=q_a_layernorm_weight_attrs,
            q_b_proj_weight_attrs=q_b_proj_weight_attrs,
            q_b_proj_weight_scale_attrs=q_b_proj_weight_scale_attrs,
            kv_a_proj_with_mqa_weight_attrs=kv_a_proj_with_mqa_weight_attrs,
            kv_a_proj_with_mqa_weight_scale_attrs=kv_a_proj_with_mqa_weight_scale_attrs,
            kv_a_layernorm_weight_attrs=kv_a_layernorm_weight_attrs,
            kv_b_proj_weight_attrs=kv_b_proj_weight_attrs,
            kv_b_proj_weight_scale_attrs=kv_b_proj_weight_scale_attrs,
        )

        moe_config = MoeConfig(
            num_experts=self.n_routed_experts,
            top_k=self.config.num_experts_per_tok,
            topk_group=self.config.topk_group,
            norm_topk_prob=self.config.norm_topk_prob,
            routed_scaling_factor=self.config.routed_scaling_factor,
            num_expert_group=self.config.n_group,
            topk_method=self.config.topk_method,
            moe_intermediate_size=self.config.moe_intermediate_size,
            first_k_dense_replace=self.first_k_dense_replace,
            shared_expert_with_gate=False,
            shared_expert_intermediate_size=self.config.moe_intermediate_size * self.config.n_shared_experts,
            shared_expert_ffn1_weight_attrs=shared_expert_ffn1_weight_attrs,
            shared_expert_ffn1_weight_scale_attrs=shared_expert_ffn1_weight_scale_attrs,
            shared_expert_ffn2_weight_attrs=shared_expert_ffn2_weight_attrs,
            shared_expert_ffn2_weight_scale_attrs=shared_expert_ffn2_weight_scale_attrs,
        )

        speculate_config = SpeculateConfig(
            speculate_method=config.get("speculate_method", None),
            speculate_max_draft_token_num=config.get("speculate_max_draft_token_num", 5),
            return_full_hidden_states=config.get("return_full_hidden_states", False),
        )

        transformer_config = FusedMultiTransformerConfig(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            quant_type=self.quant_type,
            activation="swiglu",
            num_layers=config.num_hidden_layers,
            nranks=config.tensor_parallel_degree,
            ring_id=ring_id,
            ln_scale_attrs=ln_scale_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            linear_weight_scale_attrs=out_proj_weight_scale_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            gate_weight_attrs=gate_weight_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn1_weight_scale_attrs=ffn1_weight_scale_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            ffn2_weight_scale_attrs=ffn2_weight_scale_attrs,
            e_score_correction_bias_attrs=e_score_correction_bias_attrs,
            epsilon=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            rotary_emb=self.rotary_emb,
            norm_type="rmsnorm",
            rank_id=config.tensor_parallel_rank,
            moe_config=moe_config,
            mla_config=mla_config,
            append_attn=config.append_attn,
            speculate_config=speculate_config,
        )

        self.set_transformer_block(transformer_config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.transformer_block.init_weight()

        dtype = paddle.get_default_dtype()
        embed_tokens_weight = paddle.to_tensor(state_dict[f"{self.base_model_prefix}.embed_tokens.weight"]).cast(
            self.embed_tokens.weight.dtype
        )
        norm_weight = paddle.to_tensor(state_dict[f"{self.base_model_prefix}.norm.weight"]).cast(
            self.norm.weight.dtype
        )
        self.embed_tokens.weight.set_value(embed_tokens_weight)
        self.norm.weight.set_value(norm_weight)

        if self.use_weight_only:
            logger.info("weight only is enabled")
        for idx in range(self.num_layers):
            logger.info(f"set state for layer {idx}")

            ln_scale = paddle.to_tensor(
                state_dict[f"{self.base_model_prefix}.layers.{idx}.input_layernorm.weight"]
            ).cast(self.transformer_block.ln_scales[idx].dtype)
            self.transformer_block.ln_scales[idx].set_value(ln_scale)

            if self.config.q_lora_rank is not None:
                q_a_proj_weight = paddle.to_tensor(
                    state_dict[f"{self.base_model_prefix}.layers.{idx}.self_attn.q_a_proj.weight"]
                ).cast(dtype)
                q_a_layernorm_weight = paddle.to_tensor(
                    state_dict[f"{self.base_model_prefix}.layers.{idx}.self_attn.q_a_layernorm.weight"]
                ).cast(self.transformer_block.q_a_layernorm_weights[idx].dtype)
                q_b_proj_weight = paddle.to_tensor(
                    state_dict[f"{self.base_model_prefix}.layers.{idx}.self_attn.q_b_proj.weight"]
                ).cast(dtype)

                if self.use_weight_only:
                    q_a_proj_quanted_weight, q_a_proj_weight_scale = weight_quantize(
                        q_a_proj_weight, algo=self.quant_algo
                    )
                    self.transformer_block.q_a_proj_weights[idx].set_value(q_a_proj_quanted_weight)
                    self.transformer_block.q_a_proj_weights_scale[idx].set_value(q_a_proj_weight_scale)

                    q_b_proj_quanted_weight, q_b_proj_weight_scale = weight_quantize(
                        q_b_proj_weight, algo=self.quant_algo
                    )
                    self.transformer_block.q_b_proj_weights[idx].set_value(q_b_proj_quanted_weight)
                    self.transformer_block.q_a_layernorm_weights[idx].set_value(q_a_layernorm_weight)
                    self.transformer_block.q_b_proj_weights_scale[idx].set_value(q_b_proj_weight_scale)
                else:
                    self.transformer_block.q_a_proj_weights[idx].set_value(q_a_proj_weight)
                    self.transformer_block.q_a_layernorm_weights[idx].set_value(q_a_layernorm_weight)
                    self.transformer_block.q_b_proj_weights[idx].set_value(q_b_proj_weight)
            else:
                q_proj_weight = paddle.to_tensor(
                    state_dict[f"{self.base_model_prefix}.layers.{idx}.self_attn.q_proj.weight"]
                ).cast(dtype)

                if self.use_weight_only:
                    q_proj_quanted_weight, q_proj_weight_scale = weight_quantize(q_proj_weight, algo=self.quant_algo)
                    self.transformer_block.q_proj_weights[idx].set_value(q_proj_quanted_weight)
                    self.transformer_block.q_proj_weights_scale[idx].set_value(q_proj_weight_scale)
                else:
                    self.transformer_block.q_proj_weights[idx].set_value(q_proj_weight)

            kv_a_proj_with_mqa_weight = paddle.to_tensor(
                state_dict[f"{self.base_model_prefix}.layers.{idx}.self_attn.kv_a_proj_with_mqa.weight"]
            ).cast(dtype)
            kv_a_layernorm_weight = paddle.to_tensor(
                state_dict[f"{self.base_model_prefix}.layers.{idx}.self_attn.kv_a_layernorm.weight"]
            ).cast(self.transformer_block.kv_a_layernorm_weights[idx].dtype)
            kv_b_proj_weight = paddle.to_tensor(
                state_dict[f"{self.base_model_prefix}.layers.{idx}.self_attn.kv_b_proj.weight"]
            ).cast(dtype)

            if self.use_weight_only:
                kv_a_proj_with_mqa_quanted_weight, kv_a_proj_with_mqa_weight_scale = weight_quantize(
                    kv_a_proj_with_mqa_weight, algo=self.quant_algo
                )
                self.transformer_block.kv_a_proj_with_mqa_weights[idx].set_value(kv_a_proj_with_mqa_quanted_weight)
                self.transformer_block.kv_a_proj_with_mqa_weights_scale[idx].set_value(kv_a_proj_with_mqa_weight_scale)

                kv_b_proj_quanted_weight, kv_b_proj_weight_scale = weight_quantize(
                    kv_b_proj_weight, algo=self.quant_algo
                )
                self.transformer_block.kv_b_proj_weights[idx].set_value(kv_b_proj_quanted_weight)
                self.transformer_block.kv_a_layernorm_weights[idx].set_value(kv_a_layernorm_weight)
                self.transformer_block.kv_b_proj_weights_scale[idx].set_value(kv_b_proj_weight_scale)
            else:
                self.transformer_block.kv_a_proj_with_mqa_weights[idx].set_value(kv_a_proj_with_mqa_weight)
                self.transformer_block.kv_a_layernorm_weights[idx].set_value(kv_a_layernorm_weight)
                self.transformer_block.kv_b_proj_weights[idx].set_value(kv_b_proj_weight)

            linear_weight = paddle.to_tensor(
                state_dict[f"{self.base_model_prefix}.layers.{idx}.self_attn.o_proj.weight"]
            ).cast(dtype)

            if self.use_weight_only:
                linear_quanted_weight, linear_weight_scale = weight_quantize(linear_weight, algo=self.quant_algo)
                self.transformer_block.linear_weights[idx].set_value(linear_quanted_weight)
                self.transformer_block.linear_weights_scale[idx].set_value(linear_weight_scale)
            else:
                self.transformer_block.linear_weights[idx].set_value(linear_weight)

            ffn_ln_scale = paddle.to_tensor(
                state_dict[f"{self.base_model_prefix}.layers.{idx}.post_attention_layernorm.weight"],
            ).cast(
                self.transformer_block.ffn_ln_scales[idx].dtype,
            )
            self.transformer_block.ffn_ln_scales[idx].set_value(ffn_ln_scale)
            if idx < self.first_k_dense_replace:
                concated_ffn1_weight = np.concatenate(
                    [
                        state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.gate_proj.weight"],
                        state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.up_proj.weight"],
                    ],
                    axis=-1,
                )
                ffn1_weight_tensor = paddle.to_tensor(concated_ffn1_weight).cast(paddle.get_default_dtype())

                if self.use_weight_only:
                    ffn1_quanted_weight_tensor, ffn1_weight_scale_tensor = weight_quantize(
                        ffn1_weight_tensor, algo=self.quant_algo
                    )
                    self.transformer_block.ffn1_weights[idx].set_value(ffn1_quanted_weight_tensor)
                    self.transformer_block.ffn1_weights_scale[idx].set_value(ffn1_weight_scale_tensor)
                else:
                    self.transformer_block.ffn1_weights[idx].set_value(ffn1_weight_tensor)

                ffn2_weight_tensor = paddle.to_tensor(
                    state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.down_proj.weight"]
                ).cast(paddle.get_default_dtype())
                if self.use_weight_only:
                    ffn2_quanted_weight_tensor, ffn2_weight_scale_tensor = weight_quantize(
                        ffn2_weight_tensor, algo=self.quant_algo
                    )
                    self.transformer_block.ffn2_weights[idx].set_value(ffn2_quanted_weight_tensor)
                    self.transformer_block.ffn2_weights_scale[idx].set_value(ffn2_weight_scale_tensor)
                else:
                    self.transformer_block.ffn2_weights[idx].set_value(ffn2_weight_tensor)
            else:
                ffn1_weights = []
                ffn2_weights = []
                ffn1_scales = []
                ffn2_scales = []

                for expert_idx in range(self.n_routed_experts):
                    concated_gate_up_weight = np.concatenate(
                        [
                            state_dict[
                                f"{self.base_model_prefix}.layers.{idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                            ],
                            state_dict[
                                f"{self.base_model_prefix}.layers.{idx}.mlp.experts.{expert_idx}.up_proj.weight"
                            ],
                        ],
                        axis=-1,
                    )
                    ffn1_weight = paddle.to_tensor(concated_gate_up_weight).cast(dtype)
                    ffn2_weight = paddle.to_tensor(
                        state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.experts.{expert_idx}.down_proj.weight"]
                    ).cast(dtype)

                    if self.use_weight_only:
                        ffn1_quanted_weight, ffn1_weight_scale = weight_quantize(ffn1_weight, algo=self.quant_algo)
                        ffn2_quanted_weight, ffn2_weight_scale = weight_quantize(ffn2_weight, algo=self.quant_algo)
                        ffn1_weights.append(ffn1_quanted_weight.reshape([self.transformer_block.config.embed_dim, -1]))
                        ffn2_weights.append(ffn2_quanted_weight.reshape([-1, self.transformer_block.config.embed_dim]))
                        ffn1_scales.append(ffn1_weight_scale)
                        ffn2_scales.append(ffn2_weight_scale)
                    else:
                        ffn1_weights.append(ffn1_weight)
                        ffn2_weights.append(ffn2_weight)

                fused_moe_ffn1_weight = paddle.to_tensor(ffn1_weights)
                fused_moe_ffn2_weight = paddle.to_tensor(ffn2_weights)
                fused_moe_ffn1_weight_scale = paddle.to_tensor(ffn1_scales)
                fused_moe_ffn2_weight_scale = paddle.to_tensor(ffn2_scales)
                gate_weight = paddle.to_tensor(
                    state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.gate.weight"]
                ).cast("float32")

                if self.base_model_prefix.startswith("deepseek_v3"):
                    e_score_correction_bias = paddle.to_tensor(
                        state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.gate.e_score_correction_bias"]
                    ).cast("float32")
                    self.transformer_block.e_score_correction_biases[idx].set_value(e_score_correction_bias)

                self.transformer_block.ffn1_weights[idx].set_value(fused_moe_ffn1_weight)
                self.transformer_block.ffn2_weights[idx].set_value(fused_moe_ffn2_weight)
                self.transformer_block.gate_weights[idx].set_value(gate_weight)

                if self.use_weight_only:
                    self.transformer_block.ffn1_weights_scale[idx].set_value(fused_moe_ffn1_weight_scale)
                    self.transformer_block.ffn2_weights_scale[idx].set_value(fused_moe_ffn2_weight_scale)

                concated_gate_up_weight = np.concatenate(
                    [
                        state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.shared_experts.gate_proj.weight"],
                        state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.shared_experts.up_proj.weight"],
                    ],
                    axis=-1,
                )
                shared_expert_ffn1_weight = paddle.to_tensor(concated_gate_up_weight).cast(dtype)
                shared_expert_ffn2_weight = paddle.to_tensor(
                    state_dict[f"{self.base_model_prefix}.layers.{idx}.mlp.shared_experts.down_proj.weight"]
                ).cast(dtype)

                if self.use_weight_only:
                    shared_expert_ffn1_quanted_weight, shared_expert_ffn1_weight_scale = weight_quantize(
                        shared_expert_ffn1_weight, algo=self.quant_algo
                    )
                    self.transformer_block.shared_expert_ffn1_weights[idx].set_value(shared_expert_ffn1_quanted_weight)
                    self.transformer_block.shared_expert_ffn1_weights_scale[idx].set_value(
                        shared_expert_ffn1_weight_scale
                    )

                    shared_expert_ffn2_quanted_weight, shared_expert_ffn2_weight_scale = weight_quantize(
                        shared_expert_ffn2_weight, algo=self.quant_algo
                    )
                    self.transformer_block.shared_expert_ffn2_weights[idx].set_value(shared_expert_ffn2_quanted_weight)
                    self.transformer_block.shared_expert_ffn2_weights_scale[idx].set_value(
                        shared_expert_ffn2_weight_scale
                    )
                else:
                    self.transformer_block.shared_expert_ffn1_weights[idx].set_value(shared_expert_ffn1_weight)
                    self.transformer_block.shared_expert_ffn2_weights[idx].set_value(shared_expert_ffn2_weight)

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
        **kwargs,
    ):

        seq_lens_this_time = kwargs.get("seq_lens_this_time", None)
        draft_tokens = kwargs.get("draft_tokens", None)
        seq_lens_encoder = kwargs.get("seq_lens_encoder", None)

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
                rotary_embs=None,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cum_offsets=cum_offsets,
        )


@register_base_model
class MTPDeepseekV2BlockInferenceModel(DeepseekV2BlockInferenceModel):
    def __init__(self, config: DeepseekV2Config, base_model_prefix: str):
        super().__init__(config, base_model_prefix)
        from paddle.distributed.fleet.layers.mpu.mp_layers import ColumnParallelLinear

        self.enorm = DeepseekV2RMSNorm(config)
        self.hnorm = DeepseekV2RMSNorm(config)
        self.norm = DeepseekV2RMSNorm(config)

        if config.tensor_parallel_degree > 1:
            self.eh_proj = ColumnParallelLinear(
                self.hidden_size * 2, self.hidden_size, has_bias=True, gather_output=True, fuse_matmul_bias=True
            )
        else:
            self.eh_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias_attr=True)

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
        pre_hidden_states = kwargs.get("pre_hidden_states", None)
        ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k = self.remove_padding(
            input_ids, seq_lens_this_time, draft_tokens, seq_lens_encoder
        )

        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_k"] = cu_seqlens_k
        kwargs["padding_offsets"] = padding_offset
        kwargs["max_input_length"] = self.max_seq_len

        inputs_embeds = self.embed_tokens(ids_remove_padding)
        inputs_embeds = paddle.concat([self.enorm(inputs_embeds), self.hnorm(pre_hidden_states)], axis=-1)
        inputs_embeds = self.eh_proj(inputs_embeds)

        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids=input_ids,
                src=inputs_embeds,
                cum_offsets=cum_offsets,
                attn_mask=attention_mask,
                caches=caches,
                pre_caches=pre_caches,
                rotary_embs=rope_emb,
                post_rebuild_padding=True,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class DeepseekV2ForCausalLMBlockInferenceModel(GenerationBlockInferenceModel, DeepseekV2PretrainedModel):
    """
    Dynamic Batching for DeepseekV2 Model with pretraining tasks on top.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: DeepseekV2Config, base_model_prefix: str = "deepseek_v2"):
        super().__init__(config)
        self.base_model_prefix = base_model_prefix

        self.max_candidate_len = config.get("speculate_max_candidate_len", 5)
        self.verify_window = config.get("speculate_verify_window", 2)
        self.max_seq_len = config.max_seq_len
        self.return_full_hidden_states = config.get("return_full_hidden_states", False)

        self.deepseek_v2 = DeepseekV2BlockInferenceModel(config, base_model_prefix)
        if config.tie_word_embeddings:
            self.lm_head = DeepseekV2LMHead(
                config, embedding_weights=self.deepseek_v2.embed_tokens.weight, transpose_y=True
            )
            self.tie_weights()
        else:
            self.lm_head = DeepseekV2LMHead(config)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: DeepseekV2Config, is_split=True):

        logger.info("DeepseekV2 inference model _get_tensor_parallel_mappings")

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
                "eh_proj.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
            }

            # Column Linear
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_b_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.kv_b_proj.weight"] = partial(fn, is_column=True)

            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.down_proj.weight"] = partial(fn, is_column=False)

            for expert_idx in range(config.n_routed_experts):
                base_actions[f"layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = partial(fn, is_column=True)
                base_actions[f"layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions[f"layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = partial(fn, is_column=False)
            base_actions["layers.0.mlp.shared_experts.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.down_proj.weight"] = partial(fn, is_column=False)

            # MTP parts
            base_actions["layers.61.embed_tokens.weight"] = partial(fn, is_column=False)
            base_actions["layers.61.eh_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.61.shared_head.head.weight"] = partial(fn, is_column=True)

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
        cls, config: DeepseekV2Config, max_batch_size: int = None, max_length: int = None
    ) -> list[list[int]]:
        """get cache_kvs tensor for DeepseekV2 model

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
            cache_k_shape = [
                max_block_nums,
                config.num_key_value_heads // max(config.tensor_parallel_degree, 1),
                config.block_size,
                config.qk_nope_head_dim + config.qk_rope_head_dim,
            ]
            cache_v_shape = [
                max_block_nums,
                config.num_key_value_heads // max(config.tensor_parallel_degree, 1),
                config.block_size,
                config.v_head_dim,
            ]
            cache_kvs.append(cache_k_shape)
            cache_kvs.append(cache_v_shape)
        return cache_kvs

    def prepare_inputs_for_generation(self, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        src_mask = kwargs.get("src_mask", None)
        block_tables = kwargs.get("block_tables", None)

        pre_caches = kwargs.get("pre_caches", None)
        caches = kwargs.get("caches", None)

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
            "rope_emb": None,
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
        outputs = self.deepseek_v2(
            input_ids,
            src_mask=src_mask,
            caches=caches,
            rope_emb=None,
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
        if self.return_full_hidden_states:
            from paddlenlp_ops import rebuild_padding_v2

            full_hidden_states = outputs[0]
            cum_offsets = outputs[1]
            hidden_states = rebuild_padding_v2(
                full_hidden_states,
                cum_offsets,
                seq_lens_decoder,
                seq_lens_encoder,
                output_padding_offset,
                self.max_seq_len,
            )
        else:
            hidden_states = outputs[0]
        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        )
        if self.return_full_hidden_states:
            return logits, full_hidden_states
        else:
            return logits

        return logits

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(
                paddle.to_tensor(state_dict["lm_head.weight"]).cast(self.lm_head.weight.dtype)
            )
        self.deepseek_v2.set_state_dict({k: state_dict[k] for k in state_dict.keys()})


class MTPDeepseekV2ForCausalLMBlockInferenceModel(DeepseekV2ForCausalLMBlockInferenceModel):
    def __init__(self, config, base_model_prefix):
        super(DeepseekV2ForCausalLMBlockInferenceModel, self).__init__(config, base_model_prefix="deepseek_v3_mtp")
        self.max_candidate_len = config.get("speculate_max_candidate_len", 5)
        self.verify_window = config.get("speculate_verify_window", 2)
        self.max_seq_len = config.max_seq_len

        self.mtp = MTPDeepseekV2BlockInferenceModel(config, base_model_prefix="deepseek_v3_mtp")
        self.tensor_parallel_rank = config.tensor_parallel_rank
        if config.tie_word_embeddings:
            self.lm_head = DeepseekV2LMHead(config, embedding_weights=self.llama.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = DeepseekV2LMHead(config)

    def prepare_inputs_for_generation(self, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        src_mask = kwargs.get("src_mask", None)
        block_tables = kwargs.get("block_tables", None)

        pre_caches = kwargs.get("pre_caches", None)
        caches = kwargs.get("caches", None)

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
        hidden_states = kwargs.get("hidden_states", None)

        model_inputs = {
            "input_ids": input_ids,
            "src_mask": src_mask,
            "rope_emb": None,
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
            "pre_hidden_states": hidden_states,
        }
        return model_inputs

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(
                paddle.to_tensor(state_dict["lm_head.weight"]).cast(self.lm_head.weight.dtype)
            )

        self.mtp.enorm.weight.set_value(
            paddle.to_tensor(state_dict["deepseek_v3_mtp.enorm.weight"]).cast(self.lm_head.weight.dtype)
        )
        self.mtp.hnorm.weight.set_value(
            paddle.to_tensor(state_dict["deepseek_v3_mtp.hnorm.weight"]).cast(self.lm_head.weight.dtype)
        )
        self.mtp.norm.weight.set_value(
            paddle.to_tensor(state_dict["deepseek_v3_mtp.norm.weight"]).cast(self.lm_head.weight.dtype)
        )
        self.mtp.eh_proj.weight.set_value(
            paddle.to_tensor(state_dict["deepseek_v3_mtp.eh_proj.weight"]).cast(self.lm_head.weight.dtype)
        )

        self.mtp.set_state_dict({k: state_dict[k] for k in state_dict.keys()})

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
        pre_hidden_states=None,
    ):
        outputs = self.mtp(
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
            pre_hidden_states=pre_hidden_states,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        )

        return logits, hidden_states
