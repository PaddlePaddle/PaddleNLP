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

from typing import Any

import paddle
from paddle import nn

import paddlenlp
from paddlenlp.transformers import (
    LlamaConfig,
    LlamaModel,
    LlamaPretrainedModel,
    PretrainedConfig,
    PretrainedModel,
)
from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)

from .score_model_utils import ScoreModelMixin, ScoreModelOutput


class LlamaModelForScore(ScoreModelMixin, LlamaPretrainedModel):
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.llama = LlamaModel(config)

        # config.architectures = [self.__class__.__name__]
        self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.llama.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.llama.embed_tokens = value

    def get_decoder(self) -> PretrainedModel:
        return self.llama

    def set_decoder(self, decoder: PretrainedModel) -> None:
        self.llama = decoder

    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        past_key_values: list[paddle.Tensor] | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor] | ScoreModelOutput:
        assert attention_mask is not None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.llama(
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
        hidden_states = outputs[0]  # size = (B, L, E)
        return self.get_score(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )

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
            model_mappings.extend(
                [
                    ["score_head.weight", "score_head.weight", "transpose"],
                    ["normalizer.var", "normalizer.var"],
                    ["normalizer.mean", "normalizer.mean"],
                    ["normalizer.count", "normalizer.count"],
                ]
            )

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings


paddlenlp.transformers.LlamaModelForScore = LlamaModelForScore
