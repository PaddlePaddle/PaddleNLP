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
        """
        Initializes a `LlamaForSequenceClassification` model.

        Args:
            config (PretrainedConfig): Model configuration class with all the parameters of the model.
            kwargs (Any, optional): Additional keyword arguments passed along to the `__init__` of the parent class.
                This is necessary because of how `transformers.AutoModelWithHead` is designed. Defaults to `None`.

        Raises:
            TypeError: If the config is not an instance of `PretrainedConfig`.
        """
        super().__init__(config)
        self.llama = LlamaModel(config)

        # config.architectures = [self.__class__.__name__]
        self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)

    def get_input_embeddings(self) -> nn.Embedding:
        """
        返回输入嵌入的nn.Embedding对象，该对象用于将输入序列转换为嵌入向量。
        如果模型没有使用嵌入，则返回None。

        Returns:
            Optional[nn.Embedding]: 输入嵌入的nn.Embedding对象，或者None（如果没有使用嵌入）。
        """
        return self.llama.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """
        Set the input embeddings to be used for the model.

        Args:
            value (nn.Embedding): The embedding layer to use.

        Returns:
            NoneType: No return value is returned. Instead, the input embeddings are updated in-place.
        """
        self.llama.embed_tokens = value

    def get_decoder(self) -> PretrainedModel:
        """
        获取解码器模型。

        Returns:
            PretrainedModel (Pytorch): 返回解码器模型，类型为Pytorch的PretrainedModel。
        """
        return self.llama

    def set_decoder(self, decoder: PretrainedModel) -> None:
        """
        设置解码器，用于进行文本生成。

        Args:
            decoder (PretrainedModel): 预训练的模型对象，需要是一个有效的解码器。

        Returns:
            None; 无返回值。
        """
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
        """
        句子的前向传播过程。

        Args:
            input_ids (paddle.Tensor):
                输入序列的ID，形状为（batch_size, sequence_length）。
            attention_mask (paddle.Tensor):
                用于区分padding和非padding元素的mask，形状为（batch_size, sequence_length），值为0或1。
            position_ids (paddle.Tensor, optional):
                input_ids对应的位置ID，形状为（batch_size, sequence_length），默认为None。
            past_key_values (list[paddle.Tensor], optional):
                包含所有预处理器的键和值，默认为None。
            inputs_embeds (paddle.Tensor, optional):
                输入序列的嵌入，形状为（batch_size, sequence_length, embedding_dimension），默认为None。
            use_cache (bool, optional):
                是否使用缓存，默认为None。
            output_attentions (bool, optional):
                是否返回注意力张量，默认为None。
            output_hidden_states (bool, optional):
                是否返回隐藏状态，默认为None。
            return_dict (bool, optional):
                是否返回字典格式的结果，默认为None。

        Returns:
            tuple[paddle.Tensor, paddle.Tensor] or ScoreModelOutput:
                如果`return_dict`为True，则返回一个ScoreModelOutput类型的元组，其中包含两个元素：得分和附加信息；否则，返回一个tuple，其中包含得分和附加信息。
        Raises:
            AssertionError:
                当`attention_mask`不为None时引发。
        """
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
        """
        获取模型的名称映射列表，包括模型参数和额外的映射。
        如果配置中没有"LlamaModel"，则将基础模型前缀添加到每个映射中。

        Args:
            config (LlamaConfig): 配置对象，其中包含模型参数。

        Returns:
            list[StateDictNameMapping]: 一个包含模型参数和额外映射的名称映射列表。
            每个元素是一个三元组（原始名称，转换后的名称，转换类型），其中转换类型可以为None、"transpose"或者"add_prefix"。
        """
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"layers.{layer_index}.self_attn.q_proj.weight",
                    None,
                    "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attn.k_proj.weight",
                    None,
                    "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attn.v_proj.weight",
                    None,
                    "transpose",
                ],
                [
                    f"layers.{layer_index}.self_attn.o_proj.weight",
                    None,
                    "transpose",
                ],
                [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                [
                    f"layers.{layer_index}.mlp.gate_proj.weight",
                    None,
                    "transpose",
                ],
                [
                    f"layers.{layer_index}.mlp.down_proj.weight",
                    None,
                    "transpose",
                ],
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
