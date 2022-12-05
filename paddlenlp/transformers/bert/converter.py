# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing import List, Union, Dict, Type

from paddlenlp.transformers import PretrainedModel, BertForMaskedLM, BertModel
from paddlenlp.utils.import_utils import import_module
from paddlenlp.utils.converter import StateDictNameMapping, Converter

__all__ = ["BertConverter"]


class BertConverter(Converter):
    """Bert Converter which handle the converting operations"""

    _ignore_state_dict_keys = ["embeddings.position_ids"]
    architectures: Dict[str, Type[PretrainedModel]] = {"BertForMaskedLM": BertForMaskedLM, "BertModel": BertModel}

    def get_paddle_pytorch_model_classes(self):
        pytorch_model_class = import_module("transformers.BertModel")
        return BertModel, pytorch_model_class

    def get_name_mapping(self, config_or_num_layers: Union[dict, int] = None) -> List[StateDictNameMapping]:
        num_layer = self.resolve_num_layer(config_or_num_layers)

        mappings: List[StateDictNameMapping] = []

        hard_mappings = [
            ["embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"],
            ["embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight"],
            ["embeddings.token_type_embeddings.weight", "embeddings.token_type_embeddings.weight"],
            ["embeddings.LayerNorm.weight", "embeddings.layer_norm.weight"],
            ["embeddings.LayerNorm.bias", "embeddings.layer_norm.bias"],
            ["pooler.dense.weight", "pooler.dense.weight", "transpose"],
            ["pooler.dense.bias", "pooler.dense.bias"],
            # for TokenClassification
        ]
        for layer_index in range(num_layer):
            layer_mappings = [
                [
                    f"encoder.layer.{layer_index}.attention.self.query.weight",
                    f"encoder.layers.{layer_index}.self_attn.q_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.query.bias",
                    f"encoder.layers.{layer_index}.self_attn.q_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.key.weight",
                    f"encoder.layers.{layer_index}.self_attn.k_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.key.bias",
                    f"encoder.layers.{layer_index}.self_attn.k_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.value.weight",
                    f"encoder.layers.{layer_index}.self_attn.v_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.value.bias",
                    f"encoder.layers.{layer_index}.self_attn.v_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.dense.weight",
                    f"encoder.layers.{layer_index}.self_attn.out_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.dense.bias",
                    f"encoder.layers.{layer_index}.self_attn.out_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.intermediate.dense.weight",
                    f"encoder.layers.{layer_index}.linear1.weight",
                    "transpose",
                ],
                [f"encoder.layer.{layer_index}.intermediate.dense.bias", f"encoder.layers.{layer_index}.linear1.bias"],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight",
                    f"encoder.layers.{layer_index}.norm1.weight",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias",
                    f"encoder.layers.{layer_index}.norm1.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.output.dense.weight",
                    f"encoder.layers.{layer_index}.linear2.weight",
                    "transpose",
                ],
                [f"encoder.layer.{layer_index}.output.dense.bias", f"encoder.layers.{layer_index}.linear2.bias"],
                [f"encoder.layer.{layer_index}.output.LayerNorm.weight", f"encoder.layers.{layer_index}.norm2.weight"],
                [f"encoder.layer.{layer_index}.output.LayerNorm.bias", f"encoder.layers.{layer_index}.norm2.bias"],
            ]
            hard_mappings.extend(layer_mappings)
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mappings)]
        return mappings
