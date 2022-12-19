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

from paddlenlp.transformers import PretrainedModel, XLNetModel
from paddlenlp.utils.converter import StateDictNameMapping, Converter

__all__ = ["XLNetConverter"]


class XLNetConverter(Converter):
    _ignore_state_dict_keys = ["embeddings.position_ids"]
    num_layer_key = "n_layer"
    architectures: Dict[str, Type[PretrainedModel]] = {"XLNetModel": XLNetModel}

    def get_paddle_pytorch_model_classes(self):
        from paddlenlp.transformers import XLNetModel as PaddleModel
        from transformers import XLNetModel as PytorchModel

        return PaddleModel, PytorchModel

    def get_name_mapping(self, config_or_num_layers: Union[dict, int] = None) -> List[StateDictNameMapping]:
        num_layer = self.resolve_num_layer(config_or_num_layers)

        hard_mapping = [
            ["mask_emb", "mask_emb"],
            ["word_embedding.weight", "word_embedding.weight"],
        ]

        for layer_index in range(num_layer):
            layer_mappings = [
                [f"layer.{layer_index}.rel_attn.q", f"layer.{layer_index}.rel_attn.q", "merge_last_two_dim"],
                [f"layer.{layer_index}.rel_attn.k", f"layer.{layer_index}.rel_attn.k", "merge_last_two_dim"],
                [f"layer.{layer_index}.rel_attn.v", f"layer.{layer_index}.rel_attn.v", "merge_last_two_dim"],
                [f"layer.{layer_index}.rel_attn.o", f"layer.{layer_index}.rel_attn.o", "merge_last_two_dim"],
                [f"layer.{layer_index}.rel_attn.r", f"layer.{layer_index}.rel_attn.r", "merge_last_two_dim"],
                [f"layer.{layer_index}.rel_attn.r_r_bias", f"layer.{layer_index}.rel_attn.r_r_bias"],
                [f"layer.{layer_index}.rel_attn.r_s_bias", f"layer.{layer_index}.rel_attn.r_s_bias"],
                [f"layer.{layer_index}.rel_attn.r_w_bias", f"layer.{layer_index}.rel_attn.r_w_bias"],
                [f"layer.{layer_index}.rel_attn.seg_embed", f"layer.{layer_index}.rel_attn.seg_embed"],
                [f"layer.{layer_index}.rel_attn.layer_norm.weight", f"layer.{layer_index}.rel_attn.layer_norm.weight"],
                [f"layer.{layer_index}.rel_attn.layer_norm.bias", f"layer.{layer_index}.rel_attn.layer_norm.bias"],
                [f"layer.{layer_index}.ff.layer_norm.weight", f"layer.{layer_index}.ff.layer_norm.weight"],
                [f"layer.{layer_index}.ff.layer_norm.bias", f"layer.{layer_index}.ff.layer_norm.bias"],
                [f"layer.{layer_index}.ff.layer_1.weight", f"layer.{layer_index}.ff.layer_1.weight", "transpose"],
                [f"layer.{layer_index}.ff.layer_2.weight", f"layer.{layer_index}.ff.layer_2.weight", "transpose"],
                [f"layer.{layer_index}.ff.layer_1.bias", f"layer.{layer_index}.ff.layer_1.bias"],
                [f"layer.{layer_index}.ff.layer_2.bias", f"layer.{layer_index}.ff.layer_2.bias"],
            ]
            hard_mapping.extend(layer_mappings)
        return [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mapping)]
