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

import os
from typing import Dict, List, Type, Union

from ...utils.converter import Converter, StateDictNameMapping
from ...utils.import_utils import import_module
from ..model_utils import PretrainedModel
from .modeling import ChineseCLIPModel


class ChineseCLIPConverter(Converter):
    """Chinese-CLIP Converter which handle the converting operations"""

    num_layer_key = "num_hidden_layers"
    _ignore_state_dict_keys = ["text_model.embeddings.position_ids", "vision_model.embeddings.position_ids"]
    architectures: Dict[str, Type[PretrainedModel]] = {"ChineseCLIPModel": ChineseCLIPModel}
    try_compare_logits: bool = False

    def resolve_num_layer(self, config_or_num_layers: Union[dict, int] = None) -> int:
        """resolve the number of transformer layer based on the key of model config, eg: `num_hidden_layers` in CLIPModel
        Args:
            config_or_num_layers (Union[dict, int], optional): the instance of config or num_layers. Defaults to None.
        Raises:
            ValueError: when `config_or_num_layers` is not dict/int, it will raise the error
        Returns:
            int: the number of transformer layer
        """
        if isinstance(config_or_num_layers, dict):
            num_text_layer = 0
            num_vision_layer = 0

            if self.model_type in ["chinese_clip", "chinese_clip_text_model"]:
                if "text_config" in config_or_num_layers:
                    num_text_layer = config_or_num_layers["text_config"][self.num_layer_key]
                else:
                    num_text_layer = config_or_num_layers[self.num_layer_key]

            if self.model_type in ["chinese_clip", "chinese_clip_vision_model"]:
                if "vision_config" in config_or_num_layers:
                    num_vision_layer = config_or_num_layers["vision_config"][self.num_layer_key]
                else:
                    num_vision_layer = config_or_num_layers[self.num_layer_key]

            return num_text_layer, num_vision_layer
        elif isinstance(config_or_num_layers, int):
            num_layer = config_or_num_layers
        else:
            raise ValueError(f"the type of config_or_num_layers<{config_or_num_layers}> should be one of <dict, int>")
        return num_layer, num_layer

    def load_torch_weight_file(self, model_file: str):
        """load torch weight file with torch which should be removed later.
        Args:
            model_file (str): the path of pytorch model file
        Returns:
            Dict[str, ndarray]: the state dict object of loaded pytorch state dict
        """
        import torch

        state_dict = torch.load(model_file)
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].numpy()
            if state_dict[key].ndim == 0:
                state_dict[key] = state_dict[key].reshape((1,))
        return state_dict

    def get_paddle_pytorch_model_classes(self):
        paddle_clip_model_class = import_module(f"paddlenlp.transformers.{self.architecture}")
        pytorch_clip_model_class = import_module(f"transformers.{self.architecture}")
        return paddle_clip_model_class, pytorch_clip_model_class

    def get_name_mapping(self, config_or_num_layers: Union[dict, int] = None) -> List[StateDictNameMapping]:
        self.model_type = (
            config_or_num_layers.get("model_type", "chinese_clip")
            if isinstance(config_or_num_layers, dict)
            else "chinese_clip"
        )
        self.architecture = (
            config_or_num_layers.get("architectures", ["ChineseCLIPModel"])[0]
            if isinstance(config_or_num_layers, dict)
            else "ChineseCLIPModel"
        )

        num_text_layer, num_vision_layer = self.resolve_num_layer(config_or_num_layers)

        mappings: List[StateDictNameMapping] = []
        if self.model_type == "chinese_clip":
            hard_mappings = [["logit_scale", "logit_scale"]]
        else:
            hard_mappings = []

        # text model (bert model)
        if num_text_layer > 0:
            text_model_layer_mappings = [
                ["text_model.embeddings.word_embeddings.weight", "text_model.embeddings.word_embeddings.weight"],
                [
                    "text_model.embeddings.position_embeddings.weight",
                    "text_model.embeddings.position_embeddings.weight",
                ],
                [
                    "text_model.embeddings.token_type_embeddings.weight",
                    "text_model.embeddings.token_type_embeddings.weight",
                ],
                ["text_model.embeddings.LayerNorm.weight", "text_model.embeddings.layer_norm.weight"],
                ["text_model.embeddings.LayerNorm.bias", "text_model.embeddings.layer_norm.bias"],
                ["text_projection.weight", "text_projection", "transpose"],
                # donot add pooler
                # ["text_model.pooler.dense.weight", "text_model.pooler.dense.weight", "transpose"],
                # ["text_model.pooler.dense.bias", "text_model.pooler.dense.bias"],
            ]
            hard_mappings.extend(text_model_layer_mappings)

            for layer_index in range(num_text_layer):
                text_model_layer_mappings = [
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.self.query.weight",
                        f"text_model.encoder.layers.{layer_index}.self_attn.q_proj.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.self.query.bias",
                        f"text_model.encoder.layers.{layer_index}.self_attn.q_proj.bias",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.self.key.weight",
                        f"text_model.encoder.layers.{layer_index}.self_attn.k_proj.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.self.key.bias",
                        f"text_model.encoder.layers.{layer_index}.self_attn.k_proj.bias",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.self.value.weight",
                        f"text_model.encoder.layers.{layer_index}.self_attn.v_proj.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.self.value.bias",
                        f"text_model.encoder.layers.{layer_index}.self_attn.v_proj.bias",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.output.dense.weight",
                        f"text_model.encoder.layers.{layer_index}.self_attn.out_proj.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.output.dense.bias",
                        f"text_model.encoder.layers.{layer_index}.self_attn.out_proj.bias",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.intermediate.dense.weight",
                        f"text_model.encoder.layers.{layer_index}.linear1.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.intermediate.dense.bias",
                        f"text_model.encoder.layers.{layer_index}.linear1.bias",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.output.LayerNorm.weight",
                        f"text_model.encoder.layers.{layer_index}.norm1.weight",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.attention.output.LayerNorm.bias",
                        f"text_model.encoder.layers.{layer_index}.norm1.bias",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.output.dense.weight",
                        f"text_model.encoder.layers.{layer_index}.linear2.weight",
                        "transpose",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.output.dense.bias",
                        f"text_model.encoder.layers.{layer_index}.linear2.bias",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.output.LayerNorm.weight",
                        f"text_model.encoder.layers.{layer_index}.norm2.weight",
                    ],
                    [
                        f"text_model.encoder.layer.{layer_index}.output.LayerNorm.bias",
                        f"text_model.encoder.layers.{layer_index}.norm2.bias",
                    ],
                ]
                hard_mappings.extend(text_model_layer_mappings)

        # vision model
        if num_vision_layer > 0:
            vision_model_layer_mappings = [
                ["vision_model.embeddings.class_embedding", "vision_model.class_embedding"],
                ["vision_model.embeddings.patch_embedding.weight", "vision_model.conv1.weight"],
                ["vision_model.embeddings.position_embedding.weight", "vision_model.positional_embedding.weight"],
                ["vision_model.pre_layrnorm.weight", "vision_model.ln_pre.weight"],
                ["vision_model.pre_layrnorm.bias", "vision_model.ln_pre.bias"],
                ["vision_model.post_layernorm.weight", "vision_model.ln_post.weight"],
                ["vision_model.post_layernorm.bias", "vision_model.ln_post.bias"],
                ["visual_projection.weight", "vision_projection", "transpose"],
            ]
            hard_mappings.extend(vision_model_layer_mappings)
            for layer_index in range(num_vision_layer):
                vision_model_layer_mappings = [
                    # qkv out
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.q_proj.weight",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.q_proj.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.q_proj.bias",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.q_proj.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.k_proj.weight",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.k_proj.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.k_proj.bias",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.k_proj.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.v_proj.weight",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.v_proj.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.v_proj.bias",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.v_proj.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.out_proj.weight",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.out_proj.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.self_attn.out_proj.bias",
                        f"vision_model.transformer.layers.{layer_index}.self_attn.out_proj.bias",
                    ],
                    # fc1
                    [
                        f"vision_model.encoder.layers.{layer_index}.mlp.fc1.weight",
                        f"vision_model.transformer.layers.{layer_index}.linear1.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.mlp.fc1.bias",
                        f"vision_model.transformer.layers.{layer_index}.linear1.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.layer_norm1.weight",
                        f"vision_model.transformer.layers.{layer_index}.norm1.weight",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.layer_norm1.bias",
                        f"vision_model.transformer.layers.{layer_index}.norm1.bias",
                    ],
                    # fc2
                    [
                        f"vision_model.encoder.layers.{layer_index}.mlp.fc2.weight",
                        f"vision_model.transformer.layers.{layer_index}.linear2.weight",
                        "transpose",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.mlp.fc2.bias",
                        f"vision_model.transformer.layers.{layer_index}.linear2.bias",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.layer_norm2.weight",
                        f"vision_model.transformer.layers.{layer_index}.norm2.weight",
                    ],
                    [
                        f"vision_model.encoder.layers.{layer_index}.layer_norm2.bias",
                        f"vision_model.transformer.layers.{layer_index}.norm2.bias",
                    ],
                ]
                hard_mappings.extend(vision_model_layer_mappings)

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mappings)]
        return mappings

    def convert(self, input_dir: str, output_dir: str):
        super().convert(input_dir, output_dir)
        old_config_file = os.path.join(output_dir, "model_config.json")
        new_config_file = os.path.join(output_dir, "config.json")
        os.rename(old_config_file, new_config_file)
