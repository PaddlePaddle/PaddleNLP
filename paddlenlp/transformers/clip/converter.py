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
from typing import List, Union, Dict, Type, Tuple

from paddlenlp.utils.import_utils import import_module
from paddlenlp.utils.log import logger
from paddlenlp.transformers import CLIPModel, PretrainedModel
from paddlenlp.utils.converter import *


class CLIPConverter(Converter):
    """CLIP Converter which handle the converting operations"""

    num_layer_key = "num_hidden_layers"
    _ignore_state_dict_keys = ["text_model.embeddings.position_ids", "vision_model.embeddings.position_ids"]
    architectures: Dict[str, Type[PretrainedModel]] = {"CLIPModel": CLIPModel}
    try_compare_logits: bool = False

    def get_paddle_pytorch_model_classes(self) -> Tuple[Type, Type]:
        """get paddle & pytorch model class

        Returns:
            Tuple[object, object]: the class of pretrained-model
        """
        pytorch_model_class = import_module("transformers.CLIPModel")
        return CLIPModel, pytorch_model_class

    def get_inputs(self):
        return super().get_inputs()

    @classmethod
    def resolve_num_layer(cls, config_or_num_layers: Union[dict, int, list] = None) -> int:
        """resolve the number of transformer layer based on the key of model config, eg: `num_hidden_layers` in BertModel
        Args:
            config_or_num_layers (Union[dict, int], optional): the instance of config or num_layers. Defaults to None.
        Raises:
            ValueError: when `config_or_num_layers` is not dict/int, it will raise the error
        Returns:
            int: the number of transformer layer
        """
        if isinstance(config_or_num_layers, dict):
            return config_or_num_layers["text_layers"], config_or_num_layers["vision_layers"]

        elif isinstance(config_or_num_layers, int):
            num_layer = config_or_num_layers
        else:
            raise ValueError(f"the type of config_or_num_layers<{config_or_num_layers}> should be one of <dict, int>")
        return num_layer

    def convert_config(self, pytorch_config: dict) -> dict:
        """convert torch config to paddle config
        Args:
            pytorch_config (dict): the object of pytorch config file
        Returns:
            dict: the final converted config object
        """
        paddle_config = {
            # vision
            "image_resolution": pytorch_config["vision_config"]["image_size"],
            "vision_layers": pytorch_config["vision_config"]["num_hidden_layers"],
            "vision_heads": pytorch_config["vision_config"]["num_attention_heads"],
            "vision_embed_dim": pytorch_config["vision_config"]["hidden_size"],
            "vision_patch_size": pytorch_config["vision_config"]["patch_size"],
            "vision_mlp_ratio": pytorch_config["vision_config"]["intermediate_size"]
            // pytorch_config["vision_config"]["hidden_size"],
            "vision_hidden_act": pytorch_config["vision_config"]["hidden_act"],
            # text
            "max_text_length": pytorch_config["text_config"]["max_position_embeddings"],
            "vocab_size": pytorch_config["text_config"]["vocab_size"],
            "text_embed_dim": pytorch_config["text_config"]["hidden_size"],
            "text_heads": pytorch_config["text_config"]["num_attention_heads"],
            "text_layers": pytorch_config["text_config"]["num_hidden_layers"],
            "text_hidden_act": pytorch_config["text_config"]["hidden_act"],
            "projection_dim": pytorch_config["projection_dim"],
            "initializer_range": pytorch_config["text_config"]["initializer_range"],
            "initializer_factor": pytorch_config["initializer_factor"],
            "logit_scale_init_value": pytorch_config["logit_scale_init_value"],
            "init_class": "CLIPModel",
        }
        return paddle_config

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
        from paddlenlp.transformers import CLIPModel as PaddleCLIPModel

        pytorch_clip_model_class = import_module("transformers.CLIPModel")
        return PaddleCLIPModel, pytorch_clip_model_class

    def get_name_mapping(self, config_or_num_layers: Union[dict, int] = None) -> List[StateDictNameMapping]:
        num_text_layer, num_vision_layer = self.resolve_num_layer(config_or_num_layers)
        mappings: List[StateDictNameMapping] = []

        hard_mappings = [
            # text model
            ["text_model.embeddings.token_embedding.weight", "text_model.token_embedding.weight"],
            ["text_model.embeddings.position_embedding.weight", "text_model.positional_embedding.weight"],
            ["text_model.final_layer_norm.weight", "text_model.ln_final.weight"],
            ["text_model.final_layer_norm.bias", "text_model.ln_final.bias"],
            # vision model
            ["vision_model.embeddings.class_embedding", "vision_model.class_embedding"],
            ["vision_model.embeddings.patch_embedding.weight", "vision_model.conv1.weight"],
            ["vision_model.embeddings.position_embedding.weight", "vision_model.positional_embedding.weight"],
            ["vision_model.pre_layrnorm.weight", "vision_model.ln_pre.weight"],
            ["vision_model.pre_layrnorm.bias", "vision_model.ln_pre.bias"],
            ["vision_model.post_layernorm.weight", "vision_model.ln_post.weight"],
            ["vision_model.post_layernorm.bias", "vision_model.ln_post.bias"],
            # projection
            ["visual_projection.weight", "vision_projection", "transpose"],
            ["text_projection.weight", "text_projection", "transpose"],
            ["logit_scale", "logit_scale"],
        ]
        for layer_index in range(num_text_layer):
            text_model_layer_mappings = [
                # qkv out
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.q_proj.weight",
                    f"text_model.transformer.layers.{layer_index}.self_attn.q_proj.weight",
                    "transpose",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.q_proj.bias",
                    f"text_model.transformer.layers.{layer_index}.self_attn.q_proj.bias",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.k_proj.weight",
                    f"text_model.transformer.layers.{layer_index}.self_attn.k_proj.weight",
                    "transpose",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.k_proj.bias",
                    f"text_model.transformer.layers.{layer_index}.self_attn.k_proj.bias",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.v_proj.weight",
                    f"text_model.transformer.layers.{layer_index}.self_attn.v_proj.weight",
                    "transpose",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.v_proj.bias",
                    f"text_model.transformer.layers.{layer_index}.self_attn.v_proj.bias",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.out_proj.weight",
                    f"text_model.transformer.layers.{layer_index}.self_attn.out_proj.weight",
                    "transpose",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.out_proj.bias",
                    f"text_model.transformer.layers.{layer_index}.self_attn.out_proj.bias",
                ],
                # fc1
                [
                    f"text_model.encoder.layers.{layer_index}.mlp.fc1.weight",
                    f"text_model.transformer.layers.{layer_index}.linear1.weight",
                    "transpose",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.mlp.fc1.bias",
                    f"text_model.transformer.layers.{layer_index}.linear1.bias",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.layer_norm1.weight",
                    f"text_model.transformer.layers.{layer_index}.norm1.weight",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.layer_norm1.bias",
                    f"text_model.transformer.layers.{layer_index}.norm1.bias",
                ],
                # fc2
                [
                    f"text_model.encoder.layers.{layer_index}.mlp.fc2.weight",
                    f"text_model.transformer.layers.{layer_index}.linear2.weight",
                    "transpose",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.mlp.fc2.bias",
                    f"text_model.transformer.layers.{layer_index}.linear2.bias",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.layer_norm2.weight",
                    f"text_model.transformer.layers.{layer_index}.norm2.weight",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.layer_norm2.bias",
                    f"text_model.transformer.layers.{layer_index}.norm2.bias",
                ],
            ]
            hard_mappings.extend(text_model_layer_mappings)

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
