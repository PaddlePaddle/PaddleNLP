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
from typing import List, Union

from paddlenlp.utils.import_utils import import_module
from paddlenlp.utils.converter import Converter, StateDictNameMapping


class CLIPConverter(Converter):
    """CLIP Converter which handle the converting operations"""
    num_layer_key = "text_layer"
    _ignore_state_dict_keys = [
        'text_model.embeddings.position_ids',
        'vision_model.embeddings.position_ids'
    ]

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
                state_dict[key] = state_dict[key].reshape((1, ))
        return state_dict

    def convert_config(self, pytorch_config: dict) -> dict:
        return {
            # vision
            "image_resolution": pytorch_config['image_resolution'],
            "vision_layers": 12,
            "vision_heads": 12,
            "vision_mlp_ratio": pytorch_config['vision']['mlp_ratio'],
            "vision_embed_dim": 768,
            "vision_patch_size": 32,
            "vision_hidden_act": "quick_gelu",
            # text
            "max_text_length": 77,
            "vocab_size": 49408,
            "text_embed_dim": 512,
            "text_heads": 8,
            "text_layers": 12,
            "text_hidden_act": "quick_gelu",
            # others
            "projection_dim": 512,
            "initializer_range": 0.02,
            "logit_scale_init_value": 2.6592
        }

    def get_paddle_pytorch_model_classes(self):
        from paddlenlp.transformers import CLIPModel as PaddleCLIPModel
        pytorch_clip_model_class = import_module("transformers.CLIPModel")
        return PaddleCLIPModel, pytorch_clip_model_class

    def get_name_mapping(
        self,
        config_or_num_layers: Union[dict, int] = None
    ) -> List[StateDictNameMapping]:
        num_layer = self.resolve_num_layer(config_or_num_layers)

        mappings: List[StateDictNameMapping] = []

        hard_mappings = [
            # text model
            [
                'text_model.embeddings.token_embedding.weight',
                'text_model.token_embedding.weight'
            ],
            [
                'text_model.embeddings.position_embedding.weight',
                'text_model.positional_embedding.weight'
            ],
            [
                'text_model.final_layer_norm.weight',
                'text_model.ln_final.weight'
            ],
            ['text_model.final_layer_norm.bias', 'text_model.ln_final.bias'],

            # vision model
            [
                'vision_model.embeddings.class_embedding',
                'vision_model.class_embedding'
            ],
            [
                'vision_model.embeddings.patch_embedding.weight',
                'vision_model.conv1.weight'
            ],
            [
                'vision_model.embeddings.position_embedding.weight',
                'vision_model.positional_embedding.weight'
            ],
            ['vision_model.pre_layrnorm.weight', 'vision_model.ln_pre.weight'],
            ['vision_model.pre_layrnorm.bias', 'vision_model.ln_pre.bias'],
            [
                'vision_model.post_layernorm.weight',
                'vision_model.ln_post.weight'
            ],
            ['vision_model.post_layernorm.bias', 'vision_model.ln_post.bias'],

            # projection
            ['visual_projection.weight', 'vision_projection'],
            ['text_projection.weight', 'text_projection'],
            ['logit_scale', 'logit_scale']
        ]
        for layer_index in range(num_layer):
            text_model_layer_mappings = [
                # qkv out
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.q_proj.weight",
                    f"text_model.transformer.layers.{layer_index}.self_attn.q_proj.weight",
                    "transpose"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.q_proj.bias",
                    f"text_model.transformer.layers.{layer_index}.self_attn.q_proj.bias",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.k_proj.weight",
                    f"text_model.transformer.layers.{layer_index}.self_attn.k_proj.weight",
                    "transpose"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.k_proj.bias",
                    f"text_model.transformer.layers.{layer_index}.self_attn.k_proj.bias",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.v_proj.weight",
                    f"text_model.transformer.layers.{layer_index}.self_attn.v_proj.weight",
                    "transpose"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.v_proj.bias",
                    f"text_model.transformer.layers.{layer_index}.self_attn.v_proj.bias",
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.out_proj.weight",
                    f"text_model.transformer.layers.{layer_index}.self_attn.out_proj.weight",
                    "transpose"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.self_attn.out_proj.bias",
                    f"text_model.transformer.layers.{layer_index}.self_attn.out_proj.bias",
                ],
                # fc1
                [
                    f"text_model.encoder.layers.{layer_index}.mlp.fc1.weight",
                    f"text_model.transformer.layers.{layer_index}.linear1.weight",
                    "transpose"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.mlp.fc1.bias",
                    f"text_model.transformer.layers.{layer_index}.linear1.bias"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.layer_norm1.weight",
                    f"text_model.transformer.layers.{layer_index}.norm1.weight"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.layer_norm1.bias",
                    f"text_model.transformer.layers.{layer_index}.norm1.bias"
                ],
                # fc2
                [
                    f"text_model.encoder.layers.{layer_index}.mlp.fc2.weight",
                    f"text_model.transformer.layers.{layer_index}.linear2.weight",
                    "transpose"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.mlp.fc2.bias",
                    f"text_model.transformer.layers.{layer_index}.linear2.bias"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.layer_norm2.weight",
                    f"text_model.transformer.layers.{layer_index}.norm2.weight"
                ],
                [
                    f"text_model.encoder.layers.{layer_index}.layer_norm2.bias",
                    f"text_model.transformer.layers.{layer_index}.norm2.bias"
                ],
            ]
            vision_model_layer_mappings = [
                # qkv out
                [
                    f"vision_model.encoder.layers.{layer_index}.self_attn.q_proj.weight",
                    f"vision_model.transformer.layers.{layer_index}.self_attn.q_proj.weight",
                    "transpose"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.self_attn.q_proj.bias",
                    f"vision_model.transformer.layers.{layer_index}.self_attn.q_proj.bias",
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.self_attn.k_proj.weight",
                    f"vision_model.transformer.layers.{layer_index}.self_attn.k_proj.weight",
                    "transpose"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.self_attn.k_proj.bias",
                    f"vision_model.transformer.layers.{layer_index}.self_attn.k_proj.bias",
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.self_attn.v_proj.weight",
                    f"vision_model.transformer.layers.{layer_index}.self_attn.v_proj.weight",
                    "transpose"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.self_attn.v_proj.bias",
                    f"vision_model.transformer.layers.{layer_index}.self_attn.v_proj.bias",
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.self_attn.out_proj.weight",
                    f"vision_model.transformer.layers.{layer_index}.self_attn.out_proj.weight",
                    "transpose"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.self_attn.out_proj.bias",
                    f"vision_model.transformer.layers.{layer_index}.self_attn.out_proj.bias",
                ],
                # fc1
                [
                    f"vision_model.encoder.layers.{layer_index}.mlp.fc1.weight",
                    f"vision_model.transformer.layers.{layer_index}.linear1.weight",
                    "transpose"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.mlp.fc1.bias",
                    f"vision_model.transformer.layers.{layer_index}.linear1.bias"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.layer_norm1.weight",
                    f"vision_model.transformer.layers.{layer_index}.norm1.weight"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.layer_norm1.bias",
                    f"vision_model.transformer.layers.{layer_index}.norm1.bias"
                ],
                # fc2
                [
                    f"vision_model.encoder.layers.{layer_index}.mlp.fc2.weight",
                    f"vision_model.transformer.layers.{layer_index}.linear2.weight",
                    "transpose"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.mlp.fc2.bias",
                    f"vision_model.transformer.layers.{layer_index}.linear2.bias"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.layer_norm2.weight",
                    f"vision_model.transformer.layers.{layer_index}.norm2.weight"
                ],
                [
                    f"vision_model.encoder.layers.{layer_index}.layer_norm2.bias",
                    f"vision_model.transformer.layers.{layer_index}.norm2.bias"
                ],
            ]
            hard_mappings.extend(text_model_layer_mappings)
            hard_mappings.extend(vision_model_layer_mappings)

        mappings = [
            StateDictNameMapping(*mapping, index=index)
            for index, mapping in enumerate(hard_mappings)
        ]
        return mappings
