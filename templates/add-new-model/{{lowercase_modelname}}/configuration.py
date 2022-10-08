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
""" {{cookiecutter.model_name}} model configuration"""
from __future__ import annotations

from typing import Dict
from paddlenlp.transformers.configuration_utils import PretrainedConfig, attribute_map

__all__ = [
    "{{uppercase_modelname}}_PRETRAINED_INIT_CONFIGURATION", "{{cookiecutter.camelcase_modelname}}Config",
    "{{uppercase_modelname}}_PRETRAINED_RESOURCE_FILES_MAP"
]

{{uppercase_modelname}}_PRETRAINED_INIT_CONFIGURATION = {
}

{{uppercase_modelname}}_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
    }
}


class {{cookiecutter.camelcase_modelname}}Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`{{cookiecutter.camelcase_modelname}}Model`]. It is used to
    instantiate a {{uppercase_modelname}} model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the {{uppercase_modelname}}
    {{cookiecutter.checkpoint_identifier}} architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from paddlenlp.transformers import {{cookiecutter.camelcase_modelname}}Model, {{cookiecutter.camelcase_modelname}}Config

    >>> # Initializing a {{uppercase_modelname}} {{cookiecutter.checkpoint_identifier}} style configuration
    >>> configuration = {{cookiecutter.camelcase_modelname}}Config()

    >>> # Initializing a model from the {{cookiecutter.checkpoint_identifier}} style configuration
    >>> model = {{cookiecutter.camelcase_modelname}}Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "{{cookiecutter.lowercase_modelname}}"
    attribute_map: Dict[str, str] = {}
    pretrained_init_configuration = {{uppercase_modelname}}_PRETRAINED_INIT_CONFIGURATION

    def __init__(self,
                 vocab_size: int = 30522,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 16,
                 initializer_range: float = 0.02,
                 pad_token_id: int = 0,
                 pool_act: str = "tanh",
                 fuse: bool = False,
                 layer_norm_eps=1e-12,
                 use_cache=True,
                 **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse