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
""" NeZha model configuration"""
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["NEZHA_PRETRAINED_INIT_CONFIGURATION", "NeZhaConfig", "NEZHA_PRETRAINED_RESOURCE_FILES_MAP"]

NEZHA_PRETRAINED_INIT_CONFIGURATION = {
    "nezha-base-chinese": {
        "vocab_size": 21128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "max_relative_position": 64,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "use_relative_position": True,
    },
    "nezha-large-chinese": {
        "vocab_size": 21128,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "max_relative_position": 64,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "use_relative_position": True,
    },
    "nezha-base-wwm-chinese": {
        "vocab_size": 21128,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "max_relative_position": 64,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "use_relative_position": True,
    },
    "nezha-large-wwm-chinese": {
        "vocab_size": 21128,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "max_relative_position": 64,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "use_relative_position": True,
    },
}
NEZHA_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "nezha-base-chinese": "https://bj.bcebos.com/paddlenlp/models/transformers/nezha/nezha-base-chinese.pdparams",
        "nezha-large-chinese": "https://bj.bcebos.com/paddlenlp/models/transformers/nezha/nezha-large-chinese.pdparams",
        "nezha-base-wwm-chinese": "https://bj.bcebos.com/paddlenlp/models/transformers/nezha/nezha-base-wwm-chinese.pdparams",
        "nezha-large-wwm-chinese": "https://bj.bcebos.com/paddlenlp/models/transformers/nezha/nezha-large-wwm-chinese.pdparams",
    }
}


class NeZhaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`NezhaModel`]. It is used to instantiate an Nezha
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Nezha
    [sijunhe/nezha-cn-base](https://huggingface.co/sijunhe/nezha-cn-base) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, optional, defaults to 21128):
            Vocabulary size of the NEZHA model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method of [`NezhaModel`].
        embedding_size (`int`, optional, defaults to 128):
            Dimensionality of vocabulary embeddings.
        hidden_size (`int`, optional, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, optional, defaults to 3072):
            The dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, optional, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_dropout_prob (`float`, optional, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, optional, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, optional, defaults to 2):
            The vocabulary size of the *token_type_ids* passed into [`NezhaModel`].
        initializer_range (`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout (`float`, optional, defaults to 0.1):
            The dropout ratio for attached classifiers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    Example:
    ```python
    >>> from paddlenlp.transformers import NeZhaConfig, NeZhaModel
    >>> # Initializing an Nezha configuration
    >>> configuration = NeZhaConfig()
    >>> # Initializing a model (with random weights) from the Nezha-base style configuration model
    >>> model = NeZhaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = NEZHA_PRETRAINED_INIT_CONFIGURATION
    model_type = "nezha"

    def __init__(
        self,
        vocab_size=21128,
        embedding_size=128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_relative_position=64,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout=0.1,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        use_cache=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_position = max_relative_position
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
