# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" LayoutLM model configuration"""

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["LAYOUTLM_PRETRAINED_INIT_CONFIGURATION", "LayoutLMConfig", "LAYOUTLM_PRETRAINED_RESOURCE_FILES_MAP"]

LAYOUTLM_PRETRAINED_INIT_CONFIGURATION = {
    "layoutlm-base-uncased": {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "max_2d_position_embeddings": 1024,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
    },
    "layoutlm-large-uncased": {
        "vocab_size": 30522,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_2d_position_embeddings": 1024,
        "max_position_embeddings": 512,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
    },
}

LAYOUTLM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "layoutlm-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/layoutlm/layoutlm-base-uncased/model_state.pdparams",
        "layoutlm-large-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/layoutlm/layoutlm-large-uncased/model_state.pdparams",
    }
}


class LayoutLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`LayoutLMModel`]. It is used to instantiate an LayoutLM Model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the LayoutLM LayoutLM-base-uncased architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, optional, defaults to 30522):
            Vocabulary size of the LayoutLMModel model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method of [`LayoutLMModel`].
        embedding_size (`int`, optional, defaults to 768):
            Dimensionality of vocabulary embeddings.
        hidden_size (`int`, optional, defaults to 1024):
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
        max_2d_position_embeddings (`int`, optional, defaults to 1024):
            The maximum value that the 2D position embedding might ever used. Typically set this to something large just in case (e.g., 1024).
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
    >>> from paddlenlp.transformers import LayoutLMConfig, LayoutLMModel
    >>> # Initializing an LayoutLMConfig configuration
    >>> configuration = LayoutLMConfig()
    >>> # Initializing a model (with random weights) from the LayoutLM-base style configuration model
    >>> model = LayoutLMModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = LAYOUTLM_PRETRAINED_INIT_CONFIGURATION
    model_type = "layoutlm"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_2d_position_embeddings=1024,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout=0.1,
        pad_token_id=0,
        pool_act="tanh",
        **kwargs
    ):
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
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id
        self.pool_act = pool_act
