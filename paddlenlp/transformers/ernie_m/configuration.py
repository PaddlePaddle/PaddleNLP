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
""" ERNIE-M model configuration"""
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["ERNIE_M_PRETRAINED_INIT_CONFIGURATION", "ErnieMConfig", "ERNIE_M_PRETRAINED_RESOURCE_FILES_MAP"]

ERNIE_M_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-m-base": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 514,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 250002,
        "pad_token_id": 1,
    },
    "ernie-m-large": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "max_position_embeddings": 514,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "vocab_size": 250002,
        "pad_token_id": 1,
    },
    "uie-m-base": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 514,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 250002,
        "pad_token_id": 1,
    },
    "uie-m-large": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "max_position_embeddings": 514,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "vocab_size": 250002,
        "pad_token_id": 1,
    },
}

ERNIE_M_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "ernie-m-base": "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_m/ernie_m_base.pdparams",
        "ernie-m-large": "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_m/ernie_m_large.pdparams",
        "uie-m-base": "https://paddlenlp.bj.bcebos.com/models/transformers/uie_m/uie_m_base.pdparams",
        "uie-m-large": "https://paddlenlp.bj.bcebos.com/models/transformers/uie_m/uie_m_large.pdparams",
    }
}


class ErnieMConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a [`ErnieModel`]. It is used to
        instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
        configuration with the defaults will yield a similar configuration to that of the ERNIE
        ernie-3.0-medium-zh architecture.
        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.
    Args:
            vocab_size (int):
                Vocabulary size of `inputs_ids` in `ErnieMModel`. Also is the vocab size of token embedding matrix.
                Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ErnieMModel`.
            hidden_size (int, optional):
                Dimensionality of the embedding layer, encoder layers and pooler layer. Defaults to `768`.
            num_hidden_layers (int, optional):
                Number of hidden layers in the Transformer encoder. Defaults to `12`.
            num_attention_heads (int, optional):
                Number of attention heads for each attention layer in the Transformer encoder.
                Defaults to `12`.
            intermediate_size (int, optional):
                Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
                to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
                and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
                Defaults to `3072`.
            hidden_act (str, optional):
                The non-linear activation function in the feed-forward layer.
                ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
                are supported. Defaults to `"gelu"`.
            hidden_dropout_prob (float, optional):
                The dropout probability for all fully connected layers in the embeddings and encoder.
                Defaults to `0.1`.
            attention_probs_dropout_prob (float, optional):
                The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
                Defaults to `0.1`.
            max_position_embeddings (int, optional):
                The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
                sequence. Defaults to `512`.
            type_vocab_size (int, optional):
                The vocabulary size of the `token_type_ids`.
                Defaults to `2`.
            initializer_range (float, optional):
                The standard deviation of the normal initializer for initializing all weight matrices.
                Defaults to `0.02`.

                .. note::
                    A normal_initializer initializes weight matrices as normal distributions.
                    See :meth:`ErnieMPretrainedModel._init_weights()` for how weights are initialized in `ErnieMModel`.

            pad_token_id(int, optional):
                The index of padding token in the token vocabulary.
                Defaults to `1`.

        Examples:
        ```python
        >>> from paddlenlp.transformers import ErnieMModel, ErnieMConfig
        >>> # Initializing a configuration
        >>> configuration = ErnieMConfig()
        >>> # Initializing a model from the configuration
        >>> model = ErnieMModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```"""
    model_type = "ernie_m"
    attribute_map: Dict[str, str] = {"num_classes": "num_labels", "dropout": "classifier_dropout"}
    pretrained_init_configuration = ERNIE_M_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 250002,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 514,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
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
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
