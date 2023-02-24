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

""" SKEP model configuration """
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["SKEP_PRETRAINED_INIT_CONFIGURATION", "SKEP_PRETRAINED_RESOURCE_FILES_MAP", "SkepConfig"]

SKEP_PRETRAINED_INIT_CONFIGURATION = {
    "skep_ernie_1.0_large_ch": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 4,
        "vocab_size": 12800,
        "pad_token_id": 0,
    },
    "skep_ernie_2.0_large_en": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 4,
        "vocab_size": 30522,
        "pad_token_id": 0,
    },
    "skep_roberta_large_en": {
        "attention_probs_dropout_prob": 0.1,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "max_position_embeddings": 514,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 0,
        "vocab_size": 50265,
        "pad_token_id": 1,
    },
}

SKEP_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "skep_ernie_1.0_large_ch": "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_ernie_1.0_large_ch.pdparams",
        "skep_ernie_2.0_large_en": "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_ernie_2.0_large_en.pdparams",
        "skep_roberta_large_en": "https://bj.bcebos.com/paddlenlp/models/transformers/skep/skep_roberta_large_en.pdparams",
    }
}


class SkepConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`SKEPModel`]. It is used to instantiate an SKEP Model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the SKEP skep_ernie_1.0_large_ch architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, optional, defaults to 12800): Vocabulary size of the SKEP model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [`SKEPModel`].
        hidden_size (`int`, optional, defaults to 768): Dimensionality of the embedding layer, encoder layers and the pooler layer.
        num_hidden_layers (int, optional, defaults to 12): Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, optional, defaults to 12):  Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, optional, defaults to 3072): Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors to ff layers are firstly projected from `hidden_size` to `intermediate_size`, and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
        hidden_act (`str`, optional, defaults to "relu"):The non-linear activation function in the encoder and pooler. "gelu", "relu" and any other paddle supported activation functions are supported.
        hidden_dropout_prob (`float`, optional, defaults to 0.1): The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, optional, defaults to 0.1): The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
        max_position_embeddings (`int`, optional, defaults to 512): The maximum sequence length that this model might ever be used with. Typically set this to something large (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, optional, defaults to 4): The vocabulary size of the *token_type_ids* passed into [`SKEPModel`].
        initializer_range (`float`, optional, defaults to 0.02): The standard deviation of the normal initializer.
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`SkepPretrainedModel.init_weights()` for how weights are initialized in [`SkepModel`].
        pad_token_id(int, optional, defaults to 0): The index of padding token in the token vocabulary.
    Examples:
    ```python
    >>> from paddlenlp.transformers import SKEPModel, SkepConfig
    >>> # Initializing an SKEP configuration
    >>> configuration = SkepConfig()
    >>> # Initializing a model (with random weights) from the SKEP-base style configuration model
    >>> model = SKEPModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = SKEP_PRETRAINED_INIT_CONFIGURATION
    model_type = "skep"

    def __init__(
        self,
        vocab_size=12800,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=4,
        initializer_range=0.02,
        pad_token_id=0,
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
        self.pad_token_id = pad_token_id
