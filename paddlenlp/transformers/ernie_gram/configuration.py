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
""" Ernie Doc model configuration"""
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["ERNIE_GRAM_PRETRAINED_INIT_CONFIGURATION", "ErnieGramConfig", "ERNIE_GRAM_PRETRAINED_RESOURCE_FILES_MAP"]

ERNIE_GRAM_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-gram-zh": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 18018,
    },
    "ernie-gram-zh-finetuned-dureader-robust": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 18018,
    },
}

ERNIE_GRAM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "ernie-gram-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_gram_zh/ernie_gram_zh.pdparams",
        "ernie-gram-zh-finetuned-dureader-robust": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-gram-zh-finetuned-dureader-robust/model_state.pdparams",
    },
}


class ErnieGramConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieGramModel`]. It is used to instantiate
    an ErnieGram model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (int):
            Vocabulary size of the ERNIE-Gram model. Also is the vocab size of token embedding matrix.
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
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoders.
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
                See :meth:`ErniePretrainedModel._init_weights()` for how weights are initialized in `ErnieGramModel`.

        rel_pos_size (int, optional):
            The relative position size just for ERNIE-Gram English model. Defaults to None.
        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
    Example:
    ```python
    >>> from paddlenlp.transformers import ErnieGramConfig, ErnieGramModel
    >>> # Initializing an ErnieGram style configuration
    >>> configuration = ErnieGramConfig()
    >>> # Initializing a model (with random weights) from the ErnieGram-base style configuration
    >>> model = ErnieGramModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = ERNIE_GRAM_PRETRAINED_INIT_CONFIGURATION
    model_type = "ernie_gram"

    def __init__(
        self,
        vocab_size=18018,
        embedding_size=768,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        rel_pos_size=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
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
        self.rel_pos_size = rel_pos_size
