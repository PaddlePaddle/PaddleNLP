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
""" ConvBERT model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["CONVBERT_PRETRAINED_INIT_CONFIGURATION", "ConvBertConfig", "CONVBERT_PRETRAINED_RESOURCE_FILES_MAP"]

CONVBERT_PRETRAINED_INIT_CONFIGURATION = {
    "convbert-base": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 1,
    },
    "convbert-medium-small": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "initializer_range": 0.02,
        "intermediate_size": 1536,
        "max_position_embeddings": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 2,
    },
    "convbert-small": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 1,
    },
    "convbert-base-generator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 1,
    },
    "convbert-medium-small-generator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 96,
        "initializer_range": 0.02,
        "intermediate_size": 384,
        "max_position_embeddings": 512,
        "num_attention_heads": 2,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 2,
    },
    "convbert-small-generator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 256,
        "max_position_embeddings": 512,
        "num_attention_heads": 1,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 1,
    },
    "convbert-base-discriminator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 1,
    },
    "convbert-medium-small-discriminator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 384,
        "initializer_range": 0.02,
        "intermediate_size": 1536,
        "max_position_embeddings": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 2,
    },
    "convbert-small-discriminator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "conv_kernel_size": 9,
        "head_ratio": 2,
        "num_groups": 1,
    },
}

CONVBERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "convbert-base": "http://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-base/model_state.pdparams",
        "convbert-medium-small": "http://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-medium-small/model_state.pdparams",
        "convbert-small": "http://bj.bcebos.com/paddlenlp/models/transformers/convbert/convbert-small/model_state.pdparams",
    }
}


class ConvBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConvBertModel`]. It is used to instantiate a
    ConvBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ConvBert
    convbert-base architecture. Configuration objects.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    ======================================================
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        pool_act (`str`, *optional*):
            The non-linear activation function in the pooler.
            Defaults to `"tanh"`.
        embedding_size (int, optional):
            Dimensionality of the embedding layer. Defaults to `768`.
        conv_kernel_size (int, optional):
            The size of the convolutional kernel.
            Defaults to `9`.
        head_ratio (int, optional):
            Ratio gamma to reduce the number of attention heads.
            Defaults to `2`.
        num_groups (int, optional):
            The number of groups for grouped linear layers for ConvBert model.
            Defaults to `1`.

    Examples:

    ```python
    >>> from paddlenlp.transformers import ConvBertModel, ConvBertConfig

    >>> # Initializing a ConvBERT configuration
    >>> configuration = ConvBertConfig()

    >>> # Initializing a model from the ConvBERT-base style configuration model
    >>> model = ConvBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ======================================================
    ```"""
    model_type = "convbert"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = CONVBERT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        embedding_size: int = 768,
        conv_kernel_size: int = 9,
        head_ratio: int = 2,
        num_groups: int = 1,
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
        self.pool_act = pool_act
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size
        self.conv_kernel_size = conv_kernel_size
        self.head_ratio = head_ratio
        self.num_groups = num_groups
