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
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["DISTILBERT_PRETRAINED_INIT_CONFIGURATION", "DistilBertConfig", "DISTILBERT_PRETRAINED_RESOURCE_FILES_MAP"]

DISTILBERT_PRETRAINED_INIT_CONFIGURATION = {
    "distilbert-base-uncased": {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "distilbert-base-cased": {
        "vocab_size": 28996,
        "hidden_size": 768,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
}

DISTILBERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "distilbert-base-uncased": "http://bj.bcebos.com/paddlenlp/models/transformers/distilbert/distilbert-base-uncased.pdparams",
        "distilbert-base-cased": "http://bj.bcebos.com/paddlenlp/models/transformers/distilbert/distilbert-base-cased.pdparams",
    }
}


class DistilBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DistilBertModel`]. It is used to
    instantiate a DistilBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    bert-base-uncased architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be represented by the
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
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from paddlenlp.transformers import BertModel, BertConfig

    >>> # Initializing a DistilBERT distilbert-base-uncased style configuration
    >>> configuration = DistilBertConfig()

    >>> # Initializing a model from the distilbert-base-uncased style configuration
    >>> model = DistilBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "distilbert"
    attribute_map: Dict[str, str] = {
        "dropout": "classifier_dropout",
        "num_classes": "num_labels",
        "n_layers": "num_hidden_layers",  # for `transformers`
        "n_heads": "num_attention_heads",  # for `transformers`
        "dim": "hidden_size",  # for `transformers`
        "hidden_dim": "intermediate_size",  # for `transformers`
    }
    pretrained_init_configuration = DISTILBERT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        fuse: bool = False,
        layer_norm_eps=1e-12,
        use_cache=False,
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
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse

        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
