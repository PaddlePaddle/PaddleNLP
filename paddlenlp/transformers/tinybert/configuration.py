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
""" TinyBERT model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["TINYBERT_PRETRAINED_INIT_CONFIGURATION", "TinyBertConfig", "TINYBERT_PRETRAINED_RESOURCE_FILES_MAP"]

TINYBERT_PRETRAINED_INIT_CONFIGURATION = {
    "tinybert-4l-312d": {
        "vocab_size": 30522,
        "hidden_size": 312,
        "num_hidden_layers": 4,
        "num_attention_heads": 12,
        "intermediate_size": 1200,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "tinybert-6l-768d": {
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
    "tinybert-4l-312d-v2": {
        "vocab_size": 30522,
        "hidden_size": 312,
        "num_hidden_layers": 4,
        "num_attention_heads": 12,
        "intermediate_size": 1200,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "tinybert-6l-768d-v2": {
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
    "tinybert-4l-312d-zh": {
        "vocab_size": 21128,
        "hidden_size": 312,
        "num_hidden_layers": 4,
        "num_attention_heads": 12,
        "intermediate_size": 1200,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
    },
    "tinybert-6l-768d-zh": {
        "vocab_size": 21128,
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

TINYBERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "tinybert-4l-312d": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d.pdparams",
        "tinybert-6l-768d": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d.pdparams",
        "tinybert-4l-312d-v2": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-v2.pdparams",
        "tinybert-6l-768d-v2": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-v2.pdparams",
        "tinybert-4l-312d-zh": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-zh.pdparams",
        "tinybert-6l-768d-zh": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-zh.pdparams",
    }
}


class TinyBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TinyBertModel`]. It is used to
    instantiate a TinyBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TinyBERT
    tinybert-6l-768d-v2 architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


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
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        fit_size (int, optional):
            Dimensionality of the output layer of `fit_dense(s)`, which is the hidden size of the teacher model.
            `fit_dense(s)` means a hidden states' transformation from student to teacher.
            `fit_dense(s)` will be generated when bert model is distilled during the training, and will not be generated
            during the prediction process.
            `fit_denses` is used in v2 models and it has `num_hidden_layers+1` layers.
            `fit_dense` is used in other pretraining models and it has one linear layer.
            Defaults to `768`.

    Examples:

    ```python
    >>> from paddlenlp.transformers import TinyBertModel, TinyBertConfig

    >>> # Initializing a TinyBERT tinybert-6l-768d-v2 style configuration
    >>> configuration = TinyBertConfig()

    >>> # Initializing a model from the tinybert-6l-768d-v2 style configuration
    >>> model = TinyBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "tinybert"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = TINYBERT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        pool_act="tanh",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 16,
        layer_norm_eps=1e-12,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        fit_size: int = 768,
        **kwargs
    ):

        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.pool_act = pool_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.fit_size = fit_size
