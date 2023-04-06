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
""" ROFORMER model configuration"""
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["ROFORMER_PRETRAINED_INIT_CONFIGURATION", "RoFormerConfig", "ROFORMER_PRETRAINED_RESOURCE_FILES_MAP"]

ROFORMER_PRETRAINED_INIT_CONFIGURATION = {
    "roformer-chinese-small": {
        "vocab_size": 50000,
        "embedding_size": 384,
        "hidden_size": 384,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "rotary_value": False,
    },
    "roformer-chinese-base": {
        "vocab_size": 50000,
        "embedding_size": 768,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1536,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "rotary_value": False,
    },
    "roformer-chinese-char-small": {
        "vocab_size": 12000,
        "embedding_size": 384,
        "hidden_size": 384,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "rotary_value": False,
    },
    "roformer-chinese-char-base": {
        "vocab_size": 12000,
        "embedding_size": 768,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "rotary_value": False,
    },
    "roformer-chinese-sim-char-ft-small": {
        "vocab_size": 12000,
        "embedding_size": 384,
        "hidden_size": 384,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "eos_token_id": 102,
        "rotary_value": False,
        "pool_act": "linear",
    },
    "roformer-chinese-sim-char-ft-base": {
        "vocab_size": 12000,
        "embedding_size": 768,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "eos_token_id": 102,
        "rotary_value": False,
        "pool_act": "linear",
    },
    "roformer-chinese-sim-char-small": {
        "vocab_size": 12000,
        "embedding_size": 384,
        "hidden_size": 384,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "eos_token_id": 102,
        "rotary_value": False,
        "pool_act": "linear",
    },
    "roformer-chinese-sim-char-base": {
        "vocab_size": 12000,
        "embedding_size": 768,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "eos_token_id": 102,
        "rotary_value": False,
        "pool_act": "linear",
    },
    "roformer-english-small-discriminator": {
        "vocab_size": 30522,
        "embedding_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 12,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 128,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "rotary_value": True,
    },
    "roformer-english-small-generator": {
        "vocab_size": 30522,
        "embedding_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 12,
        "num_attention_heads": 1,
        "intermediate_size": 256,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 128,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "rotary_value": True,
    },
}

ROFORMER_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "roformer-chinese-small": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-small/model_state.pdparams",
        "roformer-chinese-base": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-base/model_state.pdparams",
        "roformer-chinese-char-small": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-char-small/model_state.pdparams",
        "roformer-chinese-char-base": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-char-base/model_state.pdparams",
        "roformer-chinese-sim-char-ft-small": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-sim-char-ft-small/model_state.pdparams",
        "roformer-chinese-sim-char-ft-base": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-sim-char-ft-base/model_state.pdparams",
        "roformer-chinese-sim-char-small": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-sim-char-small/model_state.pdparams",
        "roformer-chinese-sim-char-base": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-sim-char-base/model_state.pdparams",
        "roformer-english-small-discriminator": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-english-small-discriminator/model_state.pdparams",
        "roformer-english-small-generator": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-english-small-generator/model_state.pdparams",
    }
}


class RoFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RoFormerModel`]. It is used to
    instantiate a RoFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the RoFormer
    roformer-chinese-base architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the RoFormer model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RoFormer`].
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
        max_position_embeddings (`int`, *optional*, defaults to 1536):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 1536).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`RoFormerModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        pad_token_id (`int`, *optional*):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        eos_token_id (`int`, *optional*):
            The id of the `eos` token. Defaults to `102`.
        pool_act (`str`, *optional*):
            The non-linear activation function in the pooler.
            Defaults to `"tanh"`.
        rotary_value (`bool`, *optional*):
            Whether or not apply rotay position embeddings to value.
            Defaults to `False`.

    Examples:

    ```python
    >>> from paddlenlp.transformers import RoFormerModel, RoFormerConfig

    >>> # Initializing a RoFormer roformer-chinese-base style configuration
    >>> configuration = RoFormerConfig()

    >>> # Initializing a model from the roformer-chinese-base style configuration
    >>> model = RoFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "roformer"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = ROFORMER_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_size: int = 768,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1536,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        layer_norm_eps: float = 1e-12,
        rotary_value: bool = False,
        eos_token_id: int = 102,
        use_cache=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        if embedding_size is None:
            embedding_size = hidden_size
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
        self.pool_act = pool_act
        self.rotary_value = rotary_value

        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
