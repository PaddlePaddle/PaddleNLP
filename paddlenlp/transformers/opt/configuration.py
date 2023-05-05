# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

""" OPT Model Configuration"""
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = [
    "OPT_PRETRAINED_INIT_CONFIGURATION",
    "OPT_PRETRAINED_RESOURCE_FILES_MAP",
    "OPTConfig",
]

OPT_PRETRAINED_INIT_CONFIGURATION = {
    "facebook/opt-1.3b": {
        "init_args": [
            {
                "intermediate_size": 8192,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.1,
                "normalize_before": True,
                "word_embed_proj_dim": 2048,
                "num_attention_heads": 32,
                "bos_token_id": 2,
                "hidden_size": 2048,
                "eos_token_id": 2,
                "hidden_act": "relu",
                "initializer_range": 0.02,
                "max_position_embeddings": 2048,
                "num_hidden_layers": 24,
                "pad_token_id": 1,
                "vocab_size": 50272,
                "type_vocab_size": 16,
                "init_class": "OPTModel",
            }
        ],
        "init_class": "OPTForCausalLM",
    },
}

OPT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "facebook/opt-1.3b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/opt-1.3b/model_state.pdparams"
    }
}


class OPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OPTModel`]. It is used to instantiate
    an OPT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OPT
    [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

     Args:
        vocab_size (`int`, *optional*, defaults to 50272):
            Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OPTModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether to perform layer normalization before the attention block.
        word_embed_proj_dim (`int`, *optional*):
            `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-1.3b`. Defaults to
            `hidden_size`.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`. Defaults to `16`.
            .. note::
                Please NOT using `type_vocab_size`, for it will be obsolete in the future..
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Default to `0.02`.
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`OPTPretrainedModel._init_weights()` for how weights are initialized in `OPTModel`.

    Example:
    ```python
    >>> from paddlenlp.transformers import OPTModel, OPTConfig
    >>> # Initializing a OPT facebook/opt-1.3b style configuration
    >>> config = OPTConfig()
    >>> # Initializing a model from the facebook/opt-1.3b style configuration
    >>> model = OPTModel(config)
    >>> # Accessing the model config
    >>> config = model.config
    ```"""

    attribute_map: Dict[str, str] = {
        "dropout": "classifier_dropout",
        "num_classes": "num_labels",
        "ffn_dim": "intermediate_size",
        "activation_function": "hidden_act",
    }
    pretrained_init_configuration = OPT_PRETRAINED_INIT_CONFIGURATION
    model_type = "opt"

    def __init__(
        self,
        vocab_size=50272,
        hidden_size=2048,
        num_hidden_layers=24,
        intermediate_size=8192,
        num_attention_heads=32,
        hidden_act="relu",
        max_position_embeddings=2048,
        normalize_before=True,
        word_embed_proj_dim=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        type_vocab_size=16,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        enable_bias: bool = True,
        mp_degree: int = 1,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.normalize_before = normalize_before
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size

        self.enable_bias = enable_bias
        self.mp_degree = mp_degree
