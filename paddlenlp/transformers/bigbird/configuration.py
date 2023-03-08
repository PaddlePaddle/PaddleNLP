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
""" BIGBIRD model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["BIGBIRD_PRETRAINED_INIT_CONFIGURATION", "BigBirdConfig", "BIGBIRD_PRETRAINED_RESOURCE_FILES_MAP"]

BIGBIRD_PRETRAINED_INIT_CONFIGURATION = {
    "bigbird-base-uncased": {
        "num_layers": 12,
        "vocab_size": 50358,
        "nhead": 12,
        "attn_dropout": 0.1,
        "dim_feedforward": 3072,
        "activation": "gelu",
        "normalize_before": False,
        "block_size": 16,
        "window_size": 3,
        "num_global_blocks": 2,
        "num_rand_blocks": 3,
        "seed": None,
        "pad_token_id": 0,
        "hidden_size": 768,
        "hidden_dropout_prob": 0.1,
        "max_position_embeddings": 4096,
        "type_vocab_size": 2,
        "num_labels": 2,
        "initializer_range": 0.02,
    },
}

BIGBIRD_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "bigbird-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/bigbird/bigbird-base-uncased.pdparams",
    }
}


class BigBirdConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BigBirdModel`]. It is used to instantiate an
    BigBird model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the BigBird
    [google/bigbird-roberta-base](https://huggingface.co/google/bigbird-roberta-base) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 50358):
            Vocabulary size of the BigBird model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BigBirdModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 1024 or 2048 or 4096).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BigBirdModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        attention_type (`str`, *optional*, defaults to `"bigbird"`)
            Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
            layer (with n^2 complexity). Possible values are `"original_full"` and `"bigbird"`.
        use_bias (`bool`, *optional*, defaults to `True`)
            Whether to use bias in query, key, value.
        rescale_embeddings (`bool`, *optional*, defaults to `False`)
            Whether to rescale embeddings with (hidden_size ** 0.5).
        block_size (`int`, *optional*, defaults to 64)
            Size of each block. Useful only when `attention_type == "bigbird"`.
        num_random_blocks (`int`, *optional*, defaults to 3)
            Each query is going to attend these many number of random blocks. Useful only when `attention_type ==
            "bigbird"`.
        dropout (`float`, *optional*):
            The dropout ratio for the classification head.
    Example:
    ```python
    >>> from transformers import BigBirdConfig, BigBirdModel
    >>> # Initializing a BigBird google/bigbird-roberta-base style configuration
    >>> configuration = BigBirdConfig()
    >>> # Initializing a model (with random weights) from the google/bigbird-roberta-base style configuration
    >>> model = BigBirdModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "big_bird"
    attribute_map: Dict[str, str] = {
        "num_classes": "num_labels",
        "nhead": "num_attention_heads",
        "num_layers": "num_hidden_layers",
        "dim_feedforward": "intermediate_size",
        "d_model": "hidden_size",
    }
    pretrained_init_configuration = BIGBIRD_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=50358,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=4096,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sep_token_id=66,
        attention_type="bigbird",
        use_bias=True,
        rescale_embeddings=False,
        block_size=1,
        num_random_blocks=3,
        dropout=0.1,
        padding_idx=0,
        attn_dropout=0.1,
        act_dropout=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
        window_size=3,
        num_global_blocks=2,
        num_rand_blocks=2,
        seed=None,
        activation="relu",
        embedding_weights=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache

        self.rescale_embeddings = rescale_embeddings
        self.attention_type = attention_type
        self.use_bias = use_bias
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.dropout = dropout

        self.padding_idx = padding_idx
        self.attn_dropout = attn_dropout
        self.act_dropout = act_dropout
        self.normalize_before = normalize_before
        self.weight_attr = weight_attr
        self.bias_attr = bias_attr
        self.window_size = window_size
        self.num_global_blocks = num_global_blocks
        self.num_rand_blocks = num_rand_blocks
        self.seed = seed
        self.activation = activation
        self.embedding_weights = embedding_weights
