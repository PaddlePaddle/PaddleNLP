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

from ..configuration_utils import PretrainedConfig

__all__ = ["GAUAlPHA_PRETRAINED_INIT_CONFIGURATION", "GAUAlphaConfig", "GAUAlPHA_PRETRAINED_RESOURCE_FILES_MAP"]

GAUAlPHA_PRETRAINED_INIT_CONFIGURATION = {
    "chinese_GAU-alpha-char_L-24_H-768": {
        "vocab_size": 12000,
        "hidden_size": 768,
        "intermediate_size": 1536,
        "num_hidden_layers": 24,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "attention_key_size": 128,
        "norm_eps": 1e-12,
        "pad_token_id": 0,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "swish",
        "use_bias": False,
        "normalization": "softmax_plus",
        "attention_scale": True,
    },
}

GAUAlPHA_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "chinese_GAU-alpha-char_L-24_H-768": "https://bj.bcebos.com/paddlenlp/models/transformers/gau_alpha/chinese_GAU-alpha-char_L-24_H-768/model_state.pdparams",
    }
}


class GAUAlphaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GAUAlphaModel`]. It is used to
    instantiate a GAUAlpha model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GAUAlpha
    chinese_GAU-alpha-char_L-24_H-768 architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the GAUAlpha model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GAUAlphaModel`].
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
            The vocabulary size of the `token_type_ids` passed when calling [`GAUAlphaModel`].
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
    >>> from paddlenlp.transformers import GAUAlphaModel, GAUAlphaConfig
    >>> # Initializing a GAUAlpha chinese_GAU-alpha-char_L-24_H-768style configuration
    >>> configuration = GAUAlphaConfig()
    >>> # Initializing a model from the  style configuration
    >>> model = GAUAlphaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "gau_alpha"
    pretrained_init_configuration = GAUAlPHA_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        task_id=0,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        task_type_vocab_size: int = 3,
        type_vocab_size: int = 16,
        attention_key_size=128,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        activation: str = "swish",
        normalization: str = "softmax_plus",
        fuse: bool = False,
        layer_norm_eps=1e-12,
        norm_eps=1e-12,
        use_cache=False,
        use_task_id=True,
        use_bias=False,
        enable_recompute=False,
        attention_scale=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.task_id = task_id
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.task_type_vocab_size = task_type_vocab_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse
        self.layer_norm_eps = layer_norm_eps
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.use_task_id = use_task_id
        self.enable_recompute = enable_recompute
        self.use_bias = use_bias
        self.activation = activation
        self.attention_key_size = attention_key_size
        self.normalization = normalization
        self.attention_scale = attention_scale
