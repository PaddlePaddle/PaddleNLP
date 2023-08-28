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
""" GPT model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["GPT_PRETRAINED_INIT_CONFIGURATION", "GPTConfig", "GPT_PRETRAINED_RESOURCE_FILES_MAP"]

GPT_PRETRAINED_INIT_CONFIGURATION = {
    "gpt-cpm-large-cn": {  # 2.6B
        "vocab_size": 30000,
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 10240,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "eos_token_id": 7,
        "bos_token_id": 0,
        "eol_token_id": 3,
    },
    "gpt-cpm-small-cn-distill": {  # 109M
        "vocab_size": 30000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "eos_token_id": 7,
        "bos_token_id": 0,
        "eol_token_id": 3,
    },
    "gpt3-13B-en": {  # 13B
        "vocab_size": 50304,
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_attention_heads": 128,
        "intermediate_size": 20480,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "eos_token_id": 50256,
        "eol_token_id": 198,
    },
    "gpt3-1.3B-en": {  # 1.3B
        "vocab_size": 50304,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 8192,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "eos_token_id": 50256,
        "eol_token_id": 198,
    },
    "gpt2-xl-en": {  # 1558M
        "vocab_size": 50257,
        "hidden_size": 1600,
        "num_hidden_layers": 48,
        "num_attention_heads": 25,
        "intermediate_size": 6400,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "eos_token_id": 50256,
        "eol_token_id": 198,
    },
    "gpt2-large-en": {  # 774M
        "vocab_size": 50257,
        "hidden_size": 1280,
        "num_hidden_layers": 36,
        "num_attention_heads": 20,
        "intermediate_size": 5120,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "eos_token_id": 50256,
        "eol_token_id": 198,
    },
    "gpt2-medium-en": {  # 345M
        "vocab_size": 50304,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "eos_token_id": 50256,
        "eol_token_id": 198,
    },
    "gpt2-en": {  # 117M
        "vocab_size": 50257,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "eos_token_id": 50256,
        "eol_token_id": 198,
    },
    "gpt2-small-en": {  # config for CE
        "vocab_size": 50304,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 1024,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "eos_token_id": 50256,
        "eol_token_id": 198,
    },
}

GPT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "gpt-cpm-large-cn": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt-cpm-large-cn.pdparams",
        "gpt-cpm-small-cn-distill": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt-cpm-small-cn-distill.pdparams",
        "gpt2-en": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-en.pdparams",
        "gpt2-medium-en": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-medium-en.pdparams",
        "gpt2-large-en": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-large-en.pdparams",
        "gpt2-xl-en": "https://bj.bcebos.com/paddlenlp/models/transformers/gpt/gpt2-xl-en.pdparams",
    }
}


class GPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTModel`] or a [`TFGPTModel`]. It is used to
    instantiate a GPT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPT
    gpt-base-uncased architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the GPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTModel`] or [`TFGPTModel`].
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
            The vocabulary size of the `token_type_ids` passed when calling [`GPTModel`] or [`TFGPTModel`].
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
    >>> from paddlenlp.transformers import GPTModel, GPTConfig

    >>> # Initializing a GPT gpt-base-uncased style configuration
    >>> configuration = GPTConfig()

    >>> # Initializing a model from the gpt-base-uncased style configuration
    >>> model = GPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "gpt"
    attribute_map: Dict[str, str] = {
        "num_classes": "num_labels",
        "dropout": "classifier_dropout",
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
        "resid_pdrop": "attention_probs_dropout_prob",
    }

    pretrained_init_configuration = GPT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        fuse_attention_qkv: bool = False,
        use_flash_attention: bool = False,
        pad_token_id: int = 0,
        eos_token_id: int = 7,
        bos_token_id: int = 0,
        eol_token_id: int = 3,
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
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.eol_token_id = eol_token_id

        self.fuse_attention_qkv = fuse_attention_qkv
        self.use_flash_attention = use_flash_attention
