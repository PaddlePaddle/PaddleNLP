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
""" BERT model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["DALLEBART_PRETRAINED_INIT_CONFIGURATION", "DalleBartConfig", "DALLEBART_PRETRAINED_RESOURCE_FILES_MAP"]

DALLEBART_PRETRAINED_INIT_CONFIGURATION = {
    "dalle-mini": {
            "text_vocab_size": 50264,
            "image_vocab_size": 16384,
            "bos_token_id": 16384,
            "pad_token_id": 16384,
            "eos_token_id": 16384,
            "max_text_length": 64,
            "max_image_length": 256,
            "decoder_start_token_id": 16384,
            "d_model": 1024,
            "num_encoder_layers": 12,
            "num_decoder_layers": 12,
            "encoder_attention_heads": 16,
            "decoder_attention_heads": 16,
            "encoder_ffn_dim": 2730,
            "decoder_ffn_dim": 2730,
            "dropout": 0.0,
            "activation_function": "gelu",
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "use_bias": False,
            "init_std": 0.02,
        },
        "dalle-mega-v16": {
            "text_vocab_size": 50272,
            "image_vocab_size": 16415,
            "bos_token_id": 16384,
            "pad_token_id": 16384,
            "eos_token_id": 16384,
            "max_text_length": 64,
            "max_image_length": 256,
            "decoder_start_token_id": 16384,
            "d_model": 2048,
            "num_encoder_layers": 24,
            "num_decoder_layers": 24,
            "encoder_attention_heads": 32,
            "decoder_attention_heads": 32,
            "encoder_ffn_dim": 4096,
            "decoder_ffn_dim": 4096,
            "dropout": 0.0,
            "activation_function": "gelu",
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "use_bias": False,
            "init_std": 0.02,
        },
        "dalle-mega-v26": {
            "text_vocab_size": 50272,
            "image_vocab_size": 16415,
            "bos_token_id": 16384,
            "pad_token_id": 16384,
            "eos_token_id": 16384,
            "max_text_length": 64,
            "max_image_length": 256,
            "decoder_start_token_id": 16384,
            "d_model": 2048,
            "num_encoder_layers": 24,
            "num_decoder_layers": 24,
            "encoder_attention_heads": 32,
            "decoder_attention_heads": 32,
            "encoder_ffn_dim": 4096,
            "decoder_ffn_dim": 4096,
            "dropout": 0.0,
            "activation_function": "gelu",
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "use_bias": False,
            "init_std": 0.02,
        },
        "dalle-mega": {
            "text_vocab_size": 50272,
            "image_vocab_size": 16415,
            "bos_token_id": 16384,
            "pad_token_id": 16384,
            "eos_token_id": 16384,
            "max_text_length": 64,
            "max_image_length": 256,
            "decoder_start_token_id": 16384,
            "d_model": 2048,
            "num_encoder_layers": 24,
            "num_decoder_layers": 24,
            "encoder_attention_heads": 32,
            "decoder_attention_heads": 32,
            "encoder_ffn_dim": 4096,
            "decoder_ffn_dim": 4096,
            "dropout": 0.0,
            "activation_function": "gelu",
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "use_bias": False,
            "init_std": 0.02,
        },
    }

DALLEBART_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "dalle-mini": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mini/model_state.pdparams",
         "dalle-mega-v16": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v16/model_state.pdparams",
          "dalle-mega-v26": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/model_state.pdparams",
            "dalle-mega": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/model_state.pdparams",    }
}


class DalleBartConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    bert-base-uncased architecture.

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

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = BertConfig()

    >>> # Initializing a model from the bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "dallebart"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = DALLEBART_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        text_vocab_size: int = 50264,
        image_voacb_size: int = 16384,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        bos_token_id: int = 16384,
        pad_token_id: int = 16384,
        eos_token_id: int = 16384,
        max_text_length: int = 64,
        max_image_length: int = 256,
        decoder_start_token_id: int = 16384,
        d_model: int = 1024,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        encoder_attention_heads: int = 16,
        decoder_attention_heads: int = 16,
        encoder_ffn_dim: int = 2730,
        decoder_ffn_dim: int = 2730,
        dropout: int = 0.0,
        activation_function: str = "gelu",
        use_bias: bool =  False,
        init_std: float = 0.02,      
        fuse: bool = False,
        layer_norm_eps=1e-12,
        use_cache=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.max_text_length = max_text_length
        self.max_image_length = max_image_length
        self.decoder_start_token_id = decoder_start_token_id
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.dropout = dropout
        self.activation_function = activation_function
          
        
        self.fuse = fuse

        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
