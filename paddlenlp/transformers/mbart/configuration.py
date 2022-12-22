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
""" MBart model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["MBART_PRETRAINED_INIT_CONFIGURATION", "MBartConfig", "MBART_PRETRAINED_RESOURCE_FILES_MAP"]

MBART_PRETRAINED_INIT_CONFIGURATION = {
    "mbart-large-cc25": {
        "vocab_size": 250027,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "d_model": 1024,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "encoder_attention_heads": 16,
        "decoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "decoder_ffn_dim": 4096,
        "dropout": 0.1,
        "activation_function": "gelu",
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "max_position_embeddings": 1024,
        "init_std": 0.02,
        "scale_embedding": True,
    },
    "mbart-large-en-ro": {
        "vocab_size": 250027,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 250020,
        "d_model": 1024,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "encoder_attention_heads": 16,
        "decoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "decoder_ffn_dim": 4096,
        "dropout": 0.1,
        "activation_function": "gelu",
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "max_position_embeddings": 1024,
        "init_std": 0.02,
        "scale_embedding": True,
    },
    "mbart-large-50-one-to-many-mmt": {
        "vocab_size": 250054,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 2,
        "d_model": 1024,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "encoder_attention_heads": 16,
        "decoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "decoder_ffn_dim": 4096,
        "dropout": 0.1,
        "activation_function": "relu",
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "max_position_embeddings": 1024,
        "init_std": 0.02,
        "scale_embedding": True,
    },
    "mbart-large-50-many-to-one-mmt": {
        "vocab_size": 250054,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 2,
        "forced_bos_token_id": 250004,
        "d_model": 1024,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "encoder_attention_heads": 16,
        "decoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "decoder_ffn_dim": 4096,
        "dropout": 0.1,
        "activation_function": "relu",
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "max_position_embeddings": 1024,
        "init_std": 0.02,
        "scale_embedding": True,
    },
    "mbart-large-50-many-to-many-mmt": {
        "vocab_size": 250054,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 2,
        "d_model": 1024,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "encoder_attention_heads": 16,
        "decoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "decoder_ffn_dim": 4096,
        "dropout": 0.1,
        "activation_function": "relu",
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "max_position_embeddings": 1024,
        "init_std": 0.02,
        "scale_embedding": True,
    },
}

MBART_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "mbart-large-cc25": "https://bj.bcebos.com/paddlenlp/models/transformers/mbart/mbart-large-cc25.pdparams",
        "mbart-large-en-ro": "https://bj.bcebos.com/paddlenlp/models/transformers/mbart/mbart-large-en-ro.pdparams",
        "mbart-large-50-one-to-many-mmt": "https://bj.bcebos.com/paddlenlp/models/transformers/mbart50/mbart-large-50-one-to-many-mmt.pdparams",
        "mbart-large-50-many-to-one-mmt": "https://bj.bcebos.com/paddlenlp/models/transformers/mbart50/mbart-large-50-many-to-one-mmt.pdparams",
        "mbart-large-50-many-to-many-mmt": "https://bj.bcebos.com/paddlenlp/models/transformers/mbart50/mbart-large-50-many-to-many-mmt.pdparams",
    }
}


class MBartConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MBartModel`]. It is used to instantiate a MBART
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MBART mbart-large-cc25 architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `MBartModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `MBartModel`.
            Defaults to 50265.
        bos_token (int, optional):
            The beginning of sequence token that was used during pretraining. Can be
            used a sequence classifier token.
            Defaults to `0`.
        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `1`.
        eos_token (int, optional):
            A special token representing the end of a sequence that was used during pretraining.
            Defaults to `2`.
        d_model (int, optional):
            Dimensionality of the embedding layer, encoder layer and decoder layer. Defaults to `768`.
        num_encoder_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `6`.
        num_decoder_layers (int, optional):
            Number of hidden layers in the Transformer decoder. Defaults to `6`.
        encoder_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        decoder_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer decoder.
            Defaults to `12`.
        encoder_ffn_dim (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `d_model` to `encoder_ffn_dim`,
            and then projected back to `d_model`. Typically `encoder_ffn_dim` is larger than `d_model`.
            Defaults to `3072`.
        decoder_ffn_dim (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `d_model` to `decoder_ffn_dim`,
            and then projected back to `d_model`. Typically `decoder_ffn_dim` is larger than `d_model`.
            Defaults to `3072`.
        dropout (float, optional):
            The dropout probability used in all fully connected layers (pre-process and post-process of MHA and FFN sub-layer)
            in the encoders and decoders. Defaults to `0.1`.
        activation_function (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions are supported.
            Defaults to `"gelu"`.
        attention_dropout (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers and decoder layers to drop some attention target.
            Defaults to `0.1`.
        activation_dropout (float, optional):
            The dropout probability used after FFN activation in all encoder layers and decoder layers.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `1024`.
        init_std (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Default to `0.02`.
        num_labels (`int`, optional):
            The number of labels to use in [`BartForSequenceClassification`]. Defaults to 3.
        forced_eos_token_id (`int`, optional):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`. Defaults to 2.
        scale_embedding (`bool`, optional):
            Scale embeddings by diving by sqrt(d_model). Default to `True`.

    """
    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    standard_config_map: Dict[str, str] = {
        "num_encoder_layers": "encoder_layers",
        "num_decoder_layers": "decoder_layers",
    }
    pretrained_init_configuration = MBART_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 50265,
        bos_token_id: int = 0,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        forced_eos_token_id: int = 2,
        d_model: int = 768,
        encoder_layers: int = 12,
        decoder_layers: int = 12,
        encoder_attention_heads: int = 16,
        decoder_attention_heads: int = 16,
        encoder_ffn_dim: int = 4096,
        decoder_ffn_dim: int = 4096,
        dropout: float = 0.1,
        activation_function: str = "gelu",
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        max_position_embeddings: int = 1024,
        init_std: float = 0.02,
        is_encoder_decoder: bool = True,
        scale_embedding: bool = True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.scale_embedding = scale_embedding
