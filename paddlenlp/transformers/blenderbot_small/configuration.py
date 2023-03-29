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
""" blenderbot model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "BLENDERBOTSMALL_PRETRAINED_INIT_CONFIGURATION",
    "BlenderbotSmallConfig",
    "BLENDERBOTSMALL_PRETRAINED_RESOURCE_FILES_MAP",
]

BLENDERBOTSMALL_PRETRAINED_INIT_CONFIGURATION = {
    "blenderbot_small-90M": {
        "vocab_size": 54944,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "decoder_start_token_id": 1,
        "d_model": 512,
        "num_encoder_layers": 8,
        "num_decoder_layers": 8,
        "encoder_attention_heads": 16,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": 2048,
        "encoder_ffn_dim": 2048,
        "dropout": 0.1,
        "activation_function": "gelu",
        "init_std": 0.02,
        "max_position_embeddings": 512,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "scale_embedding": True,
        "normalize_before": False,
    },
}

BLENDERBOTSMALL_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "blenderbot_small-90M": "https://bj.bcebos.com/paddlenlp/models/transformers/blenderbot_small/blenderbot_small-90M.pdparams",
    }
}


class BlenderbotSmallConfig(PretrainedConfig):
    """
    Args:
         vocab_size (`int`):
             Vocabulary size of the BlenderbotSmall model.
         bos_token_id (`int`, optional):
            The id for begging of sentences token. Defaults to ``1``.
         pad_token_id (`int`, optional):
            The id for padding token. Defaults to ``0``.
         eos_token_id (`int`, optional):
            The id for end of sentence token. Defaults to ``2``.
         decoder_start_token_id (`int`, optional):
            The id indicating the start of decoding sentence. Defaults to ``1``.
         d_model (`int`, optional):
            Dimensionality of the layers and the pooler layer. Defaults to ``512``.
         num_encoder_layers (`int`, optional):
            Number of Transformer encoder layers for BlenderbotSmallEncoder. Defaults to ``8``.
         num_decoder_layers (`int`, optional):
            Number of Transformer decoder layers for BlenderbotSmallDecoder. Defaults to ``8``.
         encoder_attention_heads (`int`, optional):
            Number of attention heads for each Transformer encoder layer in BlenderbotSmallEncoder.
            Defaults to ``16``.
         decoder_attention_heads (`int`, optional):
            Number of attention heads for each Transformer decoder layer in BlenderbotSmallDecoder.
            Defaults to ``16``.
         encoder_ffn_dim (`int`, optional):
            Dimensionality of the feed-forward layer for each Transformer encoder layer in
            BlenderbotSmallEncoder. Defaults to ``2048``.
         decoder_ffn_dim (`int`, optional):
            Dimensionality of the feed-forward layer for each Transformer dncoder layer in
            BlenderbotSmallDncoder. Defaults to ``2048``.
         dropout (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            Defaults to ``0.1``.
         activation_function (`str`, optional):
            The non-linear activation function (function or string) in the encoder and pooler.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
         attention_dropout (`float`, optional):
            The dropout ratio for the attention probabilities.
            Defaults to ``0.0``.
         activation_dropout (`float`, optional):
            The dropout ratio for activations inside the fully connected layer.
         max_position_embeddings (`int`, optional):,
            The max position index of an input sequence. Defaults to ``512``.
         init_std (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
         scale_embedding (`bool`, optional):
            Indicate whether to scale embeddings by diving by sqrt(d_model). Defaults to ``True``.
         normalize_before (bool, optional):
            Indicate whether to put layer normalization into preprocessing of MHA and FFN sub-layers.
            If True, pre-process is layer normalization and post-precess includes dropout,
            residual connection. Otherwise, no pre-process and post-precess includes dropout,
            residual connection, layer normalization. Defaults to ``False``.
    """

    model_type = "blenderbot_small"
    pretrained_init_configuration = BLENDERBOTSMALL_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=54944,
        bos_token_id=1,
        pad_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=1,
        d_model=512,
        num_encoder_layers=8,
        num_decoder_layers=8,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        dropout=0.1,
        activation_function="gelu",
        attention_dropout=0.0,
        activation_dropout=0.0,
        max_position_embeddings=512,
        init_std=0.02,
        scale_embedding=True,
        normalize_before=False,
        **kwargs
    ):
        super(BlenderbotSmallConfig, self).__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.activation_function = activation_function
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.normalize_before = normalize_before
