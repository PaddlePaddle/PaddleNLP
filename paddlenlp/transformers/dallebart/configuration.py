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
""" DalleBart model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["DALLEBART_PRETRAINED_INIT_CONFIGURATION", "DalleBartConfig", "DALLEBART_PRETRAINED_RESOURCE_FILES_MAP"]

DALLEBART_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "dalle-mini": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mini/model_state.pdparams",
        "dalle-mega-v16": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v16/model_state.pdparams",
        "dalle-mega-v26": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/model_state.pdparams",
        "dalle-mega": "https://bj.bcebos.com/paddlenlp/models/transformers/dallebart/dalle-mega-v26/model_state.pdparams",
    }
}

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


class DalleBartConfig(PretrainedConfig):
    r"""
    The bare DalleBart Model outputting raw hidden-states.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    Args:
        text_vocab_size (int):
            Vocabulary size of `inputs_ids` in `DalleBartModel`. Also is the vocab size of text token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `DalleBartModel`.
        image_vocab_size (int):
            Vocabulary size of `decoder_inputs_ids` in `DalleBartModel`. Also is the vocab size of image token embedding matrix.
            Defines the number of different tokens that can be represented by the `decoder_inputs_ids` passed when calling `DalleBartModel`.
        bos_token (int, optional):
            The beginning of image sequence token that was used during pretraining.
            Defaults to `16384`.
        pad_token_id(int, optional):
            The index of padding token in the image token vocabulary.
            Defaults to `16384`.
        eos_token (int, optional):
            A special token representing the end of a image sequence.
            Defaults to `16384`.
        max_text_length (int, optional):
            The maximum value of the dimensionality of text position encoding, which dictates the maximum supported length of the text
            input sequence. Defaults to `64`.
        max_image_length (int, optional):
            The maximum value of the dimensionality of image position encoding, which dictates the maximum supported length of the image
            input sequence. Defaults to `256`.
        decoder_start_token_id (int, optional):
            The id indicating the start of decoding image sentence. Defaults to `16384`.
        d_model (int, optional):
            Dimensionality of the embedding layer, encoder layer and decoder layer. Defaults to `1024`.
        num_encoder_layers (int, optional):
            Number of hidden layers in the :class:`DalleBartEncoder`. Defaults to `12`.
        num_decoder_layers (int, optional):
            Number of hidden layers in the :class:`DalleBartDecoder`. Defaults to `12`.
        encoder_attention_heads (int, optional):
            Number of attention heads for each attention layer in the :class:`DalleBartEncoder`.
            Defaults to `16`.
        decoder_attention_heads (int, optional):
            Number of attention heads for each attention layer in the :class:`DalleBartDecoder`.
            Defaults to `16`.
        encoder_ffn_dim (int, optional):
            Dimensionality of the Gated Linear Units (glu) layer in the encoder. Input tensors
            to glu layers are firstly projected from `d_model` to `encoder_ffn_dim`,
            and then projected back to `d_model`. Typically `encoder_ffn_dim` is larger than `d_model`.
            Defaults to `2730`.
        decoder_ffn_dim (int, optional):
            Dimensionality of the Gated Linear Units (glu) layer in the encoder. Input tensors
            to glu layers are firstly projected from `d_model` to `decoder_ffn_dim`,
            and then projected back to `d_model`. Typically `decoder_ffn_dim` is larger than `d_model`.
            Defaults to `2730`.
        dropout (float, optional):
            The dropout probability used in all fully connected layers (pre-process and post-process of MHA and FFN sub-layer)
            in the encoders and decoders. Defaults to `0.`.
        activation_function (str, optional):
            The non-linear activation function in the glu layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions are supported.
            Defaults to `"gelu"`.
        attention_dropout (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers and decoder layers to drop some attention target.
            Defaults to `0.`.
        activation_dropout (float, optional):
            The dropout probability used after glu activation in all encoder layers and decoder layers.
            Defaults to `0.`.
        use_bias (bool, optional):
            Whether or not use bias in all linear layers. Defaults to `False`.
        init_std (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Default to `0.02`.
    """
    pretrained_init_configuration = DALLEBART_PRETRAINED_INIT_CONFIGURATION
    model_type = "dallebart"
    attribute_map: Dict[str, str] = {
        "text_vocab_size": "vocab_size",
    }

    def __init__(
        self,
        vocab_size=50264,
        image_vocab_size=16384,
        bos_token_id=16384,
        pad_token_id=16384,
        eos_token_id=16384,
        max_text_length=64,
        max_image_length=256,
        decoder_start_token_id=16384,
        d_model=1024,
        num_encoder_layers=12,
        num_decoder_layers=12,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=2730,
        decoder_ffn_dim=2730,
        dropout=0.0,
        activation_function="gelu",
        attention_dropout=0.0,
        activation_dropout=0.0,
        use_bias=False,
        init_std=0.02,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.image_vocab_size = image_vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_text_length = max_text_length
        self.max_image_length = max_image_length
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.dropout = dropout
        self.activation_function = activation_function
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.use_bias = use_bias
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.text_pad_token_id = 1  # encoder pad id must be 1
