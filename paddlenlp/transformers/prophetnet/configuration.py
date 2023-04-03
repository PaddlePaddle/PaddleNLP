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
""" MBart model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "PROPHETNET_PRETRAINED_INIT_CONFIGURATION",
    "PROPHETNET_PRETRAINED_RESOURCE_FILES_MAP",
    "ProphetNetConfig",
]

PROPHETNET_PRETRAINED_INIT_CONFIGURATION = {
    "prophetnet-large-uncased": {
        "activation_dropout": 0.1,
        "activation_function": "gelu",
        "attention_dropout": 0.1,
        "bos_token_id": 102,
        "decoder_ffn_dim": 4096,
        "decoder_layerdrop": 0.0,
        "decoder_max_position_embeddings": 514,
        "decoder_start_token_id": 102,
        "disable_ngram_loss": False,
        "dropout": 0.1,
        "encoder_ffn_dim": 4096,
        "encoder_layerdrop": 0.0,
        "encoder_max_position_embeddings": 513,
        "eos_token_id": 102,
        "eps": 0.1,
        "hidden_size": 1024,
        "init_std": 0.02,
        "max_position_embeddings": 512,
        "ngram": 2,
        "num_buckets": 32,
        "num_decoder_attention_heads": 16,
        "num_decoder_layers": 12,
        "num_encoder_attention_heads": 16,
        "num_encoder_layers": 12,
        "pad_token_id": 0,
        "relative_max_distance": 128,
        "length_penalty": 2.0,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "max_length": 142,
        "vocab_size": 30522,
    },
}

PROPHETNET_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "prophetnet-large-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/prophetnet/prophetnet-large-uncased.pdparams"
    }
}


class ProphetNetConfig(PretrainedConfig):

    model_type = "prophetnet"

    def __init__(
        self,
        vocab_size=30522,
        bos_token_id=102,
        pad_token_id=0,
        eos_token_id=102,
        hidden_size=1024,
        decoder_start_token_id=102,
        max_position_embeddings=512,
        activation_function="gelu",
        activation_dropout=0.1,
        dropout=0.1,
        relative_max_distance=128,
        ngram=2,
        num_buckets=32,
        encoder_ffn_dim=4096,
        num_encoder_attention_heads=16,
        num_encoder_layers=12,
        decoder_ffn_dim=4096,
        num_decoder_attention_heads=16,
        num_decoder_layers=12,
        attention_dropout=0.1,
        init_std=0.02,
        eps=0.1,
        add_cross_attention=True,
        disable_ngram_loss=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.decoder_start_token_id = decoder_start_token_id
        self.max_position_embeddings = max_position_embeddings
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.dropout = dropout
        self.relative_max_distance = relative_max_distance
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_decoder_layers = num_decoder_layers
        self.attention_dropout = attention_dropout
        self.init_std = init_std
        self.eps = eps
        self.add_cross_attention = add_cross_attention
        self.disable_ngram_loss = disable_ngram_loss
