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

""" Electra model configuration """
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["ElectraConfig", "ELECTRA_PRETRAINED_INIT_CONFIGURATION", "ELECTRA_PRETRAINED_RESOURCE_FILES_MAP"]

ELECTRA_PRETRAINED_INIT_CONFIGURATION = {
    "electra-small": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "electra-base": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "electra-large": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 1024,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "chinese-electra-small": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128,
    },
    "chinese-electra-base": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128,
    },
    "ernie-health-chinese": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 22608,
        "layer_norm_eps": 1e-5,
    },
    "electra-small-generator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "electra-base-generator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "electra-large-generator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 1024,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "electra-small-discriminator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 128,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "electra-base-discriminator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "electra-large-discriminator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 1024,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    },
    "ernie-health-chinese-generator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 22608,
        "layer_norm_eps": 1e-12,
    },
    "ernie-health-chinese-discriminator": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 22608,
        "layer_norm_eps": 1e-12,
    },
}

ELECTRA_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "electra-small": "https://bj.bcebos.com/paddlenlp/models/transformers/electra/electra-small.pdparams",
        "electra-base": "https://bj.bcebos.com/paddlenlp/models/transformers/electra/electra-base.pdparams",
        "electra-large": "https://bj.bcebos.com/paddlenlp/models/transformers/electra/electra-large.pdparams",
        "chinese-electra-small": "https://bj.bcebos.com/paddlenlp/models/transformers/chinese-electra-small/chinese-electra-small.pdparams",
        "chinese-electra-base": "https://bj.bcebos.com/paddlenlp/models/transformers/chinese-electra-base/chinese-electra-base.pdparams",
        "ernie-health-chinese": "https://paddlenlp.bj.bcebos.com/models/transformers/ernie-health-chinese/ernie-health-chinese.pdparams",
    }
}


class ElectraConfig(PretrainedConfig):
    model_type = "electra"
    pretrained_init_configuration = ELECTRA_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 22608,
        embedding_size: int = 768,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        layer_norm_eps: float = 1e-12,
        num_choices: int = 2,
        gen_weight: float = 1.0,
        disc_weight: float = 50.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
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
        self.pad_token_id = pad_token_id
        self.layer_norm_eps = layer_norm_eps
        self.num_choices = num_choices
        self.gen_weight = gen_weight
        self.disc_weight = disc_weight
