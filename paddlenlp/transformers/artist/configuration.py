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

from paddlenlp.transformers import GPTConfig

__all__ = ["ARTIST_PRETRAINED_INIT_CONFIGURATION", "ARTIST_PRETRAINED_RESOURCE_FILES_MAP", "ArtistConfig"]

ARTIST_PRETRAINED_INIT_CONFIGURATION = {
    "pai-painter-base-zh": {
        "vocab_size": 37512,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
    "pai-painter-painting-base-zh": {
        "vocab_size": 37512,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
    "pai-painter-scenery-base-zh": {
        "vocab_size": 37512,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
    "pai-painter-commercial-base-zh": {
        "vocab_size": 37512,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,  # no use
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
    "pai-painter-large-zh": {
        "vocab_size": 37512,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu_python",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 288,
        "type_vocab_size": 1,
        "initializer_range": 0.02,
        "pad_token_id": 16384,  # 0 + 16384
        "eos_token_id": 16486,  # 102 + 16384
        "bos_token_id": 16485,  # 101 + 16384
        "eol_token_id": 16486,  # 102 + 16384
    },
}
ARTIST_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "pai-painter-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-base-zh/model_state.pdparams",
        "pai-painter-painting-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-painting-base-zh/model_state.pdparams",
        "pai-painter-scenery-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-scenery-base-zh/model_state.pdparams",
        "pai-painter-commercial-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-commercial-base-zh/model_state.pdparams",
        "pai-painter-large-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/artist/pai-painter-large-zh/model_state.pdparams",
    }
}


class ArtistConfig(GPTConfig):
    pretrained_init_configuration = ARTIST_PRETRAINED_INIT_CONFIGURATION
