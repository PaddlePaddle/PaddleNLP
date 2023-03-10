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

""" RoFormerv2 model configuration """
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["RoFormerv2Config", "ROFORMERV2_PRETRAINED_INIT_CONFIGURATION", "ROFORMERV2_PRETRAINED_RESOURCE_FILES_MAP"]

ROFORMERV2_PRETRAINED_INIT_CONFIGURATION = {
    "roformer_v2_chinese_char_small": {
        "vocab_size": 12000,
        "hidden_size": 384,
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "pad_token_id": 0,
        "rotary_value": False,
        "use_bias": False,
    },
    "roformer_v2_chinese_char_base": {
        "vocab_size": 12000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "pad_token_id": 0,
        "rotary_value": False,
        "use_bias": False,
    },
    "roformer_v2_chinese_char_large": {
        "vocab_size": 12000,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "pad_token_id": 0,
        "rotary_value": False,
        "use_bias": False,
    },
}

ROFORMERV2_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "roformer_v2_chinese_char_small": "https://bj.bcebos.com/paddlenlp/models/transformers/roformerv2/roformer_v2_chinese_char_small/model_state.pdparams",
        "roformer_v2_chinese_char_base": "https://bj.bcebos.com/paddlenlp/models/transformers/roformerv2/roformer_v2_chinese_char_base/model_state.pdparams",
        "roformer_v2_chinese_char_large": "https://bj.bcebos.com/paddlenlp/models/transformers/roformerv2/roformer_v2_chinese_char_large/model_state.pdparams",
    }
}


class RoFormerv2Config(PretrainedConfig):
    model_type = "roformerv2"
    pretrained_init_configuration = ROFORMERV2_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 12000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "relu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        act_dropout: float = 0,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        rotary_value: bool = False,
        use_bias: bool = False,
        epsilon: float = 1e-12,
        normalize_before: bool = False,
        num_choices: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.act_dropout = act_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.pad_token_id = pad_token_id
        self.rotary_value = rotary_value
        self.use_bias = use_bias
        self.epsilon = epsilon
        self.normalize_before = normalize_before
        self.num_choices = num_choices
