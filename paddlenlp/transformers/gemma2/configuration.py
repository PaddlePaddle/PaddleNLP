# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Gemma2 model configuration"""

from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["Gemma2Config"]

GEMMA2_PRETRAINED_INIT_CONFIGURATION = {
    "gemma2-2b": {
        "vocab_size": 256000,
        "hidden_size": 2048,
        "num_hidden_layers": 18,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "intermediate_size": 8192,
        "hidden_act": "gelu",
        "max_position_embeddings": 8192,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
    },
    "gemma2-7b": {
        "vocab_size": 256000,
        "hidden_size": 4096,
        "num_hidden_layers": 28,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 16384,
        "hidden_act": "gelu",
        "max_position_embeddings": 8192,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
    },
}

GEMMA2_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "gemma2-2b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma2-2b.pdparams",
        "gemma2-7b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma2-7b.pdparams",
    }
}


class Gemma2Config(PretrainedConfig):
    """
    Configuration class for Gemma2 model.
    """

    model_type = "gemma2"
    attribute_map: Dict[str, str] = {"num_classes": "num_labels"}
    pretrained_init_configuration = GEMMA2_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 16384,
        hidden_act: str = "gelu",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        **kwargs,
    ) -> None:
        """Initialize the Gemma2 configuration.

        Args:
            vocab_size (int): Vocabulary size of the model.
            hidden_size (int): Dimensionality of the hidden states.
            num_hidden_layers (int): Number of hidden layers in the model.
            num_attention_heads (int): Number of attention heads for each attention layer.
            num_key_value_heads (int): Number of key value heads for each attention layer (grouped query attention).
            intermediate_size (int): Dimensionality of the intermediate feed-forward layer.
            hidden_act (str): Non-linear activation function used in the model.
            max_position_embeddings (int): Maximum sequence length supported by the model.
            initializer_range (float): Standard deviation for initializing model parameters.
            rms_norm_eps (float): Epsilon used by the RMS normalization layers.
            use_cache (bool): Whether to use the model cache to speed up decoding.
            pad_token_id (int): ID of the padding token.
            bos_token_id (int): ID of the beginning of sequence token.
            eos_token_id (int): ID of the end of sequence token.
            tie_word_embeddings (bool): Whether to tie input and output embeddings.
            rope_theta (float): Base value for rotary position embedding.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
