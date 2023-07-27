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
""" T5 model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["T5_PRETRAINED_INIT_CONFIGURATION", "T5Config", "T5_PRETRAINED_RESOURCE_FILES_MAP"]

T5_PRETRAINED_INIT_CONFIGURATION = {
    "t5-small": {
        "tie_word_embeddings": True,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "vocab_size": 32128,
        "d_model": 512,
        "d_kv": 64,
        "d_ff": 2048,
        "num_layers": 6,
        "num_decoder_layers": 6,
        "num_heads": 8,
        "relative_attention_num_buckets": 32,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-06,
        "initializer_factor": 1.0,
        "feed_forward_proj": "relu",
    },
    "t5-base": {
        "tie_word_embeddings": True,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "vocab_size": 32128,
        "d_model": 768,
        "d_kv": 64,
        "d_ff": 3072,
        "num_layers": 12,
        "num_decoder_layers": 12,
        "num_heads": 12,
        "relative_attention_num_buckets": 32,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-06,
        "initializer_factor": 1.0,
        "feed_forward_proj": "relu",
    },
    "t5-large": {
        "tie_word_embeddings": True,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "vocab_size": 32128,
        "d_model": 1024,
        "d_kv": 64,
        "d_ff": 4096,
        "num_layers": 24,
        "num_decoder_layers": 24,
        "num_heads": 16,
        "relative_attention_num_buckets": 32,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-06,
        "initializer_factor": 1.0,
        "feed_forward_proj": "relu",
    },
    "t5-v1_1-base": {
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "vocab_size": 32128,
        "d_model": 768,
        "d_kv": 64,
        "d_ff": 2048,
        "num_layers": 12,
        "num_decoder_layers": 12,
        "num_heads": 12,
        "relative_attention_num_buckets": 32,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-06,
        "initializer_factor": 1.0,
        "feed_forward_proj": "gated-gelu",
    },
    "t5-v1_1-large": {
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "vocab_size": 32128,
        "d_model": 1024,
        "d_kv": 64,
        "d_ff": 2816,
        "num_layers": 24,
        "num_decoder_layers": 24,
        "num_heads": 16,
        "relative_attention_num_buckets": 32,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-06,
        "initializer_factor": 1.0,
        "feed_forward_proj": "gated-gelu",
    },
    "t5-3b": {
        "tie_word_embeddings": True,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "vocab_size": 32128,
        "d_model": 1024,
        "d_kv": 128,
        "d_ff": 16384,
        "num_layers": 24,
        "num_decoder_layers": 24,
        "num_heads": 32,
        "relative_attention_num_buckets": 32,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-06,
        "initializer_factor": 1.0,
        "feed_forward_proj": "relu",
    },
    "t5-11b": {
        "tie_word_embeddings": True,
        "pad_token_id": 0,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "vocab_size": 32128,
        "d_model": 1024,
        "d_kv": 128,
        "d_ff": 65536,
        "num_layers": 24,
        "num_decoder_layers": 24,
        "num_heads": 128,
        "relative_attention_num_buckets": 32,
        "dropout_rate": 0.1,
        "layer_norm_epsilon": 1e-06,
        "initializer_factor": 1.0,
        "feed_forward_proj": "relu",
    },
}

T5_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "t5-small": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-small/model_state.pdparams",
        "t5-base": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-base/model_state.pdparams",
        "t5-large": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-large/model_state.pdparams",
        "t5-3b": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-3b/model_state.pdparams",
        "t5-11b": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-11b/model_state.pdparams",
        "t5-v1_1-base": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-base/model_state.pdparams",
        "t5-v1_1-large": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-large/model_state.pdparams",
    }
}


class T5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`T5Model`]. It is used to
    instantiate a bert model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the T5
    t5-small architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `T5Block`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            he non-linear activation function (function or string) in the feed forward layer in the residual attention block.
            If string, `"relu"`, `"gated-gelu"` are supported. Defaults to `"relu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (int, optional):
            The id of the `padding` token. Defaults to `0`.
        bos_token_id (int, optional):
            The id of the `bos` token. Defaults to `0`.
        eos_token_id (int, optional):
            The id of the `eos` token. Defaults to `1`.
    """
    model_type = "t5"
    attribute_map: Dict[str, str] = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_classes": "num_labels",
    }
    pretrained_init_configuration = T5_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_decoder_layers: int = None,
        num_heads: int = 8,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 1.0,
        feed_forward_proj: str = "relu",
        is_encoder_decoder: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs
    ):

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
