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
""" ErnieCode model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["ERNIECODE_PRETRAINED_INIT_CONFIGURATION", "ErnieCodeConfig", "ERNIECODE_PRETRAINED_RESOURCE_FILES_MAP"]

ERNIECODE_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-code-base": {
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 768,
        "decoder_start_token_id": 0,
        "dense_act_fn": "gelu_new",
        "dropout_rate": 0.1,
        "enable_recompute": False,
        "eos_token_id": 1,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "is_gated_act": True,
        "layer_norm_epsilon": 1e-06,
        "model_type": "ErnieCode",
        "num_decoder_layers": 12,
        "num_heads": 12,
        "num_layers": 12,
        "output_past": True,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": False,
        "tokenizer_class": "ErnieCodeTokenizer",
        "transformers_version": "4.20.1",
        "use_cache": True,
        "vocab_size": 250105,
    },
    "ernie-code-base-L512": {
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 768,
        "decoder_start_token_id": 0,
        "dense_act_fn": "gelu_new",
        "dropout_rate": 0.1,
        "enable_recompute": False,
        "eos_token_id": 1,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "is_gated_act": True,
        "layer_norm_epsilon": 1e-06,
        "model_type": "ErnieCode",
        "num_decoder_layers": 12,
        "num_heads": 12,
        "num_layers": 12,
        "output_past": True,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": False,
        "tokenizer_class": "ErnieCodeTokenizer",
        "transformers_version": "4.20.1",
        "use_cache": True,
        "vocab_size": 250105,
    },
}

ERNIECODE_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "ernie-code-base": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-code/ernie-code-base/model_state.pdparams",
        "ernie-code-base-L512": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-code/ernie-code-base-L512/model_state.pdparams",
    }
}


class ErnieCodeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieCodeModel`]. It is used to
    instantiate a bert model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 250112):
            Vocabulary size of the ErnieCode model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ErnieCodeModel`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 1024):
            Size of the intermediate feed forward layer in each `ErnieCodeBlock`.
        num_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 6):
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
        feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`):
            he non-linear activation function (function or string) in the feed forward layer in the residual attention block.
            If string, `"relu"`, `"gated-gelu"` are supported. Defaults to `"gated-gelu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (int, optional):
            The id of the `padding` token. Defaults to `0`.
        bos_token_id (int, optional):
            The id of the `bos` token. Defaults to `0`.
        eos_token_id (int, optional):
            The id of the `eos` token. Defaults to `1`.
        enable_recompute (bool, optional):
            Whether to recompute cache.

    """
    model_type = "ErnieCode"
    attribute_map: Dict[str, str] = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "num_classes": "num_labels",
    }
    pretrained_init_configuration = ERNIECODE_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 250112,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 1024,
        num_layers: int = 8,
        num_decoder_layers: int = None,
        num_heads: int = 6,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 1.0,
        feed_forward_proj: str = "gated-gelu",
        is_encoder_decoder: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 0,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        enable_recompute: bool = False,
        **kwargs
    ):

        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        self.enable_recompute = enable_recompute
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
