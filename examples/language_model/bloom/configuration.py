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
""" Bloom model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["BLOOM_PRETRAINED_INIT_CONFIGURATION", "BloomConfig", "BLOOM_PRETRAINED_RESOURCE_FILES_MAP"]

BLOOM_PRETRAINED_INIT_CONFIGURATION = {
    "bigscience/bloom-560m": {
        "apply_residual_connection_post_layernorm": False,
        "architectures": ["BloomModel"],
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "bias_dropout_fusion": True,
        "bos_token_id": 1,
        "dtype": "float32",
        "eos_token_id": 2,
        "hidden_dropout": 0.0,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "masked_softmax_fusion": True,
        "model_type": "bloom",
        "n_embed": 1024,
        "n_layer": 24,
        "num_attention_heads": 16,
        "offset_alibi": 100,
        "pad_token_id": 3,
        "pretraining_tp": 1,
        "skip_bias_add": True,
        "skip_bias_add_qkv": False,
        "slow_but_exact": False,
        "unk_token_id": 0,
        "vocab_size": 250880,
        "mp_degree": 1,
        "pp_degree": 1,
        "mp_rank": -1,
        "use_recompute": False,
        "use_pure_fp16": False,
    }
}

BLOOM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "bigscience/bloom-560m": "https://paddlenlp.bj.bcebos.com/models/community/bigscience/bloom-560m/model_state.pdparams"
    }
}


class BloomConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BloomModel`]. It is used to
    instantiate a BLOOM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BLOOM
    bigscience/bloom-560m architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from paddlenlp.transformers import BloomModel, BloomConfig

    >>> # Initializing a BLOOM bigscience/bloom-560m style configuration
    >>> configuration = BloomConfig()

    >>> # Initializing a model from the bigscience/bloom-560m style configuration
    >>> model = BloomModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bloom"
    attribute_map: Dict[str, str] = {}  # noqa: F811
    attribute_map = {"n_layer": "num_hidden_layers", "n_head": "num_attention_heads", "n_embed": "hidden_size"}

    pretrained_init_configuration = BLOOM_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        masked_softmax_fusion=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        pretraining_tp=1,  # TP rank used when training with megatron
        dtype="bfloat16",
        slow_but_exact=False,
        mp_degree=1,
        pp_degree=1,
        mp_rank=0,
        use_recompute=False,
        use_pure_fp16=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.masked_softmax_fusion = masked_softmax_fusion
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dtype = dtype
        self.slow_but_exact = slow_but_exact
        self.mp_degree = mp_degree
        self.pp_degree = mp_degree
        self.mp_rank = mp_rank
        self.use_recompute = use_recompute
        self.use_pure_fp16 = use_pure_fp16

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
