# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "LLAMA_PRETRAINED_INIT_CONFIGURATION",
    "LLaMAConfig",
    "LLAMA_PRETRAINED_RESOURCE_FILES_MAP",
]

LLAMA_PRETRAINED_INIT_CONFIGURATION = {
    "llama-7b": {
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "mp_degree": 1,
        "pp_degree": 1,
        "mp_rank": -1,
        "use_cache": False,
        "use_recompute": False,
        "use_pure_fp16": False,
    },
}

# Hypothetical model weights currently
LLAMA_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "llama-7b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/llama-7b.pdparams",
    },
}


class LLaMAConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~LLaMAModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LLaMAModel`] or [`~TFLLaMAModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:
    ```python
    >>> from paddlenlp.transformer import LLaMAModel, LLaMAConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LLaMAConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LLaMAModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_hidden_layers=2,
        num_attention_heads=2,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
