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
""" Llama model configuration"""

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "LLAMA_PRETRAINED_INIT_CONFIGURATION",
    "LlamaConfig",
    "LLAMA_PRETRAINED_RESOURCE_FILES_MAP",
]

LLAMA_PRETRAINED_INIT_CONFIGURATION = {
    # Hypothetical model weights (tiny-random-llama & micro-random-llama) for test only
    "__internal_testing__/micro-random-llama": {
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 1000,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 8,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
        "use_recompute": False,
        "use_flash_attention": False,
    },
    "__internal_testing__/tiny-random-llama": {
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 8,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
        "use_recompute": False,
        "use_flash_attention": False,
    },
    "facebook/llama-7b": {
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
        "use_cache": False,
        "use_recompute": False,
        "use_flash_attention": False,
    },
    "facebook/llama-13b": {
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 13824,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
        "use_recompute": False,
        "use_flash_attention": False,
    },
    "facebook/llama-30b": {
        "hidden_size": 6656,
        "initializer_range": 0.02,
        "intermediate_size": 17920,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 52,
        "num_hidden_layers": 60,
        "rms_norm_eps": 1e-06,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
        "use_recompute": False,
        "use_flash_attention": False,
    },
    "facebook/llama-65b": {
        "hidden_size": 8192,
        "initializer_range": 0.02,
        "intermediate_size": 22016,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "rms_norm_eps": 1e-05,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
        "use_recompute": False,
        "use_flash_attention": False,
    },
}

# Hypothetical model weights (tiny-random-llama) for test only
LLAMA_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "__internal_testing__/micro-random-llama": "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/micro-random-llama/model_state.pdparams",
        "__internal_testing__/tiny-random-llama": "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-llama/model_state.pdparams",
        "facebook/llama-7b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-7b/model_state.pdparams",
        "facebook/llama-13b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-13b/model_state.pdparams",
        "facebook/llama-30b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-30b/model_state.pdparams",
        "facebook/llama-65b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-65b/model_state.pdparams",
    },
}


class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~LlamaModel`]. It is used to instantiate an Llama
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Llama-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Llama model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LlamaModel`] or [`~TFLlamaModel`].
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
        rope_fusion_level(`str`, *optional*, defaults to ``):
            The level of fusion of rope embedding. Can be chosen from:
            (1) 'full': fuse sin cos compute and rope embedding
            (2) 'core': only fuse rope embedding, will compute the sin and cos
            (3) None: don't fuse any part of the rope embedding
        Example:
    ```python
    >>> from paddlenlp.transformer import LlamaModel, LlamaConfig

    >>> # Initializing a Llama llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }
    pretrained_init_configuration = LLAMA_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_hidden_layers=32,
        num_attention_heads=32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        use_recompute=False,
        recompute_granularity="full",
        fuse_attention_qkv=False,
        use_flash_attention=False,
        fuse_attention_ffn=False,
        use_fused_rms_norm=False,
        tensor_parallel_output=True,
        sequence_parallel=False,
        fuse_sequence_parallel_allreduce=False,
        lm_shift_labels=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        alibi=False,
        rope_fusion_level=None,
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
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.fuse_attention_qkv = fuse_attention_qkv
        self.use_flash_attention = use_flash_attention
        self.fuse_attention_ffn = fuse_attention_ffn
        self.use_fused_rms_norm = use_fused_rms_norm
        self.tensor_parallel_output = tensor_parallel_output
        self.sequence_parallel = sequence_parallel
        self.fuse_sequence_parallel_allreduce = fuse_sequence_parallel_allreduce
        self.lm_shift_labels = lm_shift_labels

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.alibi = alibi

        self.rope_fusion_level = rope_fusion_level

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            tensor_parallel_output=tensor_parallel_output,
            **kwargs,
        )

    @property
    def rope(self):
        return not self.alibi
