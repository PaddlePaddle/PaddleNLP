# !/usr/bin/env python3
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
""" Ernie35 model configuration"""

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.utils.log import logger

__all__ = [
    "ERNIE_PRETRAINED_INIT_CONFIGURATION",
    "Ernie35Config",
    "ERNIE_PRETRAINED_RESOURCE_FILES_MAP",
]

ERNIE_PRETRAINED_INIT_CONFIGURATION = {
    "ernie/tiny-random-ernie": {
        "fuse_linear": False,
        "fuse_ln": False,
        "hidden_size": 768,
        "ignored_index": -100,
        "initializer_range": 0.02,
        "intermediate_size": 2048,
        "max_position_embeddings": 4096,
        "model_type": "ernie",
        "num_attention_heads": 12,
        "num_hidden_layers": 3,
        "pad_token_id": 0,
        "parallel_attn_hatf": True,
        "enable_random_position_ids": True,
        "use_progressive_seq_len": False,
        "layer_norm_eps": 1e-06,
        "tensor_parallel_output": True,
        "tie_word_embeddings": False,
        "use_bias": True,
        "use_flash_attention": True,
        "use_recompute": False,
        "use_recompute_attn": False,
        "vocab_size": 32000,
        "weight_share_add_bias": True,
    },
    "baidu/ernie-3.5-se-3b": {
        "fuse_linear": False,
        "fuse_ln": False,
        "hidden_size": 3072,
        "ignored_index": -100,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 32768,  # 32k
        "model_type": "ernie",
        "num_attention_heads": 24,
        "num_hidden_layers": 32,
        "pad_token_id": 0,
        "parallel_attn_hatf": True,
        "enable_random_position_ids": True,
        "use_progressive_seq_len": True,
        "layer_norm_eps": 1e-06,
        "tensor_parallel_output": True,
        "tie_word_embeddings": False,
        "use_bias": True,
        "use_flash_attention": True,
        "use_recompute": False,
        "use_recompute_attn": False,
        "vocab_size": 32000,
        "weight_share_add_bias": True,
    },
}

# Hypothetical model weights currently
ERNIE_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {},
}


class Ernie35Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~Ernie35Model`]. It is used to instantiate an Ernie35
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Ernie35.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the Ernie35 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~Ernie35Model`].
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
    """
    model_type = "ernie"
    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }
    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=768,
        intermediate_size=11008,
        max_position_embeddings=2048,
        num_hidden_layers=2,
        num_attention_heads=2,
        initializer_range=0.02,  # no use
        layer_norm_eps=1e-6,
        use_cache=True,
        use_flash_attention=True,
        use_recompute=False,
        use_recompute_attn=False,
        fuse_ln=False,
        tensor_parallel_output=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_bias=False,
        sequence_parallel=False,
        weight_share=False,  # non-PP only
        weight_share_add_bias=True,
        fuse_linear=False,
        seqlen=False,
        virtual_pp_degree=1,
        ignored_index=-100,
        parallel_attn_hatf=True,
        enable_random_position_ids=False,
        use_progressive_seq_len=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_recompute_attn = use_recompute_attn
        if use_recompute_attn:
            logger.warning("set `use_recompute_attn`=True, disabling `use_recompute`")
            use_recompute = False
        self.use_recompute = use_recompute
        self.use_flash_attention = use_flash_attention
        self.tensor_parallel_output = tensor_parallel_output
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.fuse_ln = fuse_ln
        self.sequence_parallel = sequence_parallel
        self.seqlen = seqlen
        self.virtual_pp_degree = virtual_pp_degree
        self.use_bias = use_bias
        self.weight_share_add_bias = weight_share_add_bias
        kwargs["tie_word_embeddings"] = weight_share
        self.fuse_linear = fuse_linear
        self.ignored_index = ignored_index
        self.parallel_attn_hatf = parallel_attn_hatf
        self.enable_random_position_ids = enable_random_position_ids
        self.use_progressive_seq_len = use_progressive_seq_len
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tensor_parallel_output=tensor_parallel_output,
            **kwargs,
        )
        if self.sequence_parallel:
            assert self.seqlen, "seqlen not provided in sequence-parallel"
            assert (
                self.tensor_parallel_degree > 1
            ), f"senquence-parallel only works in mp, got mp={self.tensor_parallel_degree}"
