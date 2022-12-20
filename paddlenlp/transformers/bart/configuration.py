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
""" Bart model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

from ...utils.log import logger

__all__ = ["BART_PRETRAINED_INIT_CONFIGURATION", "BartConfig", "BART_PRETRAINED_RESOURCE_FILES_MAP"]

BART_PRETRAINED_INIT_CONFIGURATION = {
    "bart-base": {
        "vocab_size": 50265,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "forced_eos_token_id": 2,
        "decoder_start_token_id": 2,
        "d_model": 768,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "encoder_attention_heads": 12,
        "decoder_attention_heads": 12,
        "encoder_ffn_dim": 3072,
        "decoder_ffn_dim": 3072,
        "dropout": 0.1,
        "activation_function": "gelu",
        "attention_dropout": 0.1,
        "activation_dropout": 0.1,
        "max_position_embeddings": 1024,
        "init_std": 0.02,
        "scale_embedding": False,
    },
    "bart-large": {
        "vocab_size": 50265,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "forced_eos_token_id": 2,
        "decoder_start_token_id": 2,
        "d_model": 1024,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "encoder_attention_heads": 16,
        "decoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "decoder_ffn_dim": 4096,
        "dropout": 0.1,
        "activation_function": "gelu",
        "attention_dropout": 0.1,
        "activation_dropout": 0.1,
        "max_position_embeddings": 1024,
        "init_std": 0.02,
        "scale_embedding": False,
    },
}

BART_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "bart-base": "https://bj.bcebos.com/paddlenlp/models/transformers/bart/bart-base.pdparams",
        "bart-large": "https://bj.bcebos.com/paddlenlp/models/transformers/bart/bart-large.pdparams",
    }
}


class BartConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BartModel`]. It is used to instantiate a BART
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BART bart-base architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, optional):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BartModel`] or [`TFBartModel`]. Default to 50265.
        d_model (`int`, optional):
            Dimensionality of the layers and the pooler layer. Default to 1024
        encoder_layers (`int`, optional):
            Number of encoder layers. Default to 6.
        decoder_layers (`int`, optional):
            Number of decoder layers. Default to 6.
        encoder_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer encoder. Default to 12.
        decoder_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer decoder. Default to 12.
        decoder_ffn_dim (`int`, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder. Default to 3072.
        encoder_ffn_dim (`int`, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder. Default to 3072.
        activation_function (`str` or `function`, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions are supported.
            Default to `"gelu"`.
        dropout (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler. Default to 0.1.
        attention_dropout (`float`, optional):
            The dropout ratio for the attention probabilities. Default to 0.1.
        activation_dropout (`float`, optional):
            The dropout ratio for activations inside the fully connected layer. Default to 0.1.
        max_position_embeddings (`int`, optional):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048). Default to 1024.
        init_std (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices. Default to 0.02.
        num_labels (`int`, optional):
            The number of labels to use in [`BartForSequenceClassification`]. Default to 3.
        forced_eos_token_id (`int`, optional):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`. Default to 2.
        scale_embedding (`bool`, optional):
            Scale embeddings by diving by sqrt(d_model). Default to `False`.

    """
    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]
    standard_config_map: Dict[str, str] = {
        "num_encoder_layers": "encoder_layers",
        "num_decoder_layers": "decoder_layers",
    }
    pretrained_init_configuration = BART_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 50265,
        max_position_embeddings: int = 1024,
        encoder_layers: int = 6,
        encoder_ffn_dim: int = 3072,
        encoder_attention_heads: int = 12,
        decoder_layers: int = 6,
        decoder_ffn_dim: int = 3072,
        decoder_attention_heads: int = 12,
        activation_function: str = "gelu",
        d_model: int = 768,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        init_std: float = 0.02,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        is_encoder_decoder: bool = True,
        decoder_start_token_id: int = 2,
        forced_eos_token_id: int = 2,
        scale_embedding: bool = False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding

        # ensure backward compatibility for BART CNN models
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            logger.warning(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )
