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
""" Pegasus model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

from ...utils.log import logger

__all__ = ["PEGASUS_PRETRAINED_INIT_CONFIGURATION", "PegasusConfig"]

PEGASUS_PRETRAINED_INIT_CONFIGURATION = {}


class PegasusConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PegasusModel`]. It is used to instantiate a PEGASUS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the PEGASUS pegasus-238M architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, optional):
            Vocabulary size of the PEGASUS model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`PegasusModel`]. Default to 50000.
        d_model (`int`, optional):
            Dimensionality of the layers and the pooler layer. Default to 1024
        encoder_layers (`int`, optional):
            Number of encoder layers. Default to 12.
        decoder_layers (`int`, optional):
            Number of decoder layers. Default to 12.
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
            Default to `"relu"`.
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
            The number of labels. Default to 3.
        forced_eos_token_id (`int`, optional):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`. Default to 1.
        scale_embedding (`bool`, optional):
            Scale embeddings by diving by sqrt(d_model). Default to `False`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.

    """
    model_type = "pegasus"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads", 
        "hidden_size": "d_model",
        "num_classes": "num_labels",
    }
    pretrained_init_configuration = PEGASUS_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 50000,
        max_position_embeddings: int = 1024,
        encoder_layers: int = 12,
        encoder_ffn_dim: int = 3072,
        encoder_attention_heads: int = 12,
        decoder_layers: int = 12,
        decoder_ffn_dim: int = 3072,
        decoder_attention_heads: int = 12,
        activation_function: str = "relu",
        d_model: int = 768,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        init_std: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        is_encoder_decoder: bool = True,
        decoder_start_token_id: int = 0,
        forced_eos_token_id: int = 1,
        scale_embedding: bool = True,
        use_cache: bool = True,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        **kwargs,
    ):
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
        self.use_cache = use_cache
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            logger.warning(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )

