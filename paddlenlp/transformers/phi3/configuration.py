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
"""Phi3 model configuration"""

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["Phi3Config"]


class Phi3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Phi3Model`]. It is used to instantiate a
    Phi3 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 51200):
            Vocabulary size of the Phi3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Phi3Model`].
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key/value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA), and if
            `1 < num_key_value_heads < num_attention_heads` the model will use GQA.
            For more details, see [this paper](https://arxiv.org/pdf/2305.13245.pdf).
            If it is not specified, will default to `num_attention_heads`.
        intermediate_size (`int`, *optional*, defaults to 10240):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of sequence token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
    """
    model_type = "phi3"
    pretrained_init_configuration = {
        "phi3-small": {
            "vocab_size": 51200,
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 10240,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 2048,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "num_key_value_heads": 8,
            "rope_theta": 10000.0,
        },
        "phi3-base": {
            "vocab_size": 51200,
            "hidden_size": 5120,
            "num_hidden_layers": 48,
            "num_attention_heads": 64,
            "intermediate_size": 20480,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 2048,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
            "num_key_value_heads": 8,
            "rope_theta": 10000.0,
        },
    }
    pretrained_resource_files_map = {
        "model_state": {
            "phi3-small": "https://bj.bcebos.com/paddlenlp/models/transformers/phi3/phi3-small/model_state.pdparams",
            "phi3-base": "https://bj.bcebos.com/paddlenlp/models/transformers/phi3/phi3-base/model_state.pdparams",
        }
    }
    base_model_prefix = "phi3"

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        intermediate_size=10240,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        **kwargs,
    ):
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
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
