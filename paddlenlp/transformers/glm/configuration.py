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
"""GLM model configuration"""

from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = [
    "GLMConfig",
    "GLM_PRETRAINED_INIT_CONFIGURATION",
    "GLM_PRETRAINED_RESOURCE_FILES_MAP",
]


GLM_PRETRAINED_INIT_CONFIGURATION = {
    "THUDM/glm-515m": {
        "attention_dropout_prob": 0.1,
        "attention_scale": 1.0,
        "block_position_encoding": True,
        "checkpoint_num_layers": 1,
        "embedding_dropout_prob": 0.1,
        "hidden_size": 1152,
        "initializer_range": 0.02,
        "max_sequence_length": 512,
        "model_type": "glm",
        "num_attention_heads": 18,
        "num_layers": 30,
        "layernorm_epsilon": 1e-5,
        "output_dropout_prob": 0.1,
        "output_predict": True,
        "parallel_output": False,
        "pool_token": "cls",
        "relative_encoding": False,
        "spell_func": "lstm",
        "spell_length": None,
        "use_scaled_init_for_output_weights": True,
        "vocab_size": 30592,
    },
    "THUDM/glm-2b": {
        "attention_dropout_prob": 0.1,
        "attention_scale": 1.0,
        "block_position_encoding": True,
        "checkpoint_num_layers": 1,
        "embedding_dropout_prob": 0.1,
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "max_sequence_length": 1024,
        "model_type": "glm",
        "num_attention_heads": 32,
        "num_layers": 36,
        "output_dropout_prob": 0.1,
        "output_predict": True,
        "parallel_output": True,
        "pool_token": "cls",
        "relative_encoding": False,
        "spell_func": "lstm",
        "spell_length": None,
        "vocab_size": 50304,
    },
    "THUDM/glm-10b": {
        "attention_dropout_prob": 0.1,
        "attention_scale": 1.0,
        "block_position_encoding": True,
        "checkpoint_num_layers": 1,
        "embedding_dropout_prob": 0.1,
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "max_sequence_length": 1024,
        "model_type": "glm",
        "num_attention_heads": 64,
        "num_layers": 48,
        "output_dropout_prob": 0.1,
        "output_predict": True,
        "parallel_output": True,
        "pool_token": "cls",
        "relative_encoding": False,
        "spell_func": "lstm",
        "spell_length": None,
        "vocab_size": 50304,
    },
    "THUDM/glm-large-chinese": {
        "attention_dropout_prob": 0.1,
        "attention_scale": 1.0,
        "block_position_encoding": True,
        "checkpoint_num_layers": 1,
        "embedding_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "max_sequence_length": 1024,
        "model_type": "glm",
        "num_attention_heads": 16,
        "num_layers": 24,
        "layernorm_epsilon": 1e-5,
        "output_dropout_prob": 0.1,
        "output_predict": True,
        "parallel_output": False,
        "pool_token": "cls",
        "relative_encoding": False,
        "spell_func": "lstm",
        "spell_length": None,
        "vocab_size": 50048,
    },
    "THUDM/glm-10b-chinese": {
        "attention_dropout_prob": 0.1,
        "attention_scale": 1.0,
        "block_position_encoding": True,
        "checkpoint_num_layers": 1,
        "embedding_dropout_prob": 0.1,
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "max_sequence_length": 1024,
        "model_type": "glm",
        "num_attention_heads": 64,
        "num_layers": 48,
        "output_dropout_prob": 0.1,
        "output_predict": True,
        "parallel_output": True,
        "pool_token": "cls",
        "relative_encoding": False,
        "spell_func": "lstm",
        "spell_length": None,
        "vocab_size": 50048,
        "bad_words_id": [50009],
    },
}

GLM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "THUDM/glm-515m": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-515m.pdparams",
        "THUDM/glm-2b": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-2b.pdparams",
        "THUDM/glm-10b": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-10b.pdparams",
        "THUDM/glm-large-chinese": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-large-chinese.pdparams",
        "THUDM/glm-10b-chinese": "https://paddlenlp.bj.bcebos.com/models/community/THUDM/glm-10b-chinese.pdparams",
    }
}


class GLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GLMModel`].
    It is used to instantiate an GLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the GLM [shunxing1234/GLM-base-cased](https://huggingface.co/shunxing1234/GLM-base-cased) architecture.
    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the GLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~GLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~GLMModel`] or
            [`~TFGLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
    Example:
    ```python
    >>> from paddlenlp.transformers import GLMModel, GLMConfig
    >>> # Initializing a GLM shunxing1234/GLM-base-cased style configuration
    >>> configuration = GLMConfig()
    >>> # Initializing a model from the shunxing1234/GLM-base-cased style configuration
    >>> model = GLMModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "glm"
    attribute_map: Dict[str, str] = {"num_hidden_layers": "num_layers", "torch_dtype": "dtype"}
    pretrained_init_configuration = GLM_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        num_layers=24,
        vocab_size=30592,
        hidden_size=1024,
        num_attention_heads=16,
        embedding_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        max_sequence_length=512,
        checkpoint_num_layers=1,
        parallel_output=True,
        relative_encoding=False,
        block_position_encoding=True,
        output_predict=False,
        spell_length=None,
        spell_func="lstm",
        attention_scale=1.0,
        initializer_range=0.02,
        pool_token="cls",
        layernorm_epsilon=1e-5,
        use_scaled_init_for_output_weights=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        self.output_predict = output_predict
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale
        self.initializer_range = initializer_range
        self.pool_token = pool_token
        self.layernorm_epsilon = layernorm_epsilon
        self.use_scaled_init_for_output_weights = use_scaled_init_for_output_weights
        self._fast_entry = None
