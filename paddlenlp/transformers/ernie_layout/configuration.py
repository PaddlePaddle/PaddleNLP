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
""" ERNIE-Layout model configuration"""

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = [
    "ERNIE_LAYOUT_PRETRAINED_INIT_CONFIGURATION",
    "ErnieLayoutConfig",
    "ERNIE_LAYOUT_PRETRAINED_RESOURCE_FILES_MAP",
]

ERNIE_LAYOUT_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-layoutx-base-uncased": {
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 0,
        "coordinate_size": 128,
        "eos_token_id": 2,
        "gradient_checkpointing": False,
        "has_relative_attention_bias": True,
        "has_spatial_attention_bias": True,
        "has_visual_segment_embedding": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "image_feature_pool_shape": [7, 7, 256],
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_2d_position_embeddings": 1024,
        "max_position_embeddings": 514,
        "max_rel_2d_pos": 256,
        "max_rel_pos": 128,
        "model_type": "ernie_layout",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "output_past": True,
        "pad_token_id": 1,
        "shape_size": 128,
        "rel_2d_pos_bins": 64,
        "rel_pos_bins": 32,
        "type_vocab_size": 100,
        "vocab_size": 250002,
    },
    "uie-x-base": {
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 0,
        "coordinate_size": 128,
        "eos_token_id": 2,
        "gradient_checkpointing": False,
        "has_relative_attention_bias": True,
        "has_spatial_attention_bias": True,
        "has_visual_segment_embedding": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "image_feature_pool_shape": [7, 7, 256],
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_2d_position_embeddings": 1024,
        "max_position_embeddings": 514,
        "max_rel_2d_pos": 256,
        "max_rel_pos": 128,
        "model_type": "ernie_layout",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "output_past": True,
        "pad_token_id": 1,
        "shape_size": 128,
        "rel_2d_pos_bins": 64,
        "rel_pos_bins": 32,
        "type_vocab_size": 100,
        "vocab_size": 250002,
    },
}

ERNIE_LAYOUT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "ernie-layoutx-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_layout/ernie_layoutx_base_uncased.pdparams",
        "uie-x-base": "https://bj.bcebos.com/paddlenlp/models/transformers/uie_x/uie_x_base.pdparams",
    },
}


class ErnieLayoutConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieLayoutModel`]. It is used to
    instantiate a ErnieLayout model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ErnieLayout
    ernie-layoutx-base-uncased architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 250002):
            Vocabulary size of the ErnieLayout model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ErnieLayoutModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 514):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 514 or 1028 or 2056).
        type_vocab_size (`int`, *optional*, defaults to 100):
            The vocabulary size of the `token_type_ids` passed when calling [`ErnieModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for classifier.
        has_visual_segment_embedding (`bool`, *optional*, defaults to `False`):
            Whether or not the model has visual segment embedding.
    Examples:
    ```python
    >>> from paddlenlp.transformers import ErnieLayoutModel, ErnieLayoutConfig
    >>> # Initializing a ErnieLayout ernie-layoutx-base-uncased configuration
    >>> configuration = ErnieLayoutConfig()
    >>> # Initializing a model from the  style configuration
    >>> model = ErnieLayoutModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "ernie_layout"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = ERNIE_LAYOUT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        task_id=0,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        max_2d_position_embeddings: int = 1024,
        task_type_vocab_size: int = 3,
        type_vocab_size: int = 16,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        pool_act: str = "tanh",
        fuse: bool = False,
        image_feature_pool_shape=[7, 7, 256],
        layer_norm_eps=1e-12,
        use_cache=False,
        use_task_id=True,
        classifier_dropout=None,
        has_visual_segment_embedding=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.task_id = task_id
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.task_type_vocab_size = task_type_vocab_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pool_act = pool_act
        self.fuse = fuse
        self.image_feature_pool_shape = image_feature_pool_shape
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_task_id = use_task_id
        self.classifier_dropout = classifier_dropout
        self.has_visual_segment_embedding = has_visual_segment_embedding
