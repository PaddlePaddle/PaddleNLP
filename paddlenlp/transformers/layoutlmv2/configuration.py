# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" LayoutLMv2 model configuration"""

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["LAYOUTLMV2_PRETRAINED_INIT_CONFIGURATION", "LayoutLMv2Config", "LAYOUTLMV2_PRETRAINED_RESOURCE_FILES_MAP"]

LAYOUTLMV2_PRETRAINED_INIT_CONFIGURATION = {
    "layoutlmv2-base-uncased": {
        "attention_probs_dropout_prob": 0.1,
        "coordinate_size": 128,
        "fast_qkv": True,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "image_feature_pool_shape": [7, 7, 256],
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_2d_position_embeddings": 1024,
        "max_position_embeddings": 512,
        "max_rel_2d_pos": 256,
        "max_rel_pos": 128,
        "model_type": "layoutlmv2",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "output_past": True,
        "pad_token_id": 0,
        "shape_size": 128,
        "rel_2d_pos_bins": 64,
        "rel_pos_bins": 32,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "has_relative_attention_bias": True,
        "has_spatial_attention_bias": True,
        "has_visual_segment_embedding": False,
        "use_visual_backbone": True,
    },
    "layoutlmv2-large-uncased": {
        "attention_probs_dropout_prob": 0.1,
        "coordinate_size": 171,
        "fast_qkv": False,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "image_feature_pool_shape": [7, 7, 256],
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-12,
        "max_2d_position_embeddings": 1024,
        "max_position_embeddings": 512,
        "max_rel_2d_pos": 256,
        "max_rel_pos": 128,
        "model_type": "layoutlmv2",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "output_past": True,
        "pad_token_id": 0,
        "shape_size": 170,
        "rel_2d_pos_bins": 64,
        "rel_pos_bins": 32,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "has_relative_attention_bias": True,
        "has_spatial_attention_bias": True,
        "has_visual_segment_embedding": False,
        "use_visual_backbone": True,
    },
    "vi-layoutlmv2-base-uncased": {
        "attention_probs_dropout_prob": 0.1,
        "coordinate_size": 128,
        "fast_qkv": True,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "image_feature_pool_shape": [7, 7, 256],
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_2d_position_embeddings": 1024,
        "max_position_embeddings": 512,
        "max_rel_2d_pos": 256,
        "max_rel_pos": 128,
        "model_type": "layoutlmv2",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "output_past": True,
        "pad_token_id": 0,
        "shape_size": 128,
        "rel_2d_pos_bins": 64,
        "rel_pos_bins": 32,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "has_relative_attention_bias": True,
        "has_spatial_attention_bias": True,
        "has_visual_segment_embedding": False,
        "use_visual_backbone": False,
    },
}

LAYOUTLMV2_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "layoutlmv2-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/layoutlmv2/layoutlmv2-base-uncased/model_state.pdparams",
        "layoutlmv2-large-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/layoutlmv2/layoutlmv2-large-uncased/model_state.pdparams",
        "vi-layoutlmv2-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/layoutlmv2/vi-layoutlmv2-base-uncased/model_state.pdparams",
    }
}


class LayoutLMv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`LayoutLMv2Model`]. It is used to instantiate an LayoutLMv2 Model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the LayoutLMv2 layoutlmv2-base-uncased architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, optional, defaults to 21128):
            Vocabulary size of the NEZHA model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method of [`NezhaModel`].
        embedding_size (`int`, optional, defaults to 128):
            Dimensionality of vocabulary embeddings.
        hidden_size (`int`, optional, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, optional, defaults to 3072):
            The dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, optional, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_dropout_prob (`float`, optional, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, optional, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, optional, defaults to 2):
            The vocabulary size of the *token_type_ids* passed into [`NezhaModel`].
        initializer_range (`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout (`float`, optional, defaults to 0.1):
            The dropout ratio for attached classifiers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    Example:
    ```python
    >>> from paddlenlp.transformers import NeZhaConfig, NeZhaModel
    >>> # Initializing an Nezha configuration
    >>> configuration = NeZhaConfig()
    >>> # Initializing a model (with random weights) from the Nezha-base style configuration model
    >>> model = NeZhaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = LAYOUTLMV2_PRETRAINED_INIT_CONFIGURATION
    model_type = "layoutlmv2"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_2d_position_embeddings=1024,
        max_rel_pos=128,
        max_rel_2d_pos=256,
        rel_pos_bins=32,
        rel_2d_pos_bins=64,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        coordinate_size=128,
        shape_size=128,
        image_feature_pool_shape=[7, 7, 256],
        fast_qkv=True,
        has_relative_attention_bias=True,
        has_spatial_attention_bias=True,
        has_visual_segment_embedding=False,
        output_past=True,
        gradient_checkpointing=False,
        classifier_dropout=0.1,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        use_cache=True,
        with_pool="tanh",
        use_visual_backbone=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_rel_pos = max_rel_pos
        self.max_rel_2d_pos = max_rel_2d_pos
        self.rel_pos_bins = rel_pos_bins
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.image_feature_pool_shape = image_feature_pool_shape
        self.fast_qkv = fast_qkv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding
        self.output_past = output_past
        self.gradient_checkpointing = gradient_checkpointing
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_cache = use_cache
        self.with_pool = with_pool
        self.use_visual_backbone = use_visual_backbone
