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
""" LayoutXLM model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["LAYOUTXLM_PRETRAINED_INIT_CONFIGURATION", "LayoutXLMConfig", "LAYOUTXLM_PRETRAINED_RESOURCE_FILES_MAP"]

LAYOUTXLM_PRETRAINED_INIT_CONFIGURATION = {
    "layoutxlm-base-uncased": {
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 0,
        "coordinate_size": 128,
        "eos_token_id": 2,
        "fast_qkv": False,
        "gradient_checkpointing": False,
        "has_relative_attention_bias": False,
        "has_spatial_attention_bias": False,
        "has_visual_segment_embedding": True,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "image_feature_pool_shape": [7, 7, 256],
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_2d_position_embeddings": 1024,
        "max_position_embeddings": 514,
        "max_rel_2d_pos": 256,
        "max_rel_pos": 128,
        "model_type": "layoutlmv2",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "output_past": True,
        "pad_token_id": 1,
        "shape_size": 128,
        "rel_2d_pos_bins": 64,
        "rel_pos_bins": 32,
        "type_vocab_size": 1,
        "vocab_size": 250002,
    },
    "vi-layoutxlm-base-uncased": {
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 0,
        "coordinate_size": 128,
        "eos_token_id": 2,
        "fast_qkv": False,
        "gradient_checkpointing": False,
        "has_relative_attention_bias": False,
        "has_spatial_attention_bias": False,
        "has_visual_segment_embedding": True,
        "use_visual_backbone": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "image_feature_pool_shape": [7, 7, 256],
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_2d_position_embeddings": 1024,
        "max_position_embeddings": 514,
        "max_rel_2d_pos": 256,
        "max_rel_pos": 128,
        "model_type": "layoutlmv2",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "output_past": True,
        "pad_token_id": 1,
        "shape_size": 128,
        "rel_2d_pos_bins": 64,
        "rel_pos_bins": 32,
        "type_vocab_size": 1,
        "vocab_size": 250002,
    },
}

LAYOUTXLM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "layoutxlm-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/layoutxlm_base/model_state.pdparams",
        "vi-layoutxlm-base-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/vi-layoutxlm-base-uncased/model_state.pdparams",
    }
}


class LayoutXLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LayoutXLMtModel`]. It is used to instantiate a
    LayoutXLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutXLM.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the SqueezeBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`SqueezeBertModel`].
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
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):

        pad_token_id (`int`, *optional*, defaults to 0):
            The ID of the token in the word embedding to use as padding.
        embedding_size (`int`, *optional*, defaults to 768):
            The dimension of the word embedding vectors.

        q_groups (`int`, *optional*, defaults to 4):
            The number of groups in Q layer.
        k_groups (`int`, *optional*, defaults to 4):
            The number of groups in K layer.
        v_groups (`int`, *optional*, defaults to 4):
            The number of groups in V layer.
        post_attention_groups (`int`, *optional*, defaults to 1):
            The number of groups in the first feed forward network layer.
        intermediate_groups (`int`, *optional*, defaults to 4):
            The number of groups in the second feed forward network layer.
        output_groups (`int`, *optional*, defaults to 4):
            The number of groups in the third feed forward network layer.

    Examples:

    ```python
    >>> from transformers import SqueezeBertConfig, SqueezeBertModel

    >>> # Initializing a SqueezeBERT configuration
    >>> configuration = SqueezeBertConfig()

    >>> # Initializing a model (with random weights) from the configuration above
    >>> model = SqueezeBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    Attributes: pretrained_config_archive_map (Dict[str, str]): A dictionary containing all the available pre-trained
    checkpoints.
    """
    pretrained_init_configuration = LAYOUTXLM_PRETRAINED_INIT_CONFIGURATION
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    model_type = "layoutxlm"

    def __init__(
        self,
        attention_probs_dropout_prob=0.1,
        bos_token_id=0,
        coordinate_size=128,
        eos_token_id=2,
        fast_qkv=False,
        gradient_checkpointing=False,
        has_relative_attention_bias=False,
        has_spatial_attention_bias=False,
        has_visual_segment_embedding=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        image_feature_pool_shape=[7, 7, 256],
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-05,
        max_2d_position_embeddings=1024,
        max_position_embeddings=514,
        max_rel_2d_pos=256,
        max_rel_pos=128,
        model_type="layoutlmv2",
        num_attention_heads=12,
        num_hidden_layers=12,
        output_past=True,
        pad_token_id=1,
        shape_size=128,
        rel_2d_pos_bins=64,
        rel_pos_bins=32,
        type_vocab_size=1,
        vocab_size=250002,
        with_pool="tanh",
        use_visual_backbone=False,
        **kwargs,
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
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_type = model_type
        self.with_pool = with_pool
        self.use_visual_backbone = use_visual_backbone
