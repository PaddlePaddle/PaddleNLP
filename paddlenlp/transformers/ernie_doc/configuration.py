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
""" DalleBart model configuration"""
from __future__ import annotations

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["ERNIE_DOC_PRETRAINED_INIT_CONFIGURATION", "ErnieDocConfig", "ERNIE_DOC_PRETRAINED_RESOURCE_FILES_MAP"]

ERNIE_DOC_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-doc-base-en": {
        "attention_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "relu_dropout": 0.0,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "vocab_size": 50265,
        "memory_len": 128,
        "epsilon": 1e-12,
        "pad_token_id": 1,
    },
    "ernie-doc-base-zh": {
        "attention_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "relu_dropout": 0.0,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "task_type_vocab_size": 3,
        "vocab_size": 28000,
        "memory_len": 128,
        "epsilon": 1e-12,
        "pad_token_id": 0,
    },
}

ERNIE_DOC_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "ernie-doc-base-en": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-en/ernie-doc-base-en.pdparams",
        "ernie-doc-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-doc-base-zh/ernie-doc-base-zh.pdparams",
    }
}


class ErnieDocConfig(PretrainedConfig):
    """
    The bare ERNIE-Doc Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        num_hidden_layers (int):
            The number of hidden layers in the Transformer encoder.
        num_attention_heads (int):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_size (int):
            Dimensionality of the embedding layers, encoder layers and pooler layer.
        hidden_dropout_prob (int):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_dropout_prob (int):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
        relu_dropout (int):
            The dropout probability of FFN.
        hidden_act (str):
            The non-linear activation function of FFN.
        memory_len (int):
            The number of tokens to cache. If not 0, the last `memory_len` hidden states
            in each layer will be cached into memory.
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ErnieDocModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ErnieDocModel`.
        max_position_embeddings (int):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        task_type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`. Defaults to `3`.
        normalize_before (bool, optional):
            Indicate whether to put layer normalization into preprocessing of MHA and FFN sub-layers.
            If True, pre-process is layer normalization and post-precess includes dropout,
            residual connection. Otherwise, no pre-process and post-precess includes dropout,
            residual connection, layer normalization. Defaults to `False`.
        epsilon (float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for
            initializing layer normalization layers. Defaults to `1e-5`.
        rel_pos_params_sharing (bool, optional):
            Whether to share the relative position parameters.
            Defaults to `False`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer for initializing all weight matrices.
            Defaults to `0.02`.
        pad_token_id (int, optional):
            The token id of [PAD] token whose parameters won't be updated when training.
            Defaults to `0`.
        cls_token_idx (int, optional):
            The token id of [CLS] token. Defaults to `-1`.
    """

    model_type = "ernie_doc"
    pretrained_init_configuration = ERNIE_DOC_PRETRAINED_INIT_CONFIGURATION
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
        self,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        relu_dropout=0.0,
        hidden_act="gelu",
        memory_len=128,
        vocab_size=28000,
        max_position_embeddings=512,
        task_type_vocab_size=3,
        normalize_before=False,
        epsilon=1e-5,
        rel_pos_params_sharing=False,
        initializer_range=0.02,
        pad_token_id=0,
        cls_token_idx=-1,
        **kwargs
    ):
        super(ErnieDocConfig, self).__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.attention_dropout_prob = attention_dropout_prob
        self.relu_dropout = relu_dropout
        self.hidden_act = hidden_act
        self.memory_len = memory_len
        self.hidden_size = hidden_size
        self.task_type_vocab_size = task_type_vocab_size
        self.normalize_before = normalize_before
        self.epsilon = epsilon
        self.rel_pos_params_sharing = rel_pos_params_sharing
        self.cls_token_idx = cls_token_idx
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
