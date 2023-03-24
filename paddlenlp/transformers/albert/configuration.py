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
""" Albert model configuration"""
from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

__all__ = ["ALBERT_PRETRAINED_INIT_CONFIGURATION", "AlbertConfig", "ALBERT_PRETRAINED_RESOURCE_FILES_MAP"]

ALBERT_PRETRAINED_INIT_CONFIGURATION = {
    "albert-base-v1": {
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000,
    },
    "albert-large-v1": {
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_groups": 1,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000,
    },
    "albert-xlarge-v1": {
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 8192,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_groups": 1,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000,
    },
    "albert-xxlarge-v1": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 16384,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 64,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000,
    },
    "albert-base-v2": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000,
    },
    "albert-large-v2": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_groups": 1,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000,
    },
    "albert-xlarge-v2": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0,
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 8192,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_groups": 1,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000,
    },
    "albert-xxlarge-v2": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0,
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 16384,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 64,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000,
    },
    "albert-chinese-tiny": {
        "attention_probs_dropout_prob": 0.0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 312,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 1248,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_groups": 1,
        "num_hidden_layers": 4,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128,
    },
    "albert-chinese-small": {
        "attention_probs_dropout_prob": 0.0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 384,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 1536,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_groups": 1,
        "num_hidden_layers": 6,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128,
    },
    "albert-chinese-base": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128,
    },
    "albert-chinese-large": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_groups": 1,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128,
    },
    "albert-chinese-xlarge": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0,
        "hidden_size": 2048,
        "initializer_range": 0.014,
        "inner_group_num": 1,
        "intermediate_size": 8192,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_groups": 1,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128,
    },
    "albert-chinese-xxlarge": {
        "attention_probs_dropout_prob": 0,
        "bos_token_id": 2,
        "embedding_size": 128,
        "eos_token_id": 3,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0,
        "hidden_size": 4096,
        "initializer_range": 0.01,
        "inner_group_num": 1,
        "intermediate_size": 16384,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128,
    },
}

ALBERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "albert-base-v1": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-base-v1.pdparams",
        "albert-large-v1": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-large-v1.pdparams",
        "albert-xlarge-v1": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-xlarge-v1.pdparams",
        "albert-xxlarge-v1": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-xxlarge-v1.pdparams",
        "albert-base-v2": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-base-v2.pdparams",
        "albert-large-v2": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-large-v2.pdparams",
        "albert-xlarge-v2": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-xlarge-v2.pdparams",
        "albert-xxlarge-v2": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-xxlarge-v2.pdparams",
        "albert-chinese-tiny": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-tiny.pdparams",
        "albert-chinese-small": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-small.pdparams",
        "albert-chinese-base": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-base.pdparams",
        "albert-chinese-large": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-large.pdparams",
        "albert-chinese-xlarge": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-xlarge.pdparams",
        "albert-chinese-xxlarge": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-xxlarge.pdparams",
    }
}


class AlbertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AlbertModel`]. It is used to instantiate
    an ALBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ALBERT
    [albert-xxlarge-v2](https://huggingface.co/albert-xxlarge-v2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `AlbertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `AlbertModel`.
            Defaults to `30000`.
        embedding_size (int, optional):
            Dimensionality of the embedding layer. Defaults to `128`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        inner_group_num (int, optional):
            Number of hidden groups in the Transformer encoder. Defaults to `1`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
        inner_group_num (int, optional):
            Number of inner groups in a hidden group. Default to `1`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0`.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for attached classifiers.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`. Defaults to `12`.

        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `ElectraModel`.

        layer_norm_eps(float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for initializing layer normalization layers.
            A small value to the variance added to the normalization layer to prevent division by zero.
            Default to `1e-12`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary. Defaults to `0`.
        add_pooling_layer(bool, optional):
            Whether or not to add the pooling layer. Default to `False`.
    Example:
    ```python
    >>> from paddlenlp.transformers import AlbertConfig, AlbertModel
    >>> # Initializing an ALBERT style configuration
    >>> configuration = AlbertConfig()
    >>> # Initializing a model (with random weights) from the ALBERT-base style configuration
    >>> model = AlbertModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = ALBERT_PRETRAINED_INIT_CONFIGURATION
    model_type = "albert"

    def __init__(
        self,
        vocab_size=30000,
        embedding_size=128,
        hidden_size=768,
        num_hidden_layers=12,
        num_hidden_groups=1,
        num_attention_heads=12,
        intermediate_size=3072,
        inner_group_num=1,
        hidden_act="gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        add_pooling_layer=True,
        classifier_dropout_prob=0.1,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.inner_group_num = inner_group_num
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.add_pooling_layer = add_pooling_layer
