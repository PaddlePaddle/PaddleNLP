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
""" MBart model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "REMBERT_PRETRAINED_INIT_CONFIGURATION",
    "REMBERT_PRETRAINED_RESOURCE_FILES_MAP",
    "RemBertConfig",
]

REMBERT_PRETRAINED_INIT_CONFIGURATION = {
    "rembert": {
        "attention_probs_dropout_prob": 0,
        "input_embedding_size": 256,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "hidden_size": 1152,
        "initializer_range": 0.02,
        "intermediate_size": 4608,
        "max_position_embeddings": 512,
        "num_attention_heads": 18,
        "num_hidden_layers": 32,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 250300,
        "layer_norm_eps": 1e-12,
    }
}

REMBERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "rembert": "https://bj.bcebos.com/paddlenlp/models/transformers/rembert/model_state.pdparams",
    }
}


class RemBertConfig(PretrainedConfig):
    r"""
    Args:
    vocab_size (int):
        Vocabulary size of `inputs_ids` in `RemBertModel`. Also is the vocab size of token embedding matrix.
        Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `RemBertModel`.
    input_embedding_size (int, optional):
        Dimensionality of the embedding layer. Defaults to `256`.
    hidden_size (int, optional):
        Dimensionality of the encoder layer and pooler layer. Defaults to `1152`.
    num_hidden_layers (int, optional):
        Number of hidden layers in the Transformer encoder. Defaults to `32`.
    num_attention_heads (int, optional):
        Number of attention heads for each attention layer in the Transformer encoder.
        Defaults to `18`.
    intermediate_size (int, optional):
        Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
        to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
        and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
        Defaults to `3072`.
    hidden_act (str, optional):
        The non-linear activation function in the feed-forward layer.
        ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
        are supported. Defaults to `"gelu"`.
    hidden_dropout_prob (float, optional):
        The dropout probability for all fully connected layers in the embeddings and encoder.
        Defaults to `0.1`.
    attention_probs_dropout_prob (float, optional):
        The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
        Defaults to `0.1`.
    max_position_embeddings (int, optional):
        The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
        sequence. Defaults to `512`.
    type_vocab_size (int, optional):
        The vocabulary size of `token_type_ids`.
        Defaults to `16`.

    initializer_range (float, optional):
        The standard deviation of the normal initializer.
        Defaults to 0.02.

        .. note::
            A normal_initializer initializes weight matrices as normal distributions.
            See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `BertModel`.

    pad_token_id (int, optional):
        The index of padding token in the token vocabulary.
        Defaults to `0`.
    """

    model_type = "rembert"

    def __init__(
        self,
        vocab_size=250300,
        input_embedding_size=256,
        hidden_size=1152,
        num_hidden_layers=32,
        num_attention_heads=18,
        intermediate_size=4608,
        hidden_act="gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_embedding_size = input_embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.layer_norm_eps = layer_norm_eps
