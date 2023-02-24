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
""" MBart model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "MegatronBert_PRETRAINED_INIT_CONFIGURATION",
    "MegatronBert_PRETRAINED_RESOURCE_FILES_MAP",
    "MegatronBertConfig",
]

MegatronBert_PRETRAINED_INIT_CONFIGURATION = {
    "megatronbert-cased": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 29056,
        "pad_token_id": 0,
    },
    "megatronbert-uncased": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 30592,
        "pad_token_id": 0,
    },
}

MegatronBert_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "megatronbert-cased": "http://bj.bcebos.com/paddlenlp/models/transformers/megatron-bert/megatronbert-cased/model_state.pdparams",
        "megatronbert-uncased": "http://bj.bcebos.com/paddlenlp/models/transformers/megatron-bert/megatronbert-uncased/model_state.pdparams",
    }
}


class MegatronBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MegatronBertModel`]. It is used to instantiate a
    MEGATRON_BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MEGATRON_BERT
    [nvidia/megatron-bert-uncased-345m](https://huggingface.co/nvidia/megatron-bert-uncased-345m) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `MegatronBertModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `MegatronBert`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer and pooler layer. Defaults to `1024`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`.
            Defaults to `2`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `16`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `24`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `4096`.
        position_embedding_type (str, optional):
            Type of position embedding. Defaults to "absolute"
        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`MegatronBertPretrainedModel.init_weights()` for how weights are initialized in `MegatronBertModel`.

    """
    model_type = "megatronbert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=29056,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        # use_cache=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        # self.use_cache = use_cache
