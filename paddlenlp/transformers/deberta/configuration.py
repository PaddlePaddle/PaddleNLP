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

""" DeBERTa model configuration"""
from __future__ import annotations

from typing import Dict, List

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["DEBERTA_PRETRAINED_INIT_CONFIGURATION", "DebertaConfig", "DEBERTA_PRETRAINED_RESOURCE_FILES_MAP"]

DEBERTA_PRETRAINED_INIT_CONFIGURATION = {
    "deberta-base": {
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-07,
        "max_position_embeddings": 512,
        "max_relative_positions": -1,
        "model_type": "deberta",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "output_hidden_states": True,
        "pad_token_id": 0,
        "pos_att_type": ["c2p", "p2c"],
        "position_biased_input": False,
        "relative_attention": True,
        "type_vocab_size": 0,
        "vocab_size": 50265,
    },
}

DEBERTA_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "microsoft/deberta-base": "https://paddlenlp.bj.bcebos.com/models/community/microsoft/deberta-base/model_state.pdparams"
    }
}


class DebertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeBERTaModel`] . It is used to
    instantiate a DeBERTa model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DeBERTa
    DeBERTa-v1-base architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the DeBERTa model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling [`DeBERTaModel`].
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        embedding_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the embedding layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 0):
            The vocabulary size of the :obj:`token_type_ids` passed when calling [`DeBERTaModel`].
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether add position bias to the input embeddings.
        pos_att_type (:obj:`List[str]`, `optional`, defaults to :obj:`["p2c", "c2p"]`):
            The type of relative position attention. It should be a subset of `["p2c", "c2p", "p2p"]`.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model returns attentions weights.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether the model returns all hidden-states.
        relative_attention (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether use relative position encoding.

    Examples:

    ```python
    >>> from paddlenlp.transformers import DeBERTaModel, DeBERTaConfig

    >>> # Initializing a DeBERTa DeBERTa-base style configuration
    >>> configuration = DeBERTaConfig()

    >>> # Initializing a model from the DeBERTa-base-uncased style configuration
    >>> model = DeBERTaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "deberta"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = DEBERTA_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 50265,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-7,
        pad_token_id: int = 0,
        position_biased_input: bool = False,
        pos_att_type: List[str] = ["p2c", "c2p"],
        output_attentions: bool = False,
        output_hidden_states: bool = True,
        relative_attention: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = kwargs.get("embedding_size", hidden_size)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_biased_input = position_biased_input
        self.pos_att_type = pos_att_type
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.relative_attention = relative_attention
        self.pad_token_id = pad_token_id
