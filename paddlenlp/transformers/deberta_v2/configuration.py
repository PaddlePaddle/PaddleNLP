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

from typing import Dict

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["DEBERTA_V2_PRETRAINED_INIT_CONFIGURATION", "DebertaV2Config", "DEBERTA_V2_PRETRAINED_RESOURCE_FILES_MAP"]

DEBERTA_V2_PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v3-base": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "relative_attention": True,
        "position_buckets": 256,
        "norm_rel_ebd": "layer_norm",
        "share_att_key": True,
        "pos_att_type": ["p2c", "c2p"],
        "layer_norm_eps": 1e-7,
        "max_relative_positions": -1,
        "position_biased_input": False,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 0,
        "vocab_size": 128100,
    },
    "microsoft/deberta-v3-large": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "relative_attention": True,
        "position_buckets": 256,
        "norm_rel_ebd": "layer_norm",
        "share_att_key": True,
        "pos_att_type": ["p2c", "c2p"],
        "layer_norm_eps": 1e-7,
        "max_relative_positions": -1,
        "position_biased_input": False,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 0,
        "vocab_size": 128100,
    },
    "microsoft/deberta-v2-xlarge": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1536,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 512,
        "relative_attention": True,
        "position_buckets": 256,
        "norm_rel_ebd": "layer_norm",
        "share_att_key": True,
        "pos_att_type": ["p2c", "c2p"],
        "layer_norm_eps": 1e-7,
        "conv_kernel_size": 3,
        "conv_act": "gelu",
        "max_relative_positions": -1,
        "position_biased_input": False,
        "num_attention_heads": 24,
        "attention_head_size": 64,
        "num_hidden_layers": 24,
        "type_vocab_size": 0,
        "vocab_size": 128100,
    },
    "deepset/deberta-v3-large-squad2": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "language": "english",
        "layer_norm_eps": 1e-07,
        "max_position_embeddings": 512,
        "max_relative_positions": -1,
        "norm_rel_ebd": "layer_norm",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "pooler_dropout": 0,
        "pooler_hidden_act": "gelu",
        "pooler_hidden_size": 1024,
        "pos_att_type": ["p2c", "c2p"],
        "position_biased_input": False,
        "position_buckets": 256,
        "relative_attention": True,
        "share_att_key": True,
        "summary_activation": "tanh",
        "summary_last_dropout": 0,
        "summary_type": "first",
        "summary_use_proj": False,
        "type_vocab_size": 0,
        "vocab_size": 128100,
    },
}

DEBERTA_V2_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "microsoft/deberta-v2-xlarge": "https://paddlenlp.bj.bcebos.com/models/community/microsoft/deberta-v2-xlarge/model_state.pdparams",
        "microsoft/deberta-v3-base": "https://paddlenlp.bj.bcebos.com/models/community/microsoft/deberta-v3-base/model_state.pdparams",
        "microsoft/deberta-v3-large": "https://paddlenlp.bj.bcebos.com/models/community/microsoft/deberta-v3-large/model_state.pdparams",
        "deepset/deberta-v3-large-squad2": "https://paddlenlp.bj.bcebos.com/models/community/deepset/deberta-v3-large-squad2/model_state.pdparams",
    }
}


class DebertaV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeBERTaV2Model`] . It is used to
    instantiate a DeBERTaV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DeBERTa
    DeBERTa-v2-xlarge architecture.

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

    >>> # Initializing a DeBERTa DeBERTa-v2-base style configuration
    >>> configuration = DeBERTaV2Config()

    >>> # Initializing a model from the DeBERTa-base-uncased style configuration
    >>> model = DeBERTaV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "deberta-v2"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = DEBERTA_V2_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size=128100,
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        intermediate_size=6144,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=False,
        max_relative_positions=-1,
        pad_token_id=0,
        position_biased_input=True,
        pos_att_type=None,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        share_attn_key=True,
        output_hidden_states=True,
        output_attentions=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
        self.share_attn_key = share_attn_key
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
