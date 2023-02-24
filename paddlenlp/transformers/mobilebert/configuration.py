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
""" MobileBert model configuration"""
from __future__ import annotations

from ..configuration_utils import PretrainedConfig

__all__ = ["MOBILEBERT_PRETRAINED_INIT_CONFIGURATION", "MobileBertConfig", "MOBILEBERT_PRETRAINED_RESOURCE_FILES_MAP"]

MOBILEBERT_PRETRAINED_INIT_CONFIGURATION = {
    "mobilebert-uncased": {
        "attention_probs_dropout_prob": 0.1,
        "classifier_activation": False,
        "embedding_size": 128,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 512,
        "initializer_range": 0.02,
        "intermediate_size": 512,
        "intra_bottleneck_size": 128,
        "key_query_shared_bottleneck": True,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "mobilebert",
        "normalization_type": "no_norm",
        "num_attention_heads": 4,
        "num_feedforward_networks": 4,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "transformers_version": "4.6.0.dev0",
        "trigram_input": True,
        "true_hidden_size": 128,
        "type_vocab_size": 2,
        "use_bottleneck": True,
        "use_bottleneck_attention": False,
        "vocab_size": 30522,
    }
}
MOBILEBERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "mobilebert-uncased": "https://bj.bcebos.com/paddlenlp/models/transformers/mobilebert/mobilebert-uncased/model_state.pdparams"
    }
}


class MobileBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~paddlenlp.transformers.MobileBertModel`.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the MobileBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MobileBertModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 512):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`MobileBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The ID of the token in the word embedding to use as padding.
        embedding_size (`int`, *optional*, defaults to 128):
            The dimension of the word embedding vectors.
        trigram_input (`bool`, *optional*, defaults to `True`):
            Use a convolution of trigram as input.
        use_bottleneck (`bool`, *optional*, defaults to `True`):
            Whether to use bottleneck in BERT.
        intra_bottleneck_size (`int`, *optional*, defaults to 128):
            Size of bottleneck layer output.
        use_bottleneck_attention (`bool`, *optional*, defaults to `False`):
            Whether to use attention inputs from the bottleneck transformation.
        key_query_shared_bottleneck (`bool`, *optional*, defaults to `True`):
            Whether to use the same linear transformation for query&key in the bottleneck.
        num_feedforward_networks (`int`, *optional*, defaults to 4):
            Number of FFNs in a block.
        normalization_type (`str`, *optional*, defaults to `"no_norm"`):
            The normalization type in MobileBERT.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:
    ```python
    >>> from paddlenlp.transformers import MobileBertConfig, MobileBertModel
    >>> # Initializing a MobileBERT configuration
    >>> configuration = MobileBertConfig()
    >>> # Initializing a model (with random weights) from the configuration above
    >>> model = MobileBertModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "mobilebert"
    pretrained_init_configuration = MOBILEBERT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = MOBILEBERT_PRETRAINED_RESOURCE_FILES_MAP
    keys_to_ignore_at_inference = ["pooled_output"]

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=24,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        embedding_size=128,
        true_hidden_size=128,
        normalization_type="no_norm",
        use_bottleneck=True,
        use_bottleneck_attention=False,
        intra_bottleneck_size=128,
        key_query_shared_bottleneck=True,
        num_feedforward_networks=4,
        trigram_input=True,
        classifier_activation=False,
        classifier_dropout=None,
        add_pooling_layer=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
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
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.embedding_size = embedding_size
        self.true_hidden_size = true_hidden_size
        self.normalization_type = normalization_type
        self.use_bottleneck = use_bottleneck
        self.use_bottleneck_attention = use_bottleneck_attention
        self.intra_bottleneck_size = intra_bottleneck_size
        self.key_query_shared_bottleneck = key_query_shared_bottleneck
        self.num_feedforward_networks = num_feedforward_networks
        self.trigram_input = trigram_input
        self.classifier_activation = classifier_activation
        if self.use_bottleneck:
            self.true_hidden_size = intra_bottleneck_size
        else:
            self.true_hidden_size = hidden_size

        self.classifier_dropout = classifier_dropout
        self.add_pooling_layer = add_pooling_layer
