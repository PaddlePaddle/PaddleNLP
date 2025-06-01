# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""ModernBERT model configuration"""

from __future__ import annotations

from typing import Dict

from ..configuration_utils import PretrainedConfig

MODERNBERT_PRETRAINED_INIT_CONFIGURATION = {
    "modernbert-base": {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "geglu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 8192,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "sliding_window_size": 512,
        "tie_word_embeddings": True,
        "tensor_parallel_degree": 1,
        "rope_theta": 10000.0,
    },
    "modernbert-large": {
        "vocab_size": 30522,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "geglu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 8192,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "sliding_window_size": 512,
        "tie_word_embeddings": True,
        "tensor_parallel_degree": 1,
        "rope_theta": 10000.0,
    },
}

MODERNBERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "modernbert-base": "https://bj.bcebos.com/paddlenlp/models/transformers/modernbert/modernbert-base.pdparams",
        "modernbert-large": "https://bj.bcebos.com/paddlenlp/models/transformers/modernbert/modernbert-large.pdparams",
    }
}


class ModernBertConfig(PretrainedConfig):
    """Configuration class for ModernBERT model."""

    model_type = "modernbert"
    attribute_map: Dict[str, str] = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = MODERNBERT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "geglu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 8192,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        sliding_window_size: int = 512,
        position_embedding_type: str = "rotary",
        use_cache: bool = True,
        classifier_dropout: float | None = None,
        tie_word_embeddings: bool = True,
        tensor_parallel_degree: int = 1,
        rope_theta: float = 10000.0,
        **kwargs,
    ) -> None:
        """Initialize the ModernBERT configuration.

        Args:
            vocab_size: Vocabulary size of the ModernBERT model.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward) layer.
            hidden_act: The non-linear activation function in the encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected layers.
            attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
            max_position_embeddings: The maximum sequence length that this model might ever be used with.
            type_vocab_size: The vocabulary size of the `token_type_ids`.
            initializer_range: The standard deviation of the truncated_normal_initializer.
            layer_norm_eps: The epsilon used by the layer normalization layers.
            pad_token_id: The value used to pad input_ids.
            sliding_window_size: Size of the sliding window for attention computation.
            position_embedding_type: Type of position embedding. Choose from "absolute" or "rotary".
            use_cache: Whether to use the model cache to speed up decoding.
            classifier_dropout: Dropout probability for the classification head.
            tie_word_embeddings: Whether to tie input and output embeddings.
            tensor_parallel_degree: Degree of tensor parallelism for distributed training.
            rope_theta: Base value for rotary position embedding.
        """
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
        self.sliding_window_size = sliding_window_size
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.tensor_parallel_degree = tensor_parallel_degree
        self.rope_theta = rope_theta
