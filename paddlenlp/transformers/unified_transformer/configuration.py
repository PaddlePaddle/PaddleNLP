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
"""UNIFIED_TRANSFORMER model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "UNIFIED_TRANSFORMER_PRETRAINED_INIT_CONFIGURATION",
    "UnifiedTransformerConfig",
    "UNIFIED_TRANSFORMER_PRETRAINED_RESOURCE_FILES_MAP",
]

UNIFIED_TRANSFORMER_PRETRAINED_INIT_CONFIGURATION = {
    "unified_transformer-12L-cn": {
        "vocab_size": 30004,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": True,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "unk_token_id": 0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "mask_token_id": 30000,
    },
    "unified_transformer-12L-cn-luge": {
        "vocab_size": 30004,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": True,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "unk_token_id": 0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "mask_token_id": 30000,
    },
    "plato-mini": {
        "vocab_size": 30001,
        "hidden_size": 768,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": True,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "unk_token_id": 0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "mask_token_id": 30000,
    },
    "plato-xl": {
        "vocab_size": 8001,
        "hidden_size": 3072,
        "num_hidden_layers": 72,
        "num_attention_heads": 32,
        "intermediate_size": 18432,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": True,
        "max_position_embeddings": 1024,
        "type_vocab_size": 3,
        "role_type_size": 128,
        "initializer_range": 0.02,
        "unk_token_id": 0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "mask_token_id": 8000,
    },
}


UNIFIED_TRANSFORMER_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "unified_transformer-12L-cn": "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/unified_transformer-12L-cn.pdparams",
        "unified_transformer-12L-cn-luge": "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/unified_transformer-12L-cn-luge.pdparams",
        "plato-mini": "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/plato-mini.pdparams",
        "plato-xl": "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/plato-xl.pdparams",
    }
}


class UnifiedTransformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UnifiedTransformerModel`]. It is used to
    instantiate a Unified TransformerModel model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Unified TransformerModel
    unified_transformer-12L-cn architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

        Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in :class:`UnifiedTransformerModel`.
            Also is the vocab size of token embedding matrix. Defaults to 30004.
        hidden_size (int, optional):
            Dimensionality of the embedding layers, encoder layers and pooler
            layer. Defaults to 768.
        num_hidden_layers (int, optional):
            The number of hidden layers in the encoder. Defaults to 12.
        num_attention_heads (int, optional):
            The number of heads in multi-head attention(MHA). Defaults to 12.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward layer in the encoder. Input
            tensors to feed-forward layers are firstly projected from
            `hidden_size` to `intermediate_size`, and then projected back to
            `hidden_size`. Typically `intermediate_size` is larger than
            `hidden_size`. Defaults to 3072.
        hidden_act (str, optional):
            The activation function in the feedforward network. Defaults to
            "gelu".
        hidden_dropout_prob(float, optional):
            The dropout probability used in pre-process and post-precess of MHA
            and FFN sub-layer. Defaults to 0.1.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MHA to drop some attention target.
            Defaults to 0.1.
        normalize_before (bool, optional):
            Indicate whether to put layer normalization into preprocessing of
            MHA and FFN sub-layers. If True, pre-process is layer normalization
            and post-precess includes dropout, residual connection. Otherwise,
            no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Defaults to True.
        max_position_embeddings (int, optional):
            The maximum length of input `position_ids`. Defaults to 512.
        type_vocab_size (int, optional):
            The size of the input `token_type_ids`. Defaults to 2.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal
                distributions. See
                :meth:`UnifiedTransformerPretrainedModel.init_weights` method
                for how weights are initialized in
                :class:`UnifiedTransformerModel`.
        unk_token_id (int, optional):
            The id of special token `unk_token`. Defaults to 0.
        pad_token_id (int, optional):
            The id of special token `pad_token`. Defaults to 0.
        bos_token_id (int, optional):
            The id of special token `bos_token`. Defaults to 1.
        eos_token_id (int, optional):
            The id of special token `eos_token`. Defaults to 2.
        mask_token_id (int, optional):
            The id of special token `mask_token`. Defaults to 30000.
    ```"""
    model_type = "unified_transformer"
    pretrained_init_configuration = UNIFIED_TRANSFORMER_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30004,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        normalize_before: bool = True,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        unk_token_id: int = 0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        mask_token_id: int = 30000,
        role_type_size: int = None,
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
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.normalize_before = normalize_before
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id
        self.role_type_size = role_type_size
