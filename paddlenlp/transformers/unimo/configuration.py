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
""" UNIMO model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["UNIMO_PRETRAINED_INIT_CONFIGURATION", "UNIMOConfig", "UNIMO_PRETRAINED_RESOURCE_FILES_MAP"]

UNIMO_PRETRAINED_INIT_CONFIGURATION = {
    "unimo-text-1.0": {
        "vocab_size": 18000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": False,
        "max_position_embeddings": 513,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "unk_token_id": 17963,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "mask_token_id": 3,
    },
    "unimo-text-1.0-lcsts-new": {
        "vocab_size": 18000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": False,
        "max_position_embeddings": 513,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "unk_token_id": 17963,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "mask_token_id": 3,
    },
    "unimo-text-1.0-summary": {
        "vocab_size": 18000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": False,
        "max_position_embeddings": 513,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "unk_token_id": 17963,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "mask_token_id": 3,
    },
    "unimo-text-1.0-large": {
        "vocab_size": 12800,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": False,
        "max_position_embeddings": 512,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "unk_token_id": 12088,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "mask_token_id": 3,
    },
    "unimo-text-1.0-dureader_qg": {
        "vocab_size": 18000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": False,
        "max_position_embeddings": 513,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "unk_token_id": 17963,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "mask_token_id": 3,
    },
    "unimo-text-1.0-question-generation": {
        "vocab_size": 18000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": False,
        "max_position_embeddings": 513,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "unk_token_id": 17963,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "mask_token_id": 3,
    },
    "unimo-text-1.0-question-generation-full_domain": {
        "vocab_size": 18000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": False,
        "max_position_embeddings": 513,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "unk_token_id": 17963,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "mask_token_id": 3,
    },
    "unimo-text-1.0-question-generation-dureader_qg": {
        "vocab_size": 18000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "normalize_before": False,
        "max_position_embeddings": 513,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "unk_token_id": 17963,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "mask_token_id": 3,
    },
}

UNIMO_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "unimo-text-1.0": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0.pdparams",
        "unimo-text-1.0-lcsts-new": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-lcsts-new.pdparams",
        "unimo-text-1.0-large": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-large.pdparams",
        "unimo-text-1.0-summary": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-summary.pdparams",
        "unimo-text-1.0-dureader_qg": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-dureader_qg.pdparams",
        "unimo-text-1.0-question-generation": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-question-generation.pdparams",
        "unimo-text-1.0-question-generation-v2": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-question-generation-full_domain.pdparams",
        "unimo-text-1.0-question-generation-dureader_qg": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-question-generation-dureader_qg.pdparams",
    }
}


class UNIMOConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UNIMOModel`]. It is used to
    instantiate a UNIMO model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UNIMO
    unimo-text-1.0 architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `UNIMOModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `UNIMOModel`.
            Defaults to `18000`.
        hidden_size (int, optional):
            Dimensionality of the embedding layers and encoder layers. Defaults to `768`.
        num_hidden_layers (int, optional):
            The number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob(float, optional):
            The dropout probability used in pre-process and post-precess of MHA
            and FFN sub-layer. Defaults to 0.1.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        normalize_before (bool, optional):
            Indicate whether to put layer normalization into preprocessing of
            MHA and FFN sub-layers. If True, pre-process is layer normalization
            and post-precess includes dropout, residual connection. Otherwise,
            no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Defaults to `True`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.UNIMOModel`.
            Defaults to `2`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`UNIMOPretrainedModel._init_weights()` for how weights are initialized in `UNIMOModel`.

        unk_token_id (int, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` in order to be converted to an ID.
            Defaults to `17963`.
        pad_token_id (int, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to `0`.
        bos_token_id (int, optional):
            A special token representing the beginning of a sequence that was used during pretraining.
            Defaults to `1`.
        eos_token_id (int, optional):
            A special token representing the end of a sequence that was used during pretraining.
            Defaults to `3`.
        mask_token_id (int, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to `3`.
    ```"""
    model_type = "unimo"
    pretrained_init_configuration = UNIMO_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 18000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "relu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        normalize_before: int = False,
        max_position_embeddings: int = 513,
        type_vocab_size: int = 4,
        initializer_range: float = 0.02,
        unk_token_id: int = 17963,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 3,
        mask_token_id: int = 3,
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
