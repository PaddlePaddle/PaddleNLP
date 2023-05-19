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
""" Ernie-CTM model configuration """
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "ERNIE_CTM_CONFIG",
    "ERNIE_CTM_PRETRAINED_INIT_CONFIGURATION",
    "ERNIE_CTM_PRETRAINED_RESOURCE_FILES_MAP",
    "ErnieCtmConfig",
]


ERNIE_CTM_CONFIG = {
    "vocab_size": 23000,
    "embedding_size": 128,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02,
    "pad_token_id": 0,
    "use_content_summary": True,
    "content_summary_index": 1,
    "cls_num": 2,
    "num_prompt_placeholders": 5,
    "prompt_vocab_ids": None,
}


ERNIE_CTM_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-ctm": ERNIE_CTM_CONFIG,
    "wordtag": ERNIE_CTM_CONFIG,
    "nptag": ERNIE_CTM_CONFIG,
}

ERNIE_CTM_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "ernie-ctm": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/ernie_ctm_v3.pdparams",
        "wordtag": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/wordtag_v3.pdparams",
        "nptag": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm/nptag_v3.pdparams",
    }
}


class ErnieCtmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieCtmModel`]. It is used to instantiate
    a Ernie-CTM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Ernie-CTM-base architecture.

    Configure objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documents from [`PretrainedConfig`] for more informations.


    Args:
        vocab_size (`int`, *optional*, defaults to 23000):
            Vocabulary size of the Ernie-CTM model. Defines the number of different tokens that can be represented by
            the `input_ids` passed when calling [`ErnieCtmModel`].
        embedding_size (`int` *optional*, defaults to 128):
            Dimensionality of vocabulary embeddings.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large.
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when call [`ErnieCtmModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_content_summary (`bool`, *optional*, defaults to True):
            Whether to use content summary token and content representation when inputs passed into [`ErnieCtmModel`].
        content_summary_index (`int`, *optional*, defaults to 1):
            If `use_content_summary` is set, content summary token position is defined by this argument.
        cls_num (`int`, *optional*, defaults to 2):
            Number of [CLS] token in model.
        num_prompt_placeholders (`int`, *optional*, defaults to 5):
            Number of maximum length of prompt answer.
        prompt_vocab_ids (`dict`, *optional*, defaults to None):
            Prompt vocabulary of decode procedure.
    """
    model_type = "ernie-ctm"
    pretrained_init_configuration = ERNIE_CTM_PRETRAINED_INIT_CONFIGURATION
    attribute_map = {"num_tag": "num_labels", "dropout": "classifier_dropout", "num_classes": "num_labels"}

    def __init__(
        self,
        vocab_size: int = 23000,
        embedding_size: int = 128,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        use_content_summary: bool = True,
        content_summary_index: int = 1,
        cls_num: int = 2,
        pad_token_id: int = 0,
        num_prompt_placeholders: int = 5,
        prompt_vocab_ids: set = None,
        **kwargs
    ):
        super(ErnieCtmConfig, self).__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_content_summary = use_content_summary
        self.content_summary_index = content_summary_index
        self.cls_num = cls_num
        self.num_prompt_placeholders = num_prompt_placeholders
        self.prompt_vocab_ids = prompt_vocab_ids
