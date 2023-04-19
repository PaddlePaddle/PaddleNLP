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
""" funnel model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "FUNNEL_PRETRAINED_INIT_CONFIGURATION",
    "FUNNEL_PRETRAINED_RESOURCE_FILES_MAP",
    "FunnelConfig",
]

FUNNEL_PRETRAINED_INIT_CONFIGURATION = {
    "funnel-transformer/small": {},  # B4-4-4H768
    "funnel-transformer/small-base": {},  # B4-4-4H768, no decoder
    "funnel-transformer/medium": {},  # B6-3x2-3x2H768
    "funnel-transformer/medium-base": {},  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate": {},  # B6-6-6H768
    "funnel-transformer/intermediate-base": {},  # B6-6-6H768, no decoder
    "funnel-transformer/large": {},  # B8-8-8H1024
    "funnel-transformer/large-base": {},  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base": {},  # B10-10-10H1024
    "funnel-transformer/xlarge": {},  # B10-10-10H1024, no decoder
}

FUNNEL_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "funnel-transformer/small": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small/model_state.pdparams",
        "funnel-transformer/small-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small-base/model_state.pdparams",
        "funnel-transformer/medium": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium/model_state.pdparams",
        "funnel-transformer/medium-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium-base/model_state.pdparams",
        "funnel-transformer/intermediate": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate/model_state.pdparams",
        "funnel-transformer/intermediate-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate-base/model_state.pdparams",
        "funnel-transformer/large": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large/model_state.pdparams",
        "funnel-transformer/large-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large-base/model_state.pdparams",
        "funnel-transformer/xlarge-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge-base/model_state.pdparams",
        "funnel-transformer/xlarge": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge/model_state.pdparams",
    },
    "model_config": {
        "funnel-transformer/small": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small/model_config.json",
        "funnel-transformer/small-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small-base/model_config.json",
        "funnel-transformer/medium": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium/model_config.json",
        "funnel-transformer/medium-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium-base/model_config.json",
        "funnel-transformer/intermediate": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate/model_config.json",
        "funnel-transformer/intermediate-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate-base/model_config.json",
        "funnel-transformer/large": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large/model_config.json",
        "funnel-transformer/large-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large-base/model_config.json",
        "funnel-transformer/xlarge-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge-base/model_config.json",
        "funnel-transformer/xlarge": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge/model_config.json",
    },
}

FUNNEL_RESOURCE_FILES_NAMES = {"model_state": "model_state.pdparams", "model_config": "model_config.json"}


class FunnelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~hf_paddle.FunnelModel` or a
    :class:`~hf_paddle.TFBertModel`. It is used to instantiate a Funnel Transformer model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the Funnel Transformer `funnel-transformer/small
    <https://huggingface.co/funnel-transformer/small>`__ architecture.

    Configuration objects inherit from :class:`~hf_paddle.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~hf_paddle.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the Funnel transformer. Defines the number of different tokens that can be represented
            by the :obj:`inputs_ids` passed when calling :class:`~hf_paddle.FunnelModel` or
            :class:`~hf_paddle.TFFunnelModel`.
        block_sizes (:obj:`List[int]`, `optional`, defaults to :obj:`[4, 4, 4]`):
            The sizes of the blocks used in the model.
        block_repeats (:obj:`List[int]`, `optional`):
            If passed along, each layer of each block is repeated the number of times indicated.
        num_decoder_layers (:obj:`int`, `optional`, defaults to 2):
            The number of layers in the decoder (when not using the base model).
        d_model (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the model's hidden states.
        n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_head (:obj:`int`, `optional`, defaults to 64):
            Dimensionality of the model's heads.
        d_inner (:obj:`int`, `optional`, defaults to 3072):
            Inner dimension in the feed-forward blocks.
        hidden_act (:obj:`str` or :obj:`callable`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probability used between the two layers of the feed-forward blocks.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 3):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~hf_paddle.FunnelModel` or
            :class:`~hf_paddle.TFFunnelModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.1):
            The standard deviation of the `uniform initializer` for initializing all weight matrices in attention
            layers.
        initializer_std (:obj:`float`, `optional`):
            The standard deviation of the `normal initializer` for initializing the embedding matrix and the weight of
            linear layers. Will default to 1 for the embedding matrix and the value given by Xavier initialization for
            linear layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-9):
            The epsilon used by the layer normalization layers.
        pooling_type (:obj:`str`, `optional`, defaults to :obj:`"mean"`):
            Possible values are ``"mean"`` or ``"max"``. The way pooling is performed at the beginning of each block.
        attention_type (:obj:`str`, `optional`, defaults to :obj:`"relative_shift"`):
            Possible values are ``"relative_shift"`` or ``"factorized"``. The former is faster on CPU/GPU while the
            latter is faster on TPU.
        separate_cls (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to separate the cls token when applying pooling.
        truncate_seq (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When using ``separate_cls``, whether or not to truncate the last token when pooling, to avoid getting a
            sequence length that is not a multiple of 2.
        pool_q_only (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to apply the pooling only to the query or to query, key and values for the attention layers.
    """
    model_type = "funnel"
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "n_head"}

    def __init__(
        self,
        vocab_size=30522,
        block_sizes=[4, 4, 4],
        block_repeats=None,
        num_decoder_layers=2,
        d_model=768,
        n_head=12,
        d_head=64,
        d_inner=3072,
        hidden_act="gelu_new",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        max_position_embeddings=512,
        type_vocab_size=3,
        initializer_range=0.1,
        initializer_std=None,
        layer_norm_eps=1e-9,
        pooling_type="mean",
        attention_type="relative_shift",
        separate_cls=True,
        truncate_seq=True,
        pool_q_only=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.block_sizes = block_sizes
        self.block_repeats = [1] * len(block_sizes) if block_repeats is None else block_repeats
        assert len(block_sizes) == len(
            self.block_repeats
        ), "`block_sizes` and `block_repeats` should have the same length."
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.initializer_std = initializer_std
        self.layer_norm_eps = layer_norm_eps
        assert pooling_type in [
            "mean",
            "max",
        ], f"Got {pooling_type} for `pooling_type` but only 'mean' and 'max' are supported."
        self.pooling_type = pooling_type
        assert attention_type in [
            "relative_shift",
            "factorized",
        ], f"Got {attention_type} for `attention_type` but only 'relative_shift' and 'factorized' are supported."
        self.attention_type = attention_type
        self.separate_cls = separate_cls
        self.truncate_seq = truncate_seq
        self.pool_q_only = pool_q_only

    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)

    @property
    def num_blocks(self):
        return len(self.block_sizes)
