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
""" fnet model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "FNET_PRETRAINED_INIT_CONFIGURATION",
    "FNET_PRETRAINED_RESOURCE_FILES_MAP",
    "FNetConfig",
]

FNET_PRETRAINED_INIT_CONFIGURATION = {
    "fnet-base": {
        "vocab_size": 32000,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 3,
        "bos_token_id": 1,
        "eos_token_id": 2,
    },
    "fnet-large": {
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "intermediate_size": 4096,
        "hidden_act": "gelu_new",
        "hidden_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 4,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 3,
        "bos_token_id": 1,
        "eos_token_id": 2,
    },
}
FNET_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "fnet-base": "https://bj.bcebos.com/paddlenlp/models/transformers/fnet/fnet-base/model_state.pdparams",
        "fnet-large": "https://bj.bcebos.com/paddlenlp/models/transformers/fnet/fnet-large/model_state.pdparams",
    }
}


class FNetConfig(PretrainedConfig):
    r"""
    Args:
        vocab_size (int, optional):
            Vocabulary size of `inputs_ids` in `FNetModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `FNetModel`.
            Defaults to `32000`.
        hidden_size (int, optional):
            Dimensionality of the encoder layer and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `glue_new`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`. Defaults to `4`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to `0.02`.
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `ElectraModel`.
        layer_norm_eps(float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for initializing layer normalization layers.
            A small value to the variance added to the normalization layer to prevent division by zero.
            Defaults to `1e-12`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary. Defaults to `3`.
        add_pooling_layer(bool, optional):
            Whether or not to add the pooling layer. Defaults to `True`.
    """

    model_type = "fnet"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=4,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        add_pooling_layer=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.add_pooling_layer = add_pooling_layer
