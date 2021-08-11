# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 Huawei Technologies Co., Ltd.
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

import paddle
import paddle.nn as nn

from ..bert.modeling import BertPooler, BertEmbeddings
from .. import PretrainedModel, register_base_model

__all__ = [
    'TinyBertModel',
    'TinyBertPretrainedModel',
    'TinyBertForPretraining',
    'TinyBertForSequenceClassification',
]


class TinyBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained TinyBERT models. It provides TinyBERT
    related `model_config_file`, `resource_files_names`,
    `pretrained_resource_files_map`, `pretrained_init_configuration`,
    `base_model_prefix` for downloading and loading pretrained models. See
    `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "tinybert-4l-312d": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-v2": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-v2": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-zh": {
            "vocab_size": 21128,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-zh": {
            "vocab_size": 21128,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "tinybert-4l-312d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d.pdparams",
            "tinybert-6l-768d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d.pdparams",
            "tinybert-4l-312d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d-v2.pdparams",
            "tinybert-6l-768d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d-v2.pdparams",
            "tinybert-4l-312d-zh":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d-zh.pdparams",
            "tinybert-6l-768d-zh":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d-zh.pdparams",
        }
    }
    base_model_prefix = "tinybert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.tinybert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class TinyBertModel(TinyBertPretrainedModel):
    """
    The bare TinyBert Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Check the superclass documentation for the generic methods and the library implements for all its model.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `TinyBertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `TinyBertModel`.
        hidden_size (int, optional):
            Dimensionality of the encoder layers and the pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding. The dimensionality of position encoding
            is the dimensionality of the sequence in `TinyBertModel`.
            Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids` passed when calling `~ transformers.TinyBertModel`.
            Defaults to `16`.
            `token_type_ids` are segment token indices to indicate first
             and second portions of the inputs. Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

        initializer_range (float, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`BertPretrainedModel.init_weights()` for how weights are initialized in `BertModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        fit_size (int, optional):
            Dimensionality of the output layer of `fit_dense(s)`.
            `fit_dense(s)` means a hidden states' transformation from student to teacher.
            Defaults to `768"`.

    """
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 fit_size=768):
        super(TinyBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = BertPooler(hidden_size)
        # fit_dense(s) means a hidden states' transformation from student to teacher.
        # `fit_denses` is used in v2 model, and `fit_dense` is used in other pretraining models.
        self.fit_denses = nn.LayerList([
            nn.Linear(hidden_size, fit_size)
            for i in range(num_hidden_layers + 1)
        ])
        self.fit_dense = nn.Linear(hidden_size, fit_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r'''
        The TinyBertModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set to **-infinity**
                for any positions in mask that are **0**, and will be **unchanged** for positions that
                are **1**.

                - **1** for tokens that **not masked**,
                - **0** for tokens that **masked**.

                It's data type should be 'float32' and has a shape of [batch_size, sequence_length].
                Defaults to 'None'.

        Returns:
            Tuple: A tuple of shape (`encoder_output`, `pooled_output`).

            With the fields:

            - `encoder_output` (Tensor):
                Sequence of output at hidden layers of the model. Its data type should be float32 and
                has a shape of [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and
                has a shape of [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TinyBertModel, TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-4l-312d')
                model = TinyBertModel.from_pretrained('tinybert-4l-312d')

                inputs = tokenizer("Hey, paddle-paddle is awesome!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        '''

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layer = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoded_layer)

        return encoded_layer, pooled_output


class TinyBertForPretraining(TinyBertPretrainedModel):
    """
    TinyBert Model for pretraining tasks on top.

    Args:
        tinybert (:class:`TinyBertModel`):
            An instance of :class:`TinyBertModel`.

    """

    def __init__(self, tinybert):
        super(TinyBertForPretraining, self).__init__()
        self.tinybert = tinybert
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`TinyBertModel`.
            token_tycpe_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            attention_mask (Tensor, optional):
                See :class:`TinyBertModel`.

        Returns:
            Tensor: `sequence_output`:
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and has a shape of (`batch_size, seq_lens, hidden_size`].
                `seq_lens` corresponds to the length of input sequence.

        """
        sequence_output, pooled_output = self.tinybert(
            input_ids, token_type_ids, attention_mask)

        return sequence_output


class TinyBertForSequenceClassification(TinyBertPretrainedModel):
    """
    TinyBert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.

    Args:
        tinybert (:class:`TinyBertModel`):
            An instance of TinyBertModel.
        num_classes (int, optional):
            The number of classes. Default `2`.
        dropout (float, optional):
            The dropout probability for output of TinyBert.
            If None, use the same value as `hidden_dropout_prob` of `TinyBertModel`
            instance `tinybert`. Default None.
    """
    def __init__(self, tinybert, num_classes=2, dropout=None):
        super(TinyBertForSequenceClassification, self).__init__()
        self.tinybert = tinybert
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.tinybert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.tinybert.config["hidden_size"],
                                    num_classes)
        self.activation = nn.ReLU()
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        The TinyBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`TinyBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            position_ids(Tensor, optional):
                See :class:`TinyBertModel`.
            attention_mask_list (list, optional):
                See :class:`TinyBertModel`.

        Returns:
            logits (Tensor):
                A Tensor of the input text classification logits.
                Shape as `(batch_size, num_classes)` and dtype as `float`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.tinybert.modeling import TinyBertForSequenceClassification
                from paddlenlp.transformers.tinybert.tokenizer import TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-4l-312d')
                model = TinyBertForSequenceClassification.from_pretrained('tinybert-4l-312d')

                inputs = tokenizer("Hey, Paddle-paddle is awesome !")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """

        sequence_output, pooled_output = self.tinybert(
            input_ids, token_type_ids, attention_mask)

        logits = self.classifier(self.activation(pooled_output))
        return logits
