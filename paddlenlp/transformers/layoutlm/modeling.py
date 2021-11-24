# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" Modeling classes for LayoutLM model."""

import copy
import math
import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.nn import CrossEntropyLoss

from .. import PretrainedModel, register_base_model

__all__ = [
    "LayoutLMModel",
    "LayoutLMPretrainedModel",
    #"LayoutLMForMaskedLM",
    "LayoutLMForTokenClassification",
    "LayoutLMForSequenceClassification",
]


class LayoutLMPooler(Layer):
    def __init__(self, hidden_size, pool_act='tanh'):
        super(LayoutLMPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == 'tanh':
            pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutLMEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 max_2d_position_embeddings=1024,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 type_vocab_size=16):
        super(LayoutLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        # gry add for layoutlm
        self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings,
                                                  hidden_size)
        self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings,
                                                  hidden_size)
        self.h_position_embeddings = nn.Embedding(max_2d_position_embeddings,
                                                  hidden_size)
        self.w_position_embeddings = nn.Embedding(max_2d_position_embeddings,
                                                  hidden_size)
        # end of gry add for layoutlm
        #self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size, padding_idx=pad_token_id)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.register_buffer("position_ids",
                             paddle.arange(max_position_embeddings).expand(
                                 (1, -1)))

    def forward(self,
                input_ids,
                bbox=None,
                token_type_ids=None,
                position_ids=None):
        #input_shape = input_ids.size()
        #seq_length = input_shape[1]
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # gry add
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :,
                                                                        1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :,
                                                                        2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :,
                                                                        3])
        except IndexError as e:
            raise IndexError(
                "The :obj:`bbox`coordinate values should be within 0-1000 range."
            ) from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] -
                                                           bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] -
                                                           bbox[:, :, 0])
        # end of gry add

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            word_embeddings + position_embeddings + left_position_embeddings +
            upper_position_embeddings + right_position_embeddings +
            lower_position_embeddings + h_position_embeddings +
            w_position_embeddings + token_type_embeddings)

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutLMPretrainedModel(PretrainedModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "layoutlm-base-uncased": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "max_2d_position_embeddings": 1024,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "use_cache": True,
        },
        "layoutlm-large-uncased": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_2d_position_embeddings": 1024,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "output_attentions": False,
            "output_hidden_states": False,
            "num_labels": 2,
            "use_cache": True,
            "vocab_size": 30522
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "layoutlm-base-uncased":
            "https://paddlenlp.bj.bcebos.com/models/transformers/layoutlm_base-uncased/model_state.pdparams",
            "layoutlm-large-uncased":
            "https://paddlenlp.bj.bcebos.com/models/transformers/layoutlm_large-uncased/model_state.pdparams",
        },
    }
    base_model_prefix = "layoutlm"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.pretrained_init_configuration[
                            "initializer_range"] if "initializer_range" in
                        self.pretrained_init_configuration else 0.02,
                        shape=layer.weight.shape))

        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class LayoutLMModel(LayoutLMPretrainedModel):
    """
    The bare LayoutLM Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of the LayoutLM model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling LayoutLMModel.
        hidden_size (int):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (int):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (int):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (int):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported.
        hidden_dropout_prob (float):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (float):
            The dropout probability for all fully connected layers in the pooler.
        initializer_range (float):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            max_2d_position_embeddings=1024,
            type_vocab_size=16,
            initializer_range=0.02,
            pad_token_id=0,
            pool_act="tanh", ):
        super(LayoutLMModel, self).__init__()
        #self.config = kwargs
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = LayoutLMEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, max_2d_position_embeddings, layer_norm_eps,
            pad_token_id, type_vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = LayoutLMPooler(hidden_size, pool_act)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            bbox=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None,
            output_hidden_states=False, ):
        r'''
        The LayoutLMModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate different portions of the inputs.
                Selected in the range ``[0, type_vocab_size - 1]``.
                If `type_vocab_size` is 2, which means the inputs have two portions.
                Indices can either be 0 or 1:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to `None`, which means we don't add segment embeddings.
            position_ids(Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (`sequence_output`, `pooled_output`).

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
        '''

        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2])
        '''
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)
        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand_as(input_ids)'''
        if bbox is None:
            bbox = paddle.zeros(tuple(list(input_shape) + [4]), dtype="int64")
        '''
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1)
                head_mask = head_mask.expand(self.num_hidden_layers,
                                             -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.num_hidden_layers
        '''

        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            position_ids=position_ids,
            token_type_ids=token_type_ids, )

        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class LayoutLMForTokenClassification(LayoutLMPretrainedModel):
    def __init__(self, layoutlm, num_classes=2, dropout=None):
        super(LayoutLMForTokenClassification, self).__init__()
        self.num_classes = num_classes
        if isinstance(layoutlm, dict):
            self.layoutlm = LayoutLMModel(**layoutxlm)
        else:
            self.layoutlm = layoutlm
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.layoutlm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.layoutlm.config["hidden_size"],
                                    num_classes)
        self.classifier.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def forward(
            self,
            input_ids=None,
            bbox=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None, ):

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask, )
        seq_length = input_ids.shape[1]
        sequence_output, pooled_output = outputs[0][:, :seq_length], outputs[
            0][:, seq_length:]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = logits
        return outputs


class LayoutLMForSequenceClassification(LayoutLMPretrainedModel):
    def __init__(self, layoutlm, num_classes=2, dropout=None):
        super(LayoutLMForSequenceClassification, self).__init__()
        self.layoutlm = layoutlm
        self.num_classes = num_classes
        #if isinstance(layoutlm, dict):
        #   self.layoutlm = LayoutLMModel(**layoutlm)
        #else:
        #   self.layoutlm = layoutlm
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.layoutlm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.layoutlm.config["hidden_size"],
                                    num_classes)
        #self.classifier.apply(self.init_weights)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def forward(
            self,
            input_ids=None,
            bbox=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            output_hidden_states=False, ):
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states)
        pooled_outputs = outputs[1]
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)
        outputs = logits
        return outputs
