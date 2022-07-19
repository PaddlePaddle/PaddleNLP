# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import paddle.nn.functional as F

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
    MultipleChoiceModelOutput,
    MaskedLMOutput,
    ModelOutput,
)

__all__ = [
    'RobertaModel',
    'RobertaPretrainedModel',
    'RobertaForSequenceClassification',
    'RobertaForTokenClassification',
    'RobertaForQuestionAnswering',
    'RobertaForMaskedLM',
    'RobertaForMultipleChoice',
    'RobertaForCausalLM',
]


class RobertaEmbeddings(nn.Layer):
    r"""
    Include embeddings from word, position and token_type embeddings.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 pad_token_id=0,
                 cls_token_id=101):
        super(RobertaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.padding_idx = pad_token_id
        self.cls_token_id = cls_token_id

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            if self.cls_token_id == 0 or input_ids[0][
                    0] == 0:  # postion_ids for RobertaBPETokenizer
                position_ids = seq_length + self.padding_idx + 1 - ones
            else:  # postion_ids for RobertaTokenizer
                position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaPooler(nn.Layer):

    def __init__(self, hidden_size):
        super(RobertaPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained RoBerta models. It provides RoBerta related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "hfl/roberta-wwm-ext": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "hfl/roberta-wwm-ext-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "hfl/rbt6": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0,
        },
        "hfl/rbt4": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 4,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0,
        },
        "hfl/rbt3": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0,
        },
        "hfl/rbtl3": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "hfl/roberta-wwm-ext":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roberta_base/roberta_chn_base.pdparams",
            "hfl/roberta-wwm-ext-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/roberta_large/roberta_chn_large.pdparams",
            "hfl/rbt6":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rbt6/rbt6_chn_large.pdparams",
            "hfl/rbt4":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rbt4/rbt4_chn_large.pdparams",
            "hfl/rbt3":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rbt3/rbt3_chn_large.pdparams",
            "hfl/rbtl3":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rbtl3/rbtl3_chn_large.pdparams",
        }
    }
    base_model_prefix = "roberta"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(mean=0.0,
                                     std=self.initializer_range if hasattr(
                                         self, "initializer_range") else
                                     self.roberta.config["initializer_range"],
                                     shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = self.layer_norm_eps if hasattr(
                self,
                "layer_norm_eps") else self.roberta.config["layer_norm_eps"]


@register_base_model
class RobertaModel(RobertaPretrainedModel):
    r"""
    The bare Roberta Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `RobertaModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `RobertaModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layers and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
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
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.RobertaModel`.
            Defaults to `2`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`RobertaPretrainedModel._init_weights()` for how weights are initialized in `RobertaModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        cls_token_id(int, optional):
            The index of cls token in the token vocabulary.
            Defaults to `101`.
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
                 layer_norm_eps=1e-12,
                 cls_token_id=101):
        super(RobertaModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embeddings = RobertaEmbeddings(vocab_size, hidden_size,
                                            hidden_dropout_prob,
                                            max_position_embeddings,
                                            type_vocab_size, pad_token_id,
                                            cls_token_id)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = RobertaPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can be either 0 or 1:

                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.

                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to None, which means no segment embeddings is added to token embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range ``[0, max_position_embeddings - 1]``.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            output_hidden_states (bool, optional):
                Whether or not to output hidden states for all hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (`sequence_output`, `pooled_output`) by default.
            Returns (`encoder_outputs`, `pooled_output`) if output_hidden_states is `True`.

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and its shape is [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RobertaModel, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaModel.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output = model(**inputs)

        """
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(
                    self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids)

        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs
            pooled_output = self.pooler(sequence_output)
            return (sequence_output, pooled_output)
        else:
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)
            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions)


class RobertaForQuestionAnswering(RobertaPretrainedModel):
    r"""
    Roberta Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
     and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        roberta (:class:`RobertaModel`):
            An instance of RobertaModel.
    """

    def __init__(self, roberta):
        super(RobertaForQuestionAnswering, self).__init__()
        self.roberta = roberta  # allow roberta to be config
        self.classifier = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            token_type_ids (Tensor, optional):
                See :class:`RobertaModel`.
            position_ids (Tensor, optional):
                See :class:`RobertaModel`.
            attention_mask (Tensor, optional):
                See :class:`RobertaModel`.
            output_hidden_states (bool, optional):
                See :class:`RobertaModel`.

        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`) by default if output_hidden_states is `False`.
            Returns tuple (`start_logits`, `end_logits`, `encoder_outputs`) if output_hidden_states is set to `True`.

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForSequenceClassification.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        outputs = self.roberta(input_ids,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               attention_mask=attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        total_loss = None
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss, ) +
                    output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForSequenceClassification(RobertaPretrainedModel):
    r"""
    Roberta Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        roberta (:class:`RobertaModel`):
            An instance of `RobertaModel`.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of Roberta.
            If None, use the same value as `hidden_dropout_prob`
            of `RobertaModel` instance `roberta`. Defaults to `None`.
    """

    def __init__(self, roberta, num_classes=2, dropout=None):
        super(RobertaForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.roberta = roberta  # allow roberta to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            token_type_ids (Tensor, optional):
                See :class:`RobertaModel`.
            position_ids (Tensor, optional):
                See :class:`RobertaModel`.
            attention_mask (Tensor, optional):
                See :class:`RobertaModel`.
            output_hidden_states (bool, optional):
                See :class:`RobertaModel`.

        Returns:
            Tensor or tuple: Returns tensor `logits` by default.
            Returns tuple (`logits`, `encoder_outputs`) if output_hidden_states is set to `True`.

            With the fields:

            - `logits` (Tensor):
                a tensor of the input text classification logits.
                Its data type should be float32 and it has a shape of [batch_size, num_classes].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForSequenceClassification.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        outputs = self.roberta(input_ids,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               attention_mask=attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else (
                output[0] if len(output) == 1 else output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForTokenClassification(RobertaPretrainedModel):
    r"""
    Roberta Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        roberta (:class:`RobertaModel`):
            An instance of `RobertaModel`.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of Roberta.
            If None, use the same value as `hidden_dropout_prob`
            of `RobertaModel` instance `roberta`. Defaults to `None`.
    """

    def __init__(self, roberta, num_classes=2, dropout=None):
        super(RobertaForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.roberta = roberta  # allow roberta to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            token_type_ids (Tensor, optional):
                See :class:`RobertaModel`.
            position_ids (Tensor, optional):
                See :class:`RobertaModel`.
            attention_mask (Tensor, optional):
                See :class:`RobertaModel`.
            output_hidden_states (bool, optional):
                See :class:`RobertaModel`.

        Returns:
            Tensor or tuple: Returns tensor `logits` by default.
            Returns tuple (`logits`, `encoder_outputs`) if output_hidden_states is set to `True`.

            With the fields:

            - `logits` (Tensor):
                a tensor of the input token classification logits.
                Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RobertaForTokenClassification, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForTokenClassification.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        outputs = self.roberta(input_ids,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               attention_mask=attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else (
                output[0] if len(output) == 1 else output)

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoice(RobertaPretrainedModel):

    def __init__(self, roberta):
        super().__init__()

        self.roberta = roberta
        self.dropout = nn.Dropout(self.roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"], 1)

        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):

        num_choices = input_ids.shape[1]

        input_ids = input_ids.reshape(
            (-1, input_ids.shape[-1])) if input_ids is not None else None
        position_ids = position_ids.reshape(
            (-1, position_ids.shape[-1])) if position_ids is not None else None
        token_type_ids = token_type_ids.reshape(
            (-1,
             token_type_ids.shape[-1])) if token_type_ids is not None else None
        attention_mask = attention_mask.reshape(
            (-1,
             attention_mask.shape[-1])) if attention_mask is not None else None

        outputs = self.roberta(input_ids,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               attention_mask=attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape((-1, num_choices))

        loss = None
        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else (
                output[0] if len(output) == 1 else output)

        return MultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMaskedLM(RobertaPretrainedModel):
    """
    Roberta Model with a `masked language modeling` head on top.

    Args:
        bert (:class:RobertaModel`):
            An instance of :class:`RobertaModel`.

    """

    def __init__(self, roberta):
        super().__init__()

        self.roberta = roberta
        hidden_size = self.roberta.config['hidden_size']
        layer_norm_eps = self.roberta.config['layer_norm_eps']
        vocab_size = self.roberta.config['vocab_size']

        self.lm_head = RobertaLMHead(hidden_size, layer_norm_eps, vocab_size)

        self.apply(self.init_weights)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            token_type_ids (Tensor, optional):
                See :class:`RobertaModel`.
            position_ids (Tensor, optional):
                See :class:`RobertaModel`.
            attention_mask (Tensor, optional):
                See :class:`RobertaModel`.
            output_hidden_states (bool, optional):
                See :class:`RobertaModel`.

        Returns:
            Tensor or tuple: Returns tensor `prediction_scores` by default.
            Returns tuple (`prediction_scores`, `encoder_outputs`) if output_hidden_states is set to `True`.

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction.
                Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RobertaForMaskedLM, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForMaskedLM.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 30522]
        """

        outputs = self.roberta(input_ids,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               attention_mask=attention_mask,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((masked_lm_loss, ) +
                    output) if masked_lm_loss is not None else (
                        output[0] if len(output) == 1 else output)

        return MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaLMHead(nn.Layer):
    """Roberta Head for masked language modeling."""

    def __init__(self, hidden_size, layer_norm_eps, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class RobertaForCausalLM(RobertaPretrainedModel):
    """
    Roberta Model with a `Causal language modeling` head on top.

    Args:
        bert (:class:RobertaModel`):
            An instance of :class:`RobertaModel`.

    """

    def __init__(self, roberta):
        super().__init__()

        self.roberta = roberta
        hidden_size = self.roberta.config['hidden_size']
        layer_norm_eps = self.roberta.config['layer_norm_eps']
        vocab_size = self.roberta.config['vocab_size']

        self.lm_head = RobertaLMHead(hidden_size, layer_norm_eps, vocab_size)

        self.apply(self.init_weights)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            token_type_ids (Tensor, optional):
                See :class:`RobertaModel`.
            position_ids (Tensor, optional):
                See :class:`RobertaModel`.
            attention_mask (Tensor, optional):
                See :class:`RobertaModel`.
            output_hidden_states (bool, optional):
                See :class:`RobertaModel`.


        Returns:
            Tensor or tuple: Returns tensor `prediction_scores` by default.
            Returns tuple (`prediction_scores`, `encoder_outputs`) if output_hidden_states is set to `True`.

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction.
                Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RobertaForCausalLM, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForCausalLM.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 30522]
        """

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((lm_loss, ) + output) if lm_loss is not None else (
                output[0] if len(output) == 1 else output)

        return MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past=None,
                                      attention_mask=None,
                                      **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past
