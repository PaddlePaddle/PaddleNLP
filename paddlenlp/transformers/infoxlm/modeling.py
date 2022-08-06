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

from .. import PretrainedModel, register_base_model

__all__ = [
    "InfoXLMModel",
    "InfoXLMPretrainedModel",
    "InfoXLMForSequenceClassification",
    "InfoXLMForTokenClassification",
    "InfoXLMForQuestionAnswering",
    "InfoXLMForMaskedLM",
    "InfoXLMForMultipleChoice",
]


class InfoXLMClassificationHead(nn.Layer):

    def __init__(self, embed_dim, num_labels, dropout=0.1):
        super(InfoXLMClassificationHead, self).__init__()
        self.num_labels = num_labels
        self.dense = nn.Linear(embed_dim, embed_dim)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Dropout(0.01)
        self.out_proj = nn.Linear(embed_dim, num_labels)

    def forward(self, input):
        x = input[:, 0, :]  # take the head
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.functional.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class InfoXLMEmbeddings(nn.Layer):
    r"""
    Include embeddings from word, position and token_type embeddings.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        pad_token_id=0,
        cls_token_id=101,
    ):
        super(InfoXLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=1e-05)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.padding_idx = pad_token_id
        self.cls_token_id = cls_token_id

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            if (self.cls_token_id == 0 or input_ids[0][0]
                    == 0):  # postion_ids for InfoXLMBPETokenizer
                position_ids = seq_length + self.padding_idx + 1 - ones
            else:  # postion_ids for InfoXLMTokenizer
                position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = input_embedings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class InfoXLMPooler(nn.Layer):

    def __init__(self, hidden_size):
        super(InfoXLMPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class InfoXLMPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained InfoXLM models. It only provides a InfoXLM related
    loading function. Since InfoXLM is not officially on paddleNLP model zoo yet the 
    downloading functionalities are disabled.
    
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "infoxlm-base": {
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
            "vocab_size": 250002,
            "pad_token_id": 0,
        },
        "infoxlm-large": {
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
            "vocab_size": 250002,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "infoxlm-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/infoxlm_base/infoxlm_chn_base.pdparams",
            "infoxlm-large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/infoxlm_large/infoxlm_chn_large.pdparams",
        }
    }
    base_model_prefix = "infoxlm"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range if hasattr(
                        self, "initializer_range") else
                    self.infoxlm.config["initializer_range"],
                    shape=layer.weight.shape,
                ))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = (self.layer_norm_eps if hasattr(
                self, "layer_norm_eps") else
                              self.infoxlm.config["layer_norm_eps"])


@register_base_model
class InfoXLMModel(InfoXLMPretrainedModel):
    r"""
    The bare InfoXLM Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `InfoXLMModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `InfoXLMModel`.
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
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.InfoXLMModel`.
            Defaults to `2`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`InfoXLMPretrainedModel._init_weights()` for how weights are initialized in `InfoXLMModel`.

        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        cls_token_id(int, optional):
            The index of cls token in the token vocabulary.
            Defaults to `101`.
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
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        cls_token_id=101,
    ):
        super(InfoXLMModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embeddings = InfoXLMEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            pad_token_id,
            cls_token_id,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = InfoXLMPooler(hidden_size)
        self.h = num_hidden_layers
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False,
                cache=None):
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
            cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)`.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.

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
                from paddlenlp.transformers import InfoXLMModel, InfoXLMTokenizer

                tokenizer = InfoXLMTokenizer.from_pretrained('infoxlm-base')
                model = InfoXLMModel.from_pretrained('infoxlm-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output = model(**inputs)

        """

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(
                    self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2],
            )
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # up to this line things are good
        if output_hidden_states:
            output = embedding_output
            encoder_outputs = [embedding_output]
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
            return encoder_outputs, pooled_output
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
            return sequence_output, pooled_output


class InfoXLMForQuestionAnswering(InfoXLMPretrainedModel):
    r"""
    InfoXLM Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
     and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        infoxlm (:class:`InfoXLMModel`):
            An instance of InfoXLMModel.
    """

    def __init__(self, infoxlm):
        super(InfoXLMForQuestionAnswering, self).__init__()
        self.infoxlm = infoxlm  # allow infoxlm to be config
        self.classifier = nn.Linear(self.infoxlm.config["hidden_size"], 2)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`InfoXLMModel`.
            token_type_ids (Tensor, optional):
                See :class:`InfoXLMModel`.
            position_ids (Tensor, optional):
                See :class:`InfoXLMModel`.
            attention_mask (Tensor, optional):
                See :class:`InfoXLMModel`.
            output_hidden_states (bool, optional):
                See :class:`InfoXLMModel`.

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
                from paddlenlp.transformers import InfoXLMForSequenceClassification, InfoXLMTokenizer

                tokenizer = InfoXLMTokenizer.from_pretrained('infoxlm-base')
                model = InfoXLMForSequenceClassification.from_pretrained('infoxlm-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        encoder_outputs, _ = self.infoxlm(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = (encoder_outputs[-1]
                           if output_hidden_states else encoder_outputs)
        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        if output_hidden_states:
            return start_logits, end_logits, encoder_outputs
        else:
            return start_logits, end_logits


class InfoXLMForSequenceClassification(InfoXLMPretrainedModel):
    r"""
    InfoXLM Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        infoxlm (:class:`InfoXLMModel`):
            An instance of `InfoXLMModel`.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of InfoXLM.
            If None, use the same value as `hidden_dropout_prob`
            of `InfoXLMModel` instance `infoxlm`. Defaults to `None`.
    """

    def __init__(self, infoxlm, num_classes=2, dropout=None):
        super(InfoXLMForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.infoxlm = infoxlm  # allow infoxlm to be config

        self.classifier = InfoXLMClassificationHead(
            self.infoxlm.config["hidden_size"], num_classes, dropout=dropout)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`InfoXLMModel`.
            token_type_ids (Tensor, optional):
                See :class:`InfoXLMModel`.
            position_ids (Tensor, optional):
                See :class:`InfoXLMModel`.
            attention_mask (Tensor, optional):
                See :class:`InfoXLMModel`.
            output_hidden_states (bool, optional):
                See :class:`InfoXLMModel`.

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
                from paddlenlp.transformers import InfoXLMForSequenceClassification, InfoXLMTokenizer

                tokenizer = InfoXLMTokenizer.from_pretrained('infoxlm-base')
                model = InfoXLMForSequenceClassification.from_pretrained('infoxlm-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        encoder_outputs, pooled_output = self.infoxlm(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        logits = self.classifier(encoder_outputs)

        if output_hidden_states:
            return logits, encoder_outputs
        return logits


class InfoXLMForTokenClassification(InfoXLMPretrainedModel):
    r"""
    InfoXLM Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        infoxlm (:class:`InfoXLMModel`):
            An instance of `InfoXLMModel`.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of InfoXLM.
            If None, use the same value as `hidden_dropout_prob`
            of `InfoXLMModel` instance `infoxlm`. Defaults to `None`.
    """

    def __init__(self, infoxlm, num_classes=2, dropout=None):
        super(InfoXLMForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.infoxlm = infoxlm  # allow infoxlm to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  infoxlm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.infoxlm.config["hidden_size"],
                                    num_classes)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`InfoXLMModel`.
            token_type_ids (Tensor, optional):
                See :class:`InfoXLMModel`.
            position_ids (Tensor, optional):
                See :class:`InfoXLMModel`.
            attention_mask (Tensor, optional):
                See :class:`InfoXLMModel`.
            output_hidden_states (bool, optional):
                See :class:`InfoXLMModel`.

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
                from paddlenlp.transformers import InfoXLMForTokenClassification, InfoXLMTokenizer

                tokenizer = InfoXLMTokenizer.from_pretrained('infoxlm-base')
                model = InfoXLMForTokenClassification.from_pretrained('infoxlm-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        encoder_outputs, _ = self.infoxlm(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = (encoder_outputs[-1]
                           if output_hidden_states else encoder_outputs)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if output_hidden_states:
            return logits, encoder_outputs
        return logits


class InfoXLMForMultipleChoice(InfoXLMPretrainedModel):

    def __init__(self, infoxlm):
        super().__init__()

        self.infoxlm = infoxlm
        self.dropout = nn.Dropout(self.infoxlm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.infoxlm.config["hidden_size"], 1)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
    ):

        num_choices = input_ids.shape[1]

        flat_input_ids = (input_ids.reshape(
            (-1, input_ids.shape[-1])) if input_ids is not None else None)
        flat_position_ids = (position_ids.reshape(
            (-1, position_ids.shape[-1])) if position_ids is not None else None)
        flat_token_type_ids = (token_type_ids.reshape(
            (-1,
             token_type_ids.shape[-1])) if token_type_ids is not None else None)
        flat_attention_mask = (attention_mask.reshape(
            (-1,
             attention_mask.shape[-1])) if attention_mask is not None else None)

        encoder_outputs, pooled_output = self.infoxlm(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        output = logits.reshape((-1, num_choices))

        if output_hidden_states:
            return output, encoder_outputs
        return output


class InfoXLMForMaskedLM(InfoXLMPretrainedModel):
    """
    InfoXLM Model with a `masked language modeling` head on top.

    Args:
        bert (:class:InfoXLMModel`):
            An instance of :class:`InfoXLMModel`.

    """

    def __init__(self, infoxlm):
        super().__init__()

        self.infoxlm = infoxlm
        hidden_size = self.infoxlm.config["hidden_size"]
        layer_norm_eps = self.infoxlm.config["layer_norm_eps"]
        vocab_size = self.infoxlm.config["vocab_size"]

        self.lm_head = InfoXLMLMHead(hidden_size, layer_norm_eps, vocab_size)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`InfoXLMModel`.
            token_type_ids (Tensor, optional):
                See :class:`InfoXLMModel`.
            position_ids (Tensor, optional):
                See :class:`InfoXLMModel`.
            attention_mask (Tensor, optional):
                See :class:`InfoXLMModel`.
            output_hidden_states (bool, optional):
                See :class:`InfoXLMModel`.

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
                from paddlenlp.transformers import InfoXLMForMaskedLM, InfoXLMTokenizer

                tokenizer = InfoXLMTokenizer.from_pretrained('infoxlm-base')
                model = InfoXLMForMaskedLM.from_pretrained('infoxlm-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 30522]
        """

        encoder_outputs, pooled_output = self.infoxlm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = (encoder_outputs[-1]
                           if output_hidden_states else encoder_outputs)
        prediction_scores = self.lm_head(sequence_output)

        if output_hidden_states:
            return prediction_scores, encoder_outputs
        return prediction_scores


class InfoXLMLMHead(nn.Layer):
    """InfoXLM Head for masked language modeling."""

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
