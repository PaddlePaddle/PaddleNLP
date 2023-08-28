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

import paddle
import paddle.nn as nn
from paddle.nn import Layer

from paddlenlp.utils.log import logger

from ...layers import Linear as TransposedLinear
from .. import PretrainedModel, register_base_model
from .configuration import (
    LAYOUTLM_PRETRAINED_INIT_CONFIGURATION,
    LAYOUTLM_PRETRAINED_RESOURCE_FILES_MAP,
    LayoutLMConfig,
)

__all__ = [
    "LayoutLMModel",
    "LayoutLMPretrainedModel",
    "LayoutLMForMaskedLM",
    "LayoutLMForTokenClassification",
    "LayoutLMForSequenceClassification",
]


class LayoutLMPooler(Layer):
    def __init__(self, config: LayoutLMConfig):
        super(LayoutLMPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = config.pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutLMEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config: LayoutLMConfig):
        super(LayoutLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # gry add for layoutlm
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        # end of gry add for layoutlm
        # self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size, padding_idx=pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", paddle.arange(config.max_position_embeddings, dtype="int64").expand((1, -1))
        )

    def forward(self, input_ids, bbox=None, token_type_ids=None, position_ids=None):
        # input_shape = input_ids.size()
        # seq_length = input_shape[1]
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
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        # end of gry add

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            word_embeddings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutLMPretrainedModel(PretrainedModel):
    config_class = LayoutLMConfig
    pretrained_init_configuration = LAYOUTLM_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = LAYOUTLM_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "layoutlm"

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )

        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class LayoutLMModel(LayoutLMPretrainedModel):
    """
    The bare LayoutLM Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
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
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`.
            Defaults to `16`.
        initializer_range (float):
            The standard deviation of the normal initializer.
            Defaults to 0.02.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`LayoutLMPretrainedModel.init_weights()` for how weights are initialized in `LayoutLMModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        pool_act (str, optional):
            The non-linear activation function in the pooling layer.
            Defaults to `"tanh"`.
    """

    def __init__(self, config: LayoutLMConfig):
        super(LayoutLMModel, self).__init__(config)
        # self.config = kwargs
        self.num_hidden_layers = config.num_hidden_layers
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.embeddings = LayoutLMEmbeddings(config)

        encoder_layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            attn_dropout=config.attention_probs_dropout_prob,
            act_dropout=0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        self.pooler = LayoutLMPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        num_position_embeds_diff = new_num_position_embeddings - self.config["max_position_embeddings"]

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight

        self.embeddings.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, self.config.hidden_size
        )

        with paddle.no_grad():
            if num_position_embeds_diff > 0:
                self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = old_position_embeddings_weight
            else:
                self.embeddings.position_embeddings.weight = old_position_embeddings_weight[:num_position_embeds_diff]

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        r"""
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
        """

        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
            )
        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2])
        if bbox is None:
            bbox = paddle.zeros(tuple(list(input_shape) + [4]), dtype="int64")

        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

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
    """
    LayoutLM Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`LayoutLMConfig`):
            An instance of LayoutLMConfig used to construct LayoutLMForTokenClassification.
    """

    def __init__(self, config: LayoutLMConfig):
        super(LayoutLMForTokenClassification, self).__init__(config)
        self.num_classes = config.num_classes
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self.classifier.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutlm.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        r"""
        The LayoutLMForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`LayoutLMModel`.
            bbox (Tensor):
                See :class:`LayoutLMModel`.
            attention_mask (list, optional):
                See :class:`LayoutLMModel`.
            token_type_ids (Tensor, optional):
                See :class:`LayoutLMModel`.
            position_ids(Tensor, optional):
                See :class:`LayoutLMModel`.
            output_hidden_states(Tensor, optional):
                See :class:`LayoutLMModel`.


        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import LayoutLMFForTokenClassification
                from paddlenlp.transformers import LayoutLMFTokenizer

                tokenizer = LayoutLMFTokenizer.from_pretrained('layoutlm-base-uncased')
                model = LayoutLMFForTokenClassification.from_pretrained('layoutlm-base-uncased', num_classes=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_tensors="pd")

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 2]

        """
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype("int64")
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=False,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class LayoutLMForSequenceClassification(LayoutLMPretrainedModel):
    """
    LayoutLM Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`LayoutLMConfig`):
            An instance of LayoutLMConfig used to construct LayoutLMForSequenceClassification.
    """

    def __init__(self, config: LayoutLMConfig):
        super(LayoutLMForSequenceClassification, self).__init__(config)
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.num_classes = config.num_classes
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)

    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutlm.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        r"""
        The LayoutLMForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`LayoutLMModel`.
            bbox (Tensor):
                See :class:`LayoutLMModel`.
            attention_mask (list, optional):
                See :class:`LayoutLMModel`.
            token_type_ids (Tensor, optional):
                See :class:`LayoutLMModel`.
            position_ids(Tensor, optional):
                See :class:`LayoutLMModel`.
            output_hidden_states(Tensor, optional):
                See :class:`LayoutLMModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import LayoutLMForSequenceClassification
                from paddlenlp.transformers import LayoutLMTokenizer

                tokenizer = LayoutLMTokenizer.from_pretrained('layoutlm-base-uncased')
                model = LayoutLMForSequenceClassification.from_pretrained('layoutlm-base-uncased', num_classes=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_tensors="pd")

                logits = model(**inputs)
                print(logits.shape)
                # [1, 2]

        """
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        pooled_outputs = outputs[1]
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)
        return logits


class LayoutLMLMPredictionHead(Layer):
    """
    LayoutLM Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, config: LayoutLMConfig, weight_attr=None):
        super(LayoutLMLMPredictionHead, self).__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size, weight_attr=weight_attr)
        self.activation = getattr(nn.functional, config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder = TransposedLinear(config.hidden_size, config.vocab_size)
        # link bias to load pretrained weights
        self.decoder_bias = self.decoder.bias
        # self.decoder_weight = (
        #     self.create_parameter(shape=[vocab_size, hidden_size], dtype=self.transform.weight.dtype, is_bias=False)
        #     if embedding_weights is None
        #     else embedding_weights
        # )
        # self.decoder_bias = self.create_parameter(shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states, [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states, masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class LayoutLMOnlyMLMHead(nn.Layer):
    def __init__(self, config: LayoutLMConfig, weight_attr=None):
        super().__init__()
        self.predictions = LayoutLMLMPredictionHead(config, weight_attr=weight_attr)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class LayoutLMForMaskedLM(LayoutLMPretrainedModel):
    """
    LayoutLM Model with a `masked language modeling` head on top.

    Args:
        config (:class:`LayoutLMConfig`):
            An instance of LayoutLMConfig used to construct LayoutLMForMaskedLM.

    """

    def __init__(self, config: LayoutLMConfig):
        super(LayoutLMForMaskedLM, self).__init__(config)
        self.layoutlm = LayoutLMModel(config)
        self.cls = LayoutLMOnlyMLMHead(config)

    def resize_position_embeddings(self, new_num_position_embeddings):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config["max_position_embeddings"]`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end.
        """
        self.layoutlm.resize_position_embeddings(new_num_position_embeddings)

    def forward(self, input_ids, bbox=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`LayoutLMModel`.
            bbox (Tensor):
                See :class:`LayoutLMModel`.
            token_type_ids (Tensor, optional):
                See :class:`LayoutLMModel`.
            position_ids (Tensor, optional):
                See :class:`LayoutLMModel`.
            attention_mask (Tensor, optional):
                See :class:`LayoutLMModel`.

        Returns:
            Tensor: Returns tensor `prediction_scores`, The scores of masked token prediction.
            Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import LayoutLMForMaskedLM, LayoutLMTokenizer

                tokenizer = LayoutLMTokenizer.from_pretrained('layoutlm-base-uncased')
                model = LayoutLMForMaskedLM.from_pretrained('layoutlm-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", return_tensors="pd")

                logits = model(**inputs)
                print(logits.shape)

        """

        outputs = self.layoutlm(
            input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output, masked_positions=None)
        return prediction_scores
