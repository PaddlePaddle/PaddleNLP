# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

from typing import List

import paddle
import paddle.nn as nn

from paddlenlp.utils.env import CONFIG_NAME

from ...utils.converter import StateDictNameMapping, init_name_mappings
from .. import PretrainedModel, register_base_model
from .configuration import (
    DISTILBERT_PRETRAINED_INIT_CONFIGURATION,
    DISTILBERT_PRETRAINED_RESOURCE_FILES_MAP,
    DistilBertConfig,
)

__all__ = [
    "DistilBertModel",
    "DistilBertPretrainedModel",
    "DistilBertForSequenceClassification",
    "DistilBertForTokenClassification",
    "DistilBertForQuestionAnswering",
    "DistilBertForMaskedLM",
]


class BertEmbeddings(nn.Layer):
    """
    Includes embeddings from word, position and does not include
    token_type embeddings.
    """

    def __init__(self, config: DistilBertConfig):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids.stop_gradient = True

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DistilBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained DistilBert models. It provides DistilBert related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    pretrained_init_configuration = DISTILBERT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = DISTILBERT_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "distilbert"
    config_class = DistilBertConfig
    model_config_file = CONFIG_NAME

    @classmethod
    def _get_name_mappings(cls, config: DistilBertConfig) -> List[StateDictNameMapping]:
        mappings: list[StateDictNameMapping] = []
        model_mappings = [
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            ["embeddings.LayerNorm.weight", "embeddings.layer_norm.weight"],
            ["embeddings.LayerNorm.bias", "embeddings.layer_norm.bias"],
        ]
        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"transformer.layer.{layer_index}.attention.q_lin.weight",
                    f"encoder.layers.{layer_index}.self_attn.q_proj.weight",
                    "transpose",
                ],
                [
                    f"transformer.layer.{layer_index}.attention.q_lin.bias",
                    f"encoder.layers.{layer_index}.self_attn.q_proj.bias",
                ],
                [
                    f"transformer.layer.{layer_index}.attention.k_lin.weight",
                    f"encoder.layers.{layer_index}.self_attn.k_proj.weight",
                    "transpose",
                ],
                [
                    f"transformer.layer.{layer_index}.attention.k_lin.bias",
                    f"encoder.layers.{layer_index}.self_attn.k_proj.bias",
                ],
                [
                    f"transformer.layer.{layer_index}.attention.v_lin.weight",
                    f"encoder.layers.{layer_index}.self_attn.v_proj.weight",
                    "transpose",
                ],
                [
                    f"transformer.layer.{layer_index}.attention.v_lin.bias",
                    f"encoder.layers.{layer_index}.self_attn.v_proj.bias",
                ],
                [
                    f"transformer.layer.{layer_index}.attention.out_lin.weight",
                    f"encoder.layers.{layer_index}.self_attn.out_proj.weight",
                    "transpose",
                ],
                [
                    f"transformer.layer.{layer_index}.attention.out_lin.bias",
                    f"encoder.layers.{layer_index}.self_attn.out_proj.bias",
                ],
                [
                    f"transformer.layer.{layer_index}.sa_layer_norm.weight",
                    f"encoder.layers.{layer_index}.norm1.weight",
                ],
                [
                    f"transformer.layer.{layer_index}.sa_layer_norm.bias",
                    f"encoder.layers.{layer_index}.norm1.bias",
                ],
                [
                    f"transformer.layer.{layer_index}.output_layer_norm.weight",
                    f"encoder.layers.{layer_index}.norm2.weight",
                ],
                [
                    f"transformer.layer.{layer_index}.output_layer_norm.bias",
                    f"encoder.layers.{layer_index}.norm2.bias",
                ],
                [
                    f"transformer.layer.{layer_index}.ffn.lin1.weight",
                    f"encoder.layers.{layer_index}.linear1.weight",
                    "transpose",
                ],
                [
                    f"transformer.layer.{layer_index}.ffn.lin1.bias",
                    f"encoder.layers.{layer_index}.linear1.bias",
                ],
                [
                    f"transformer.layer.{layer_index}.ffn.lin2.weight",
                    f"encoder.layers.{layer_index}.linear2.weight",
                    "transpose",
                ],
                [
                    f"transformer.layer.{layer_index}.ffn.lin2.bias",
                    f"encoder.layers.{layer_index}.linear2.bias",
                ],
            ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(model_mappings)
        # base-model prefix "DistilBertModel"
        if "DistilBertModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "distilbert." + mapping[0]
                mapping[1] = "distilbert." + mapping[1]

        # downstream mappings
        if "DistilBertForSequenceClassification" in config.architectures:
            model_mappings.extend(
                [
                    ["pre_classifier.weight", None, "transpose"],
                    "pre_classifier.bias",
                    ["classifier.weight", None, "transpose"],
                    "classifier.bias",
                ]
            )

        if "DistilBertForTokenClassification" in config.architectures:
            model_mappings.extend(
                [
                    ["classifier.weight", None, "transpose"],
                    "classifier.bias",
                ]
            )

        if "DistilBertForQuestionAnswering" in config.architectures:
            model_mappings.extend(
                [["qa_outputs.weight", "classifier.weight", "transpose"], ["qa_outputs.bias", "classifier.bias"]]
            )

        init_name_mappings(model_mappings)
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

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
class DistilBertModel(DistilBertPretrainedModel):
    """
    The bare DistilBert Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `DistilBertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `DistilBertModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layers and the pooler layer. Defaults to `768`.
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
            are supported. Defaults to `"gelu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`DistilBertPretrainedModel.init_weights()` for how weights are initialized in `DistilBertModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.

    """

    def __init__(self, config: DistilBertConfig):
        super(DistilBertModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.embeddings = BertEmbeddings(config)
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

    def forward(self, input_ids, attention_mask=None):
        r"""
        The DistilBertModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
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

        Returns:
            Tensor: Returns tensor `encoder_output`, which means the sequence of hidden-states at the last layer of the model.
            Its data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import DistilBertModel, DistilBertTokenizer

                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                model = DistilBertModel.from_pretrained('distilbert-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.encoder.layers[0].norm1.weight.dtype) * -1e4, axis=[1, 2]
            )
        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(
                    self.encoder.layers[0].norm1.weight.dtype
                )
                attention_mask = (1.0 - attention_mask) * -1e4
        embedding_output = self.embeddings(input_ids=input_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        return encoder_outputs


class DistilBertForSequenceClassification(DistilBertPretrainedModel):
    """
    DistilBert Model with a linear layer on top of the output layer, designed for
    sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`DistilBertConfig`):
            An instance of DistilBertConfig used to construct DistilBertForSequenceClassification.
    """

    def __init__(self, config: DistilBertConfig):
        super(DistilBertForSequenceClassification, self).__init__(config)
        self.num_classes = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask=None):
        r"""
        The DistilBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`DistilBertModel`.
            attention_mask (list, optional):
                See :class:`DistilBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.distilbert.modeling import DistilBertForSequenceClassification
                from paddlenlp.transformers.distilbert.tokenizer import DistilBertTokenizer

                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """

        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = distilbert_output[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class DistilBertForQuestionAnswering(DistilBertPretrainedModel):
    """
    DistilBert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`DistilBertConfig`):
            An instance of DistilBertConfig used to construct DistilBertForQuestionAnswering.
    """

    def __init__(self, config: DistilBertConfig):
        super(DistilBertForQuestionAnswering, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None):
        r"""
        The DistilBertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`DistilBertModel`.
            attention_mask (list, optional):
                See :class:`DistilBertModel`.

        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`).

            With the fields:

            - start_logits(Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - end_logits(Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.distilbert.modeling import DistilBertForQuestionAnswering
                from paddlenlp.transformers.distilbert.tokenizer import DistilBertTokenizer

                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits  =outputs[1]
        """

        sequence_output = self.distilbert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        return start_logits, end_logits


class DistilBertForTokenClassification(DistilBertPretrainedModel):
    """
    DistilBert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`DistilBertConfig`):
            An instance of DistilBertConfig used to construct DistilBertForTokenClassification.
    """

    def __init__(self, config: DistilBertConfig):
        super(DistilBertForTokenClassification, self).__init__(config)
        self.num_classes = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None):
        r"""
        The DistilBertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`DistilBertModel`.
            attention_mask (list, optional):
                See :class:`DistilBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.distilbert.modeling import DistilBertForTokenClassification
                from paddlenlp.transformers.distilbert.tokenizer import DistilBertTokenizer

                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """

        sequence_output = self.distilbert(input_ids, attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class DistilBertForMaskedLM(DistilBertPretrainedModel):
    """
    DistilBert Model with a `language modeling` head on top.

    Args:
        config (:class:`DistilBertConfig`):
            An instance of DistilBertConfig used to construct DistilBertForMaskedLM
    """

    def __init__(self, config: DistilBertConfig):
        super(DistilBertForMaskedLM, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.vocab_layer_norm = nn.LayerNorm(config.hidden_size)
        self.vocab_projector = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids=None, attention_mask=None):
        r"""
        The DistilBertForMaskedLM forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                See :class:`DistilBertModel`.
            attention_mask (Tensor, optional):
                See :class:`DistilBertModel`.

        Returns:
            Tensor: Returns tensor `prediction_logits`, the scores of masked token prediction.
            Its data type should be float32 and its shape is [batch_size, sequence_length, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import DistilBertForMaskedLM, DistilBertTokenizer

                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                prediction_logits = model(**inputs)
        """

        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        prediction_logits = self.vocab_transform(distilbert_output)
        prediction_logits = self.activation(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)
        return prediction_logits
