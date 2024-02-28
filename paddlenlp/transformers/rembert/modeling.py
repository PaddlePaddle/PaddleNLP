# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import PretrainedModel, register_base_model

from ..activations import get_activation
from .configuration import (
    REMBERT_PRETRAINED_INIT_CONFIGURATION,
    REMBERT_PRETRAINED_RESOURCE_FILES_MAP,
    RemBertConfig,
)

__all__ = [
    "RemBertModel",
    "RemBertForMaskedLM",
    "RemBertForQuestionAnswering",
    "RemBertForSequenceClassification",
    "RemBertForMultipleChoice",
    "RemBertPretrainedModel",
    "RemBertForTokenClassification",
]


class RemBertPretrainedModel(PretrainedModel):
    pretrained_init_configuration = REMBERT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = REMBERT_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "rembert"
    config_class = RemBertConfig

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=layer.weight.shape,
                )
            )
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


class RemBertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: RemBertConfig):
        super(RemBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.input_embedding_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.input_embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.input_embedding_size)

        self.layer_norm = nn.LayerNorm(config.input_embedding_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(end=config.max_position_embeddings).expand((1, -1)))

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
    ):
        input_shape = input_ids.shape

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RemBertPooler(nn.Layer):
    def __init__(self, config: RemBertConfig):
        super(RemBertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RemBertSelfAttention(nn.Layer):
    def __init__(self, config: RemBertConfig):
        super(RemBertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RemBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs)
        return outputs


class RemBertSelfOutput(nn.Layer):
    def __init__(self, config: RemBertConfig):
        super(RemBertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class RemBertAttention(nn.Layer):
    def __init__(self, config: RemBertConfig):
        super(RemBertAttention, self).__init__()
        self.self = RemBertSelfAttention(config)
        self.output = RemBertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class RemBertIntermediate(nn.Layer):
    def __init__(self, config: RemBertConfig):
        super(RemBertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RemBertOutput(nn.Layer):
    def __init__(self, config: RemBertConfig):
        super(RemBertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class RemBertLayer(nn.Layer):
    def __init__(self, config: RemBertConfig):
        super(RemBertLayer, self).__init__()
        self.attention = RemBertAttention(config)

        self.intermediate = RemBertIntermediate(config)
        self.output = RemBertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )

        layer_output = self.feed_forward_chunk(self_attention_outputs)

        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RemBertEncoder(nn.Layer):
    def __init__(self, config: RemBertConfig):
        super(RemBertEncoder, self).__init__()
        self.embedding_hidden_mapping_in = nn.Linear(config.input_embedding_size, config.hidden_size)
        self.layer = nn.LayerList([RemBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask)

            hidden_states = layer_outputs

        return hidden_states


@register_base_model
class RemBertModel(RemBertPretrainedModel):
    """
    The bare RemBERT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.


    """

    def __init__(self, config: RemBertConfig):
        super(RemBertModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.layer_norm_eps = config.layer_norm_eps
        self.embeddings = RemBertEmbeddings(config)
        self.encoder = RemBertEncoder(config)

        self.pooler = RemBertPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The RemBertModel forward method, overrides the `__call__()` special method.

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

        Returns:
            tuple: Returns tuple (`sequence_output`, `pooled_output`)

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RemBertModel, RemBertTokenizer

                tokenizer = RemBertTokenizer.from_pretrained('rembert')
                model = RemBertModel.from_pretrained('rembert')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
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
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
        )
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class RemBertForSequenceClassification(RemBertPretrainedModel):
    """
    RemBert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`RemBertConfig`):
            An instance of RemBertConfig used to construct RemBertForSequenceClassification.
    """

    def __init__(self, config: RemBertConfig):
        super(RemBertForSequenceClassification, self).__init__(config)
        self.rembert = RemBertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The RemBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RemBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`RemBertModel`.
            position_ids (Tensor, optional):
                See :class:`RemBertModel`.
            attention_mask (Tensor, optional):
                See :class:`RemBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RemBertForSequenceClassification
                from paddlenlp.transformers import RemBertTokenizer

                tokenizer = RemBertTokenizer.from_pretrained('rembert')
                model = RemBertForQuestionAnswering.from_pretrained('rembert', num_classes=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
        """

        pool_output = self.rembert(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[1]

        pool_output = self.dropout(pool_output)
        logits = self.dense(pool_output)
        return logits


class RemBertForQuestionAnswering(RemBertPretrainedModel):
    """
    RemBert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`RemBertConfig`):
            An instance of RemBertConfig used to construct RemBertForQuestionAnswering.
    """

    def __init__(self, config: RemBertConfig):
        super(RemBertForQuestionAnswering, self).__init__(config)
        self.rembert = RemBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
    ):
        r"""
        The RemBertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RemBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`RemBertModel`.
            position_ids (Tensor, optional):
                See :class:`RemBertModel`.
            attention_mask (Tensor, optional):
                See :class:`RemBertModel`.

        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`).

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RemBertForQuestionAnswering
                from paddlenlp.transformers import RemBertTokenizer

                tokenizer = RemBertTokenizer.from_pretrained('rembert')
                model = RemBertForQuestionAnswering.from_pretrained('rembert')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits = outputs[1]
        """

        outputs = self.rembert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = paddle.split(logits, num_or_sections=2, axis=-1)

        return start_logits, end_logits


class RemBertLMPredictionHead(nn.Layer):
    """
    RemBert Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(self, config: RemBertConfig, embedding_weights=None):
        super(RemBertLMPredictionHead, self).__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.hidden_size)

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


class RemBertOnlyMLMHead(nn.Layer):
    def __init__(self, config: RemBertConfig, embedding_weights):
        super(RemBertOnlyMLMHead, self).__init__()
        self.predictions = RemBertLMPredictionHead(config, embedding_weights=embedding_weights)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class RemBertForMaskedLM(RemBertPretrainedModel):
    """
    RemBert Model with a `masked language modeling` head on top.

    Args:
        config (:class:`RemBertConfig`):
            An instance of RemBertConfig used to construct RemBertForMaskedLM.

    """

    def __init__(self, config: RemBertConfig):
        super(RemBertForMaskedLM, self).__init__(config)
        self.rembert = RemBertModel(config)
        self.cls = RemBertOnlyMLMHead(
            config=config,
            embedding_weights=self.rembert.embeddings.word_embeddings.weight,
        )

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`RemBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`RemBertModel`.
            position_ids (Tensor, optional):
                See :class:`RemBertModel`.
            attention_mask (Tensor, optional):
                See :class:`RemBertModel`.

        Returns:
            Tensor: Returns tensor `prediction_scores`, The scores of masked token prediction.
            Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RemBertForMaskedLM, RemBertTokenizer

                tokenizer = RemBertTokenizer.from_pretrained('rembert')
                model = RemBertForMaskedLM.from_pretrained('rembert')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
        """

        outputs = self.rembert(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output, masked_positions=None)
        return prediction_scores


class RemBertForTokenClassification(RemBertPretrainedModel):
    """
    RemBert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`RemBertConfig`):
            An instance of RemBertConfig used to construct RemBertForTokenClassification.
    """

    def __init__(self, config: RemBertConfig):
        super(RemBertForTokenClassification, self).__init__(config)
        self.num_classes = config.num_classes
        self.rembert = RemBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The RemBertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RemBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`RemBertModel`.
            position_ids(Tensor, optional):
                See :class:`RemBertModel`.
            attention_mask (list, optional):
                See :class:`RemBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RemBertForTokenClassification
                from paddlenlp.transformers import RemBertTokenizer

                tokenizer = RemBertTokenizer.from_pretrained('rembert')
                model = RemBertForTokenClassification.from_pretrained('rembert')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
        """
        sequence_output, _ = self.rembert(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class RemBertForMultipleChoice(RemBertPretrainedModel):
    """
    RemBert Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        config (:class:`RemBertConfig`):
            An instance of RemBertConfig used to construct RemBertForMultipleChoice.
    """

    def __init__(self, config: RemBertConfig):
        super(RemBertForMultipleChoice, self).__init__(config)
        self.num_choices = config.num_choices
        self.rembert = RemBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The BertForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RemBertModel` and shape as [batch_size, num_choice, sequence_length].
            token_type_ids(Tensor, optional):
                See :class:`RemBertModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`RemBertModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`RemBertModel` and shape as [batch_size, num_choice, sequence_length].

        Returns:
            Tensor: Returns tensor `reshaped_logits`, a tensor of the multiple choice classification logits.
            Shape as `[batch_size, num_choice]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RemBertForMultipleChoice, RemBertTokenizer
                from paddlenlp.data import Pad, Dict

                tokenizer = RemBertTokenizer.from_pretrained('rembert')
                model = RemBertForMultipleChoice.from_pretrained('rembert', num_choices=2)

                data = [
                    {
                        "question": "how do you turn on an ipad screen?",
                        "answer1": "press the volume button.",
                        "answer2": "press the lock button.",
                        "label": 1,
                    },
                    {
                        "question": "how do you indent something?",
                        "answer1": "leave a space before starting the writing",
                        "answer2": "press the spacebar",
                        "label": 0,
                    },
                ]

                text = []
                text_pair = []
                for d in data:
                    text.append(d["question"])
                    text_pair.append(d["answer1"])
                    text.append(d["question"])
                    text_pair.append(d["answer2"])

                inputs = tokenizer(text, text_pair)
                batchify_fn = lambda samples, fn=Dict(
                    {
                        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
                        "token_type_ids": Pad(
                            axis=0, pad_val=tokenizer.pad_token_type_id
                        ),  # token_type_ids
                    }
                ): fn(samples)
                inputs = batchify_fn(inputs)

                reshaped_logits = model(
                    input_ids=paddle.to_tensor(inputs[0], dtype="int64"),
                    token_type_ids=paddle.to_tensor(inputs[1], dtype="int64"),
                )
        """
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(-1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1, position_ids.shape[-1]))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(shape=(-1, token_type_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(shape=(-1, attention_mask.shape[-1]))

        _, pooled_output = self.rembert(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        return reshaped_logits
