# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
from paddle import einsum

from ...utils.env import CONFIG_NAME
from .. import PretrainedModel, register_base_model
from ..activations import get_activation
from .configuration import (
    MegatronBert_PRETRAINED_INIT_CONFIGURATION,
    MegatronBert_PRETRAINED_RESOURCE_FILES_MAP,
    MegatronBertConfig,
)

__all__ = [
    "MegatronBertModel",
    "MegatronBertPretrainedModel",
    "MegatronBertForQuestionAnswering",
    "MegatronBertForSequenceClassification",
    "MegatronBertForNextSentencePrediction",
    "MegatronBertForCausalLM",
    "MegatronBertForPreTraining",
    "MegatronBertForMaskedLM",
    "MegatronBertForMultipleChoice",
    "MegatronBertForTokenClassification",
]

layer_norm_eps = 1e-12


class MegatronBertPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained MegatronBert models. It provides RoBerta related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """
    model_config_file = CONFIG_NAME
    resource_files_names = {"model_state": "model_state.pdparams"}

    pretrained_init_configuration = MegatronBert_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = MegatronBert_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "megatronbert"
    config_class = MegatronBertConfig

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range")
                    else self.megatronbert.config["initializer_range"],
                    shape=layer.weight.shape,
                )
            )
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = layer_norm_eps


class MegatronBertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", paddle.arange(end=config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = config.position_embedding_type

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings


class MegatronBertSelfAttention(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = paddle.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = paddle.arange(end=seq_length, dtype="int64").reshape((-1, 1))
            position_ids_r = paddle.arange(end=seq_length, dtype="int64").reshape((1, -1))
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MegatronBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


class MegatronBertSelfOutput(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, residual):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


class MegatronBertAttention(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.self = MegatronBertSelfAttention(config)
        self.output = MegatronBertSelfOutput(config)
        self.pruned_heads = set()

    def forward(self, hidden_states, attention_mask=None):
        ln_outputs = self.layer_norm(hidden_states)
        self_outputs = self.self(ln_outputs, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MegatronBertIntermediate(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MegatronBertOutput(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return input_tensor + hidden_states


class MegatronBertLayer(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertLayer, self).__init__()
        self.seq_len_dim = 1
        self.attention = MegatronBertAttention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.intermediate = MegatronBertIntermediate(config)
        self.output = MegatronBertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        ln_output = self.layer_norm(attention_output)
        intermediate_output = self.intermediate(ln_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MegatronBertEncoder(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertEncoder, self).__init__()
        self.layer = nn.LayerList([MegatronBertLayer(config) for _ in range(config.num_hidden_layers)])

        # The final layer norm. We removed the 1st LN, moved LN to each hidden layer and this one
        # is simply the final LN (Transformer's BERT has it attached to each hidden layer).
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask)

            hidden_states = layer_outputs[0]

        # Finalize the hidden states.
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class MegatronBertPooler(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@register_base_model
class MegatronBertModel(MegatronBertPretrainedModel):
    """
    The bare MegatronBert Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        Args:
        config (:class:`MegatronBertConfig`):
            An instance of MegatronBertConfig used to construct MBartModel.

    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertModel, self).__init__(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.embeddings = MegatronBertEmbeddings(config)
        self.encoder = MegatronBertEncoder(config)
        self.pooler = MegatronBertPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The MegatronBertModel forward method, overrides the `__call__()` special method.

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
                If its data type is int, the values should be either 0 or 1.

                - **1** for tokens that **not masked**,
                - **0** for tokens that **masked**.

                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.

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

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import MegatronBertModel, MegatronBertTokenizer

                tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
                model = MegatronBertModel.from_pretrained('megatronbert-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
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
        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class MegatronBertForQuestionAnswering(MegatronBertPretrainedModel):
    """
    MegatronBert Model with question answering tasks.

    Args:
        megatronbert (:class:`MegatronBertModel`):
            An instance of :class:`MegatronBertModel`.

    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertForQuestionAnswering, self).__init__(config)
        self.megatronbert = MegatronBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
    ):
        r"""
        The MegatronBertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`MegatronBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`MegatronBertModel`.
            position_ids(Tensor, optional):
                See :class:`MegatronBertModel`.
            attention_mask (Tensor, optional):
                See :class:`MegatronBertModel`.
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
                from paddlenlp.transformers import MegatronBertForQuestionAnswering, MegatronBertTokenizer

                tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
                model = MegatronBertForQuestionAnswering.from_pretrained('megatronbert-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits  = outputs[1]
        """

        outputs = self.megatronbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        output = (start_logits, end_logits)
        return output


class MegatronBertForSequenceClassification(MegatronBertPretrainedModel):
    """
    MegatronBert Model with sequence classification tasks.

    Args:
        megatronbert (:class:`MegatronBertModel`):
            An instance of :class:`MegatronBertModel`.
        num_labels (int):
            The number of labels.
    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.megatronbert = MegatronBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The MegatronBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`MegatronBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`MegatronBertModel`.
            position_ids(Tensor, optional):
                See :class:`MegatronBertModel`.
            attention_mask (Tensor, optional):
                See :class:`MegatronBertModel`.
        Returns:
            Tensor: Returns tensor `logits`, a tensor of the sequence classification logits.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import MegatronBertForSequenceClassification, MegatronBertTokenizer

                tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
                model = MegatronBertForSequenceClassification.from_pretrained('megatronbert-uncased', num_labels=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """

        outputs = self.megatronbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class MegatronBertPredictionHeadTransform(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class MegatronBertLMPredictionHead(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertLMPredictionHead, self).__init__()
        self.transform = MegatronBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.

        self.decoder_weight = self.create_parameter(
            shape=[config.vocab_size, config.hidden_size], dtype=self.transform.dense.weight.dtype, is_bias=False
        )
        self.decoder_bias = self.create_parameter(
            shape=[config.vocab_size], dtype=self.decoder_weight.dtype, is_bias=True
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = paddle.tensor.matmul(hidden_states, self.decoder_weight, transpose_y=True) + self.decoder_bias
        return hidden_states


class MegatronBertOnlyMLMHead(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertOnlyMLMHead, self).__init__()
        self.predictions = MegatronBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MegatronBertOnlyNSPHead(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class MegatronBertPreTrainingHeads(nn.Layer):
    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertPreTrainingHeads, self).__init__()
        self.predictions = MegatronBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MegatronBertForPreTraining(MegatronBertPretrainedModel):
    """
    Megatronbert Model with pretraining tasks on top.

    Args:
        megatronbert (:class:`MegatronBertModel`):
            An instance of :class:`MegatronBertModel`.

    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertForPreTraining, self).__init__(config)

        self.megatronbert = MegatronBertModel(config)
        self.cls = MegatronBertPreTrainingHeads(config)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The MegatronBertForPreTraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`MegatronBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`MegatronBertModel`.
            position_ids(Tensor, optional):
                See :class:`MegatronBertModel`.
            attention_mask (Tensor, optional):
                See :class:`MegatronBertModel`.
        Returns:
            tuple: Returns tuple (`prediction_scores`, `seq_relationship_score`).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import MegatronBertForPreTraining, MegatronBertTokenizer

                tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
                model = MegatronBertForPreTraining.from_pretrained('megatronbert-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                prediction_scores, seq_relationship_score = model(**inputs)
        """
        outputs = self.megatronbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        output = (prediction_scores, seq_relationship_score)
        return output


class MegatronBertForCausalLM(MegatronBertPretrainedModel):
    """
    MegatronBert Model with a `causal masked language modeling` head on top.

    Args:
        megatronbert (:class:`MegatronBertModel`):
            An instance of :class:`MegatronBertModel`.

    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertForCausalLM, self).__init__(config)

        self.megatronbert = MegatronBertModel(config)
        self.cls = MegatronBertOnlyMLMHead(config)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The MegatronBertForCausalLM forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`MegatronBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`MegatronBertModel`.
            position_ids(Tensor, optional):
                See :class:`MegatronBertModel`.
            attention_mask (Tensor, optional):
                See :class:`MegatronBertModel`.
        Returns:
            Tensor: Returns Tensor `prediction_scores`. The scores of masked token prediction.
                    Its data type should be float32. If `masked_positions` is None, its shape is
                    [batch_size, sequence_length, vocab_size]. Otherwise, its shape is
                    [batch_size, mask_token_num, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import MegatronBertForCausalLM, MegatronBertTokenizer

                tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
                model = MegatronBertForCausalLM.from_pretrained('megatronbert-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                prediction_scores = model(**inputs)
        """
        outputs = self.megatronbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores


class MegatronBertForMaskedLM(MegatronBertPretrainedModel):
    """
    MegatronBert Model with a `masked language modeling` head on top.

    Args:
        megatronbert (:class:`MegatronBertModel`):
            An instance of :class:`MegatronBertModel`.

    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertForMaskedLM, self).__init__(config)

        self.megatronbert = MegatronBertModel(config)
        self.cls = MegatronBertOnlyMLMHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
    ):
        r"""
        The MegatronBertForMaskedLM forward method, overrides the __call__() special method.

        Args:
           input_ids (Tensor):
               See :class:`MegatronBertModel`.
           token_type_ids (Tensor, optional):
               See :class:`MegatronBertModel`.
           position_ids(Tensor, optional):
               See :class:`MegatronBertModel`.
           attention_mask (Tensor, optional):
               See :class:`MegatronBertModel`.
        Returns:
           Tensor: Returns Tensor `prediction_scores`. The scores of masked token prediction.
                   Its data type should be float32. If `masked_positions` is None, its shape is
                   [batch_size, sequence_length, vocab_size]. Otherwise, its shape is
                   [batch_size, mask_token_num, vocab_size].

        Example:
           .. code-block::

               import paddle
               from paddlenlp.transformers import MegatronBertForMaskedLM, MegatronBertTokenizer

               tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
               model = MegatronBertForMaskedLM.from_pretrained('megatronbert-uncased')

               inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
               inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
               prediction_scores = model(**inputs)
        """

        outputs = self.megatronbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        return prediction_scores


class MegatronBertForNextSentencePrediction(MegatronBertPretrainedModel):
    """
    MegatronBert Model with a `next sentence prediction (classification)` head on top.

    Args:
        megatronbert (:class:`MegatronBertModel`):
            An instance of :class:`MegatronBertModel`.
    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertForNextSentencePrediction, self).__init__(config)

        self.megatronbert = MegatronBertModel(config)
        self.cls = MegatronBertOnlyNSPHead(config)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The MegatronBertForNextSentencePrediction forward method, overrides the __call__() special method.

        Args:
           input_ids (Tensor):
               See :class:`MegatronBertModel`.
           token_type_ids (Tensor, optional):
               See :class:`MegatronBertModel`.
           position_ids(Tensor, optional):
               See :class:`MegatronBertModel`.
           attention_mask (Tensor, optional):
               See :class:`MegatronBertModel`.
        Returns:
           Tensor: Returns Tensor `seq_relationship_scores`. The scores of next sentence prediction.
                   Its data type should be float32 and its shape is [batch_size, 2].

        Example:
           .. code-block::

               import paddle
               from paddlenlp.transformers import MegatronBertForNextSentencePrediction, MegatronBertTokenizer

               tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
               model = MegatronBertForNextSentencePrediction.from_pretrained('megatronbert-uncased')

               inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
               inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
               seq_relationship_scores = model(**inputs)
        """

        outputs = self.megatronbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        return seq_relationship_scores


class MegatronBertForMultipleChoice(MegatronBertPretrainedModel):
    """
    MegatronBert Model with a multiple choice classification head on top.

    Args:
        megatronbert (:class:`MegatronBertModel`):
            An instance of :class:`MegatronBertModel`.
    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertForMultipleChoice, self).__init__(config)

        self.megatronbert = MegatronBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The MegatronBertForMultipleChoice forward method, overrides the __call__() special method.

        Args:
           input_ids (Tensor):
               See :class:`MegatronBertModel`.
           token_type_ids (Tensor, optional):
               See :class:`MegatronBertModel`.
           position_ids(Tensor, optional):
               See :class:`MegatronBertModel`.
           attention_mask (Tensor, optional):
               See :class:`MegatronBertModel`.
        Returns:
           Tensor: Returns Tensor `reshaped_logits`. A tensor of the multiple choice classification logits.
                   Shape as `[batch_size, num_choice]` and dtype as `float32`.

        Example:
           .. code-block::

               import paddle
               from paddlenlp.transformers import MegatronBertForMultipleChoice, MegatronBertTokenizer

               tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
               model = MegatronBertForNextSentencePrediction.from_pretrained('megatronbert-uncased')

               inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
               inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
               reshaped_logits = model(**inputs)
        """
        num_choices = input_ids.shape[1]

        input_ids = input_ids.reshape((-1, input_ids.shape[-1])) if input_ids is not None else None
        attention_mask = attention_mask.reshape((-1, attention_mask.shape[-1])) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape((-1, token_type_ids.shape[-1])) if token_type_ids is not None else None
        position_ids = position_ids.reshape((-1, position_ids.shape[-1])) if position_ids is not None else None

        outputs = self.megatronbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape((-1, num_choices))

        return reshaped_logits


class MegatronBertForTokenClassification(MegatronBertPretrainedModel):
    """
    MegatronBert Model with a token classification head on top.

    Args:
        megatronbert (:class:`MegatronBertModel`):
            An instance of :class:`MegatronBertModel`.

        num_labels (int):
            The number of labels.
    """

    def __init__(self, config: MegatronBertConfig):
        super(MegatronBertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.megatronbert = MegatronBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        r"""
        The MegatronBertForTokenClassification forward method, overrides the __call__() special method.

        Args:
           input_ids (Tensor):
               See :class:`MegatronBertModel`.
           token_type_ids (Tensor, optional):
               See :class:`MegatronBertModel`.
           position_ids(Tensor, optional):
               See :class:`MegatronBertModel`.
           attention_mask (Tensor, optional):
               See :class:`MegatronBertModel`.
        Returns:
           Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
                   Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
           .. code-block::

               import paddle
               from paddlenlp.transformers import MegatronBertForTokenClassification, MegatronBertTokenizer

               tokenizer = MegatronBertTokenizer.from_pretrained('megatronbert-uncased')
               model = MegatronBertForTokenClassification.from_pretrained('megatronbert-uncased', num_labels=2)

               inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
               inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
               reshaped_logits = model(**inputs)
        """

        outputs = self.megatronbert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits
