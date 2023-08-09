# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
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

import copy
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .. import PretrainedModel, register_base_model
from ..activations import ACT2FN
from .configuration import MPNET_PRETRAINED_INIT_CONFIGURATION, MPNetConfig

__all__ = [
    "MPNetModel",
    "MPNetPretrainedModel",
    "MPNetForMaskedLM",
    "MPNetForSequenceClassification",
    "MPNetForMultipleChoice",
    "MPNetForTokenClassification",
    "MPNetForQuestionAnswering",
]


def create_position_ids_from_input_ids(input_ids, padding_idx=1):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`. :param paddle.Tensor x: :return paddle.Tensor:
    """
    mask = (input_ids != padding_idx).astype(paddle.int64)
    incremental_indices = paddle.cumsum(mask, axis=1).astype(mask.dtype) * mask
    return incremental_indices.astype(paddle.int64) + padding_idx


class MPNetEmbeddings(nn.Layer):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(self, config: MPNetConfig):
        super(MPNetEmbeddings, self).__init__()
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):

        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MPNetAttention(nn.Layer):
    def __init__(self, config: MPNetConfig):
        super(MPNetAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = self.attention_head_size**-0.5
        self.q = nn.Linear(config.hidden_size, self.all_head_size)
        self.k = nn.Linear(config.hidden_size, self.all_head_size)
        self.v = nn.Linear(config.hidden_size, self.all_head_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = x.reshape(new_x_shape)
        return x.transpose(perm=(0, 2, 1, 3))

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = paddle.matmul(q, k, transpose_y=True) * self.scale

        if position_bias is not None:
            attention_scores += position_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, axis=-1)

        attention_probs = self.attention_dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, v)

        context_layer = context_layer.transpose(perm=(0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        projected_context_layer = self.o(context_layer)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layer_normed_context_layer = self.layer_norm(hidden_states + projected_context_layer_dropout)

        return layer_normed_context_layer, attention_scores


class MPNetLayer(nn.Layer):
    def __init__(self, config: MPNetConfig):
        super(MPNetLayer, self).__init__()
        self.attention = MPNetAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        attention_output, layer_att = self.attention(
            hidden_states, attention_mask=attention_mask, position_bias=position_bias
        )

        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        ffn_output_dropout = self.dropout(ffn_output)
        hidden_states = self.layer_norm(ffn_output_dropout + attention_output)

        return hidden_states, layer_att


class MPNetEncoder(nn.Layer):
    def __init__(self, config: MPNetConfig):
        super(MPNetEncoder, self).__init__()
        layer = MPNetLayer(config)
        self.layer = nn.LayerList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, config.num_attention_heads)

    def forward(self, hidden_states, attention_mask=None):
        position_bias = self.compute_position_bias(hidden_states)
        all_encoder_layers = []
        all_encoder_att = []
        for i, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states, layer_att = layer_module(all_encoder_layers[i], attention_mask, position_bias)
            all_encoder_att.append(layer_att)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_att

    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        bsz, qlen, klen = x.shape[0], x.shape[1], x.shape[1]
        if position_ids is not None:
            context_position = position_ids.unsqueeze(2)
            memory_position = position_ids.unsqueeze(1)
        else:
            context_position = paddle.arange(qlen).unsqueeze(1)
            memory_position = paddle.arange(klen).unsqueeze(0)

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(relative_position, num_buckets=num_buckets)

        values = self.relative_attention_bias(rp_bucket)
        values = values.transpose(perm=[2, 0, 1]).unsqueeze(0)
        values = values.expand(shape=(bsz, values.shape[1], qlen, klen))
        return values

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).astype(paddle.int64) * num_buckets
        n = paddle.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            paddle.log(n.astype(paddle.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(paddle.int64)

        val_if_large = paddle.minimum(val_if_large, paddle.full_like(val_if_large, num_buckets - 1))
        ret += paddle.where(is_small, n, val_if_large)
        return ret


class MPNetPooler(nn.Layer):
    """
    Pool the result of MPNetEncoder.
    """

    def __init__(self, config: MPNetConfig):
        super(MPNetPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MPNetPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained MPNet models. It provides MPNet related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "mpnet"
    pretrained_resource_files_map = {
        "model_state": {
            "mpnet-base": "https://bj.bcebos.com/paddlenlp/models/transformers/mpnet/mpnet-base/model_state.pdparams",
        }
    }
    pretrained_init_configuration = MPNET_PRETRAINED_INIT_CONFIGURATION
    config_class = MPNetConfig

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


@register_base_model
class MPNetModel(MPNetPretrainedModel):
    """
    The bare MPNet Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`MPNetConfig`):
            An instance of MPNetConfig used to construct MPNetModel.
    """

    def __init__(self, config: MPNetConfig):
        super(MPNetModel, self).__init__(config)
        self.initializer_range = config.initializer_range
        self.embeddings = MPNetEmbeddings(config)
        self.encoder = MPNetEncoder(config)
        self.pooler = MPNetPooler(config)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        The MPNetModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
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
                The output of first token (`<s>`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import MPNetModel, MPNetTokenizer

                tokenizer = MPNetTokenizer.from_pretrained('mpnet-base')
                model = MPNetModel.from_pretrained('mpnet-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)
        """

        if attention_mask is None:
            attention_mask = (input_ids != self.embeddings.padding_idx).astype(input_ids.dtype)

        if attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(axis=[1, 2])
            attention_mask = (1.0 - attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids)

        encoder_outputs, _ = self.encoder(embedding_output, attention_mask)

        sequence_output = encoder_outputs[-1]
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class MPNetLMHead(nn.Layer):
    """
    MPNet Model with a `language modeling` head on top for CLM fine-tuning.
    """

    def __init__(
        self,
        config: MPNetConfig,
        embedding_weights=None,
    ):
        super(MPNetLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        self.decoder_weight = (
            self.create_parameter(
                shape=[config.vocab_size, config.hidden_size], dtype=self.dense.weight.dtype, is_bias=False
            )
            if embedding_weights is None
            else embedding_weights
        )
        self.decoder_bias = self.create_parameter(
            shape=[config.vocab_size], dtype=self.decoder_weight.dtype, is_bias=True
        )

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = paddle.matmul(hidden_states, self.decoder_weight, transpose_y=True) + self.decoder_bias

        return hidden_states


class MPNetForMaskedLM(MPNetPretrainedModel):
    """
    MPNet Model with a `language modeling` head on top.

    Args:
        config (:class:`MPNetConfig`):
            An instance of MPNetConfig used to construct MPNetModel.

    """

    def __init__(self, config: MPNetConfig):
        super(MPNetForMaskedLM, self).__init__(config)
        self.mpnet = MPNetModel(config)
        self.lm_head = MPNetLMHead(config, embedding_weights=self.mpnet.embeddings.word_embeddings.weight)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        labels=None,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`MPNetModel`.
            position_ids (Tensor, optional):
                See :class:`MPNetModel`.
            attention_mask (Tensor, optional):
                See :class:`MPNetModel`.
            labels (Tensor, optional):
                The Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ..., vocab_size]`` Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., vocab_size]`` Its shape is [batch_size, sequence_length].

        Returns:
            tuple: Returns tuple (`masked_lm_loss`, `prediction_scores`, ``sequence_output`).

            With the fields:

            - `masked_lm_loss` (Tensor):
                The masked lm loss. Its data type should be float32 and its shape is [1].

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32. Its shape is [batch_size, sequence_length, vocab_size].

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model. Its data type should be float32. Its shape is `[batch_size, sequence_length, hidden_size]`.


        """
        sequence_output, pooled_output = self.mpnet(
            input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(shape=(-1, self.mpnet.config["vocab_size"])),
                labels.reshape(shape=(-1,)),
            )
            return masked_lm_loss, prediction_scores, sequence_output

        return prediction_scores, sequence_output


class MPNetForSequenceClassification(MPNetPretrainedModel):
    """
    MPNet Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`MPNetConfig`):
            An instance of MPNetConfig used to construct MPNetModel.
    """

    def __init__(self, config: MPNetConfig):
        super(MPNetForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.mpnet = MPNetModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        The MPNetForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`MPNetModel`.
            position_ids(Tensor, optional):
                See :class:`MPNetModel`.
            attention_mask (list, optional):
                See :class:`MPNetModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import MPNetForSequenceClassification, MPNetTokenizer

                tokenizer = MPNetTokenizer.from_pretrained('mpnet-base')
                model = MPNetForSequenceClassification.from_pretrained('mpnet-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """

        _, pooled_output = self.mpnet(input_ids, position_ids=position_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits


class MPNetForMultipleChoice(MPNetPretrainedModel):
    """
    MPNet Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        config (:class:`MPNetConfig`):
            An instance of MPNetConfig used to construct MPNetModel.
        num_choices (int, optional):
            The number of choices. Defaults to `2`.
    """

    def __init__(self, config: MPNetConfig, num_choices=2):
        super(MPNetForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.mpnet = MPNetModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        The MPNetForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`MPNetModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`MPNetModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`MPNetModel` and shape as [batch_size, num_choice, sequence_length].

        Returns:
            Tensor: Returns tensor `reshaped_logits`, a tensor of the multiple choice classification logits.
            Shape as `[batch_size, num_choice]` and dtype as `float32`.

        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import MPNetForMultipleChoice, MPNetTokenizer

                tokenizer = MPNetTokenizer.from_pretrained('mpnet-base')
                model = MPNetForMultipleChoice.from_pretrained('mpnet-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)

        """
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(-1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1, position_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(shape=(-1, attention_mask.shape[-1]))

        _, pooled_output = self.mpnet(input_ids, position_ids=position_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        return reshaped_logits


class MPNetForTokenClassification(MPNetPretrainedModel):
    """
    MPNet Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`MPNetConfig`):
            An instance of MPNetConfig used to construct MPNetModel.
    """

    def __init__(self, config: MPNetConfig):
        super(MPNetForTokenClassification, self).__init__(config)
        self.mpnet = MPNetModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        The MPNetForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`MPNetModel`.
            position_ids(Tensor, optional):
                See :class:`MPNetModel`.
            attention_mask (list, optional):
                See :class:`MPNetModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import MPNetForTokenClassification, MPNetTokenizer

                tokenizer = MPNetTokenizer.from_pretrained('mpnet-base')
                model = MPNetForTokenClassification.from_pretrained('mpnet-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
        """
        sequence_output, _ = self.mpnet(input_ids, position_ids=position_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        return logits


class MPNetForQuestionAnswering(MPNetPretrainedModel):
    """
    MPNet Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`MPNetConfig`):
            An instance of MPNetConfig used to construct MPNetModel.
    """

    def __init__(self, config: MPNetConfig):
        super(MPNetForQuestionAnswering, self).__init__(config)
        self.mpnet = MPNetModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        The MPNetForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`MPNetModel`.
            position_ids (Tensor, optional):
                See :class:`MPNetModel`.
            attention_mask (Tensor, optional):
                See :class:`MPNetModel`.

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
                from paddlenlp.transformers import MPNetForQuestionAnswering, MPNetTokenizer

                tokenizer = MPNetTokenizer.from_pretrained('mpnet-base')
                model = MPNetForQuestionAnswering.from_pretrained('mpnet-base')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits  = outputs[1]

        """

        sequence_output, _ = self.mpnet(input_ids, position_ids=position_ids, attention_mask=attention_mask)
        logits = self.qa_outputs(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])

        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits
