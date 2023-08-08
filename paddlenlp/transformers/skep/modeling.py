# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional, Tuple

import paddle
import paddle.nn as nn
from paddle import Tensor

from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import compare_version

if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder

from .. import PretrainedModel, register_base_model
from ..model_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .configuration import (
    SKEP_PRETRAINED_INIT_CONFIGURATION,
    SKEP_PRETRAINED_RESOURCE_FILES_MAP,
    SkepConfig,
)

__all__ = [
    "SkepModel",
    "SkepPretrainedModel",
    "SkepForSequenceClassification",
    "SkepForTokenClassification",
    "SkepCrfForTokenClassification",
]


class SkepEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self, config: SkepConfig):
        super(SkepEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size != 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values_length: Optional[int] = 0,
    ):

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids is None:
            input_shape = paddle.shape(inputs_embeds)[:-1]
            # maybe need use shape op to unify static graph and dynamic graph
            ones = paddle.ones(input_shape, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones

            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length

            position_ids.stop_gradient = True

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        if self.type_vocab_size != 0:
            if token_type_ids is None:
                token_type_ids_shape = paddle.shape(inputs_embeds)[:-1]
                token_type_ids = paddle.zeros(token_type_ids_shape, dtype="int64")
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        elif token_type_ids is not None:
            logger.warning(
                "There is no need to pass the token type ids to SKEP based on RoBERTa model."
                "The input token type ids will be ignored."
            )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SkepPooler(nn.Layer):
    """
    The pooling layer on skep model.
    """

    def __init__(self, config: SkepConfig):
        super(SkepPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SkepPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained Skep models. It provides Skep related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    config_class = SkepConfig
    base_model_prefix = "skep"

    pretrained_init_configuration = SKEP_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = SKEP_PRETRAINED_RESOURCE_FILES_MAP

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5


@register_base_model
class SkepModel(SkepPretrainedModel):
    r"""
    The bare SKEP Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    More details refer to `SKEP <https://www.aclweb.org/anthology/2020.acl-main.374>`.

    Args:
        vocab_size (`int`, optional, defaults to 12800): Vocabulary size of the SKEP model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [`SKEPModel`].
        hidden_size (`int`, optional, defaults to 768): Dimensionality of the embedding layer, encoder layers and the pooler layer.
        num_hidden_layers (int, optional, defaults to 12): Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, optional, defaults to 12):  Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, optional, defaults to 3072): Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors to ff layers are firstly projected from `hidden_size` to `intermediate_size`, and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
        hidden_act (`str`, optional, defaults to "relu"):The non-linear activation function in the encoder and pooler. "gelu", "relu" and any other paddle supported activation functions are supported.
        hidden_dropout_prob (`float`, optional, defaults to 0.1): The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, optional, defaults to 0.1): The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
        max_position_embeddings (`int`, optional, defaults to 512): The maximum sequence length that this model might ever be used with. Typically set this to something large (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, optional, defaults to 4): The vocabulary size of the *token_type_ids* passed into [`SKEPModel`].
        initializer_range (`float`, optional, defaults to 0.02): The standard deviation of the normal initializer.
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`SkepPretrainedModel.init_weights()` for how weights are initialized in [`SkepModel`].
        pad_token_id(int, optional, defaults to 0): The index of padding token in the token vocabulary.

    """

    def __init__(self, config: SkepConfig):
        super(SkepModel, self).__init__(config)
        self.initializer_range = config.initializer_range
        self.embeddings = SkepEmbeddings(config)
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
        self.pooler = SkepPooler(config)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The SkepModel forward method, overrides the `__call__()` special method.

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
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
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
            inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
            past_key_values (tuple(tuple(Tensor)), optional):
                The length of tuple equals to the number of layers, and each inner
                tuple haves 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`)
                which contains precomputed key and value hidden states of the attention blocks.
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (bool, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`.

            if the result is tuple: Returns tuple (`sequence_output`, `pooled_output`).

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
                from paddlenlp.transformers import SkepModel, SkepTokenizer

                tokenizer = SkepTokenizer.from_pretrained('skep_ernie_2.0_large_en')
                model = SkepModel.from_pretrained('skep_ernie_2.0_large_en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False
        use_cache = use_cache if use_cache is not None else False

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids.astype("int64") == self.config.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2],
            )
            if past_key_values is not None:
                batch_size = paddle.shape(past_key_values[0][0])[0]

                past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)

        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            cache=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

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
                attentions=encoder_outputs.attentions,
            )

    def get_input_embeddings(self) -> nn.Embedding:
        """get skep input word embedding

        Returns:
            nn.Embedding: the input word embedding of skep mdoel
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, embedding: nn.Embedding) -> nn.Embedding:
        """set skep input embedding

        Returns:
            nn.Embedding: the instance of new word embedding
        """
        self.embeddings.word_embeddings = embedding


class SkepForSequenceClassification(SkepPretrainedModel):
    r"""
    SKEP Model with a linear layer on top of the pooled output,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`SkepConfig`): An instance of SkepConfig used to contruct SkepForSequenceClassification.
    """

    def __init__(self, config: SkepConfig):
        super(SkepForSequenceClassification, self).__init__(config)
        self.skep = SkepModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The SkepForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`SkepModel`.
            token_type_ids (Tensor, optional):
                See :class:`SkepModel`.
            position_ids (Tensor, `optional`):
                See :class:`SkepModel`.
            attention_mask (Tensor, optional):
                See :class:`SkepModel`.
            labels (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the sequence classification/regression loss.
                Indices should be in `[0, ..., num_labels - 1]`. If `num_labels == 1`
                a regression loss is computed (Mean-Square loss), If `num_labels > 1`
                a classification loss is computed (Cross-Entropy).
            inputs_embeds(Tensor, optional):
                See :class:`SkepModel`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

                tokenizer = SkepTokenizer.from_pretrained('skep_ernie_2.0_large_en')
                model = SkepForSequenceClassification.from_pretrained('skep_ernie_2.0_large_en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        outputs = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = paddle.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SkepForTokenClassification(SkepPretrainedModel):
    r"""
    SKEP Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`SkepConfig`): An instance of SkepConfig used to construct SkepForTokenClassification.

    """

    def __init__(self, config: SkepConfig):
        super(SkepForTokenClassification, self).__init__(config)
        self.skep = SkepModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The SkepForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`SkepModel`.
            token_type_ids (Tensor, optional):
                See :class:`SkepModel`.
            position_ids (Tensor, optional):
                See :class:`SkepModel`.
            attention_mask (Tensor, optional):
                See :class:`SkepModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the token classification loss. Indices should be in `[0, ..., num_labels - 1]`.
            inputs_embeds(Tensor, optional):
                See :class:`SkepModel`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import SkepForTokenClassification, SkepTokenizer

                tokenizer = SkepTokenizer.from_pretrained('skep_ernie_2.0_large_en')
                model = SkepForTokenClassification.from_pretrained('skep_ernie_2.0_large_en')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        outputs = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else (output[0] if len(output) == 1 else output)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SkepCrfForTokenClassification(SkepPretrainedModel):
    r"""
    SKEPCRF Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`SkepConfig`): An instance of SkepConfig used to construct SkepCrfForTokenClassification.
    """

    def __init__(self, config: SkepConfig):
        super(SkepCrfForTokenClassification, self).__init__(config)
        self.skep = SkepModel(config)
        self.num_labels = config.num_labels
        gru_hidden_size = 128

        self.gru = nn.GRU(config.hidden_size, gru_hidden_size, num_layers=2, direction="bidirect")
        self.fc = nn.Linear(
            gru_hidden_size * 2,
            self.num_labels,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(low=-0.1, high=0.1),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4),
            ),
        )
        self.crf = LinearChainCrf(self.num_labels, crf_lr=0.2, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        seq_lens: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The SkepCrfForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`SkepModel`.
            token_type_ids (Tensor, optional):
                See :class:`SkepModel`.
            position_ids (Tensor, optional):
                See :class:`SkepModel`.
            attention_mask (Tensor, optional):
                See :class:`SkepModel`.
            seq_lens (Tensor, optional):
                The input length tensor storing real length of each sequence for correctness.
                Its data type should be int64 and its shape is `[batch_size]`.
                Defaults to `None`.
            labels (Tensor, optional):
                The input label tensor.
                Its data type should be int64 and its shape is `[batch_size, sequence_length]`.
            inputs_embeds(Tensor, optional):
                See :class:`SkepModel`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput`.

            if return_dict is False, Returns tensor `loss` if `labels` is not None. Otherwise, returns tensor `prediction`.

            - `loss` (Tensor):
                The crf loss. Its data type is float32 and its shape is `[batch_size]`.

            - `prediction` (Tensor):
                The prediction tensor containing the highest scoring tag indices.
                Its data type is int64 and its shape is `[batch_size, sequence_length]`.

        """
        outputs = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        bigru_output, _ = self.gru(outputs[0])
        emission = self.fc(bigru_output)

        if seq_lens is None:
            # compute seq length according to the attention mask
            if attention_mask is not None:
                seq_lens = paddle.sum(attention_mask, axis=1, dtype="int64")
            else:
                input_ids_shape = paddle.shape(input_ids)
                seq_lens = paddle.ones(shape=[input_ids_shape[0]], dtype="int64") * input_ids_shape[1]

        loss, prediction = None, None
        if labels is not None:
            loss = self.crf_loss(emission, seq_lens, labels)
        else:
            _, prediction = self.viterbi_decoder(emission, seq_lens)

        if not return_dict:
            # when loss is None, return prediction
            if labels is not None:
                return loss if len(outputs[2:]) == 0 else (loss,) + outputs[2:]
            return prediction if len(outputs[2:]) == 0 else (prediction,) + outputs[2:]

        return TokenClassifierOutput(
            loss=loss,
            logits=prediction,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
