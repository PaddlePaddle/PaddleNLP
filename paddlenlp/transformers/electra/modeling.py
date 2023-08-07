# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from dataclasses import dataclass
from typing import List, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import TransformerEncoder, TransformerEncoderLayer

from ...utils.converter import StateDictNameMapping, init_name_mappings
from .. import PretrainedModel, register_base_model
from ..activations import get_activation
from ..model_outputs import (
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    tuple_output,
)
from .configuration import (
    ELECTRA_PRETRAINED_INIT_CONFIGURATION,
    ELECTRA_PRETRAINED_RESOURCE_FILES_MAP,
    ElectraConfig,
)

__all__ = [
    "ElectraModel",
    "ElectraPretrainedModel",
    "ElectraForTotalPretraining",
    "ElectraDiscriminator",
    "ElectraGenerator",
    "ElectraClassificationHead",
    "ElectraForSequenceClassification",
    "ElectraForTokenClassification",
    "ElectraPretrainingCriterion",
    "ElectraForMultipleChoice",
    "ElectraForQuestionAnswering",
    "ElectraForMaskedLM",
    "ElectraForPretraining",
    "ErnieHealthForTotalPretraining",
    "ErnieHealthPretrainingCriterion",
    "ErnieHealthDiscriminator",
]


class ElectraEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ElectraConfig):
        super(ElectraEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.layer_norm = nn.LayerNorm(config.embedding_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=None
    ):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            if past_key_values_length is not None:
                position_ids += past_key_values_length
            position_ids.stop_gradient = True
        position_ids = position_ids.astype("int64")

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        if input_ids is not None:
            input_embeddings = self.word_embeddings(input_ids)
        else:
            input_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ElectraDiscriminatorPredictions(nn.Layer):
    """Prediction layer for the discriminator, made up of two dense layers."""

    def __init__(self, config: ElectraConfig):
        super(ElectraDiscriminatorPredictions, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.act = get_activation(config.hidden_act)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.act(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class ElectraGeneratorPredictions(nn.Layer):
    """Prediction layer for the generator, made up of two dense layers."""

    def __init__(self, config: ElectraConfig):
        super(ElectraGeneratorPredictions, self).__init__()

        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.act = get_activation(config.hidden_act)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class ElectraPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Electra models. It provides Electra related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "electra"

    # pretrained general configuration
    gen_weight = 1.0
    disc_weight = 50.0
    tie_word_embeddings = True
    untied_generator_embeddings = False
    use_softmax_sample = True

    # model init configuration
    pretrained_init_configuration = ELECTRA_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ELECTRA_PRETRAINED_RESOURCE_FILES_MAP
    config_class = ElectraConfig

    @classmethod
    def _get_name_mappings(cls, config: ElectraConfig) -> List[StateDictNameMapping]:
        model_mappings = [
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            ["embeddings.LayerNorm.weight", "embeddings.layer_norm.weight"],
            ["embeddings.LayerNorm.bias", "embeddings.layer_norm.bias"],
            ["embeddings_project.weight", None, "transpose"],
            "embeddings_project.bias",
        ]

        for layer_index in range(config.num_hidden_layers):
            layer_mappings = [
                [
                    f"encoder.layer.{layer_index}.attention.self.query.weight",
                    f"encoder.layers.{layer_index}.self_attn.q_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.query.bias",
                    f"encoder.layers.{layer_index}.self_attn.q_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.key.weight",
                    f"encoder.layers.{layer_index}.self_attn.k_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.key.bias",
                    f"encoder.layers.{layer_index}.self_attn.k_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.value.weight",
                    f"encoder.layers.{layer_index}.self_attn.v_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.self.value.bias",
                    f"encoder.layers.{layer_index}.self_attn.v_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.dense.weight",
                    f"encoder.layers.{layer_index}.self_attn.out_proj.weight",
                    "transpose",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.dense.bias",
                    f"encoder.layers.{layer_index}.self_attn.out_proj.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight",
                    f"encoder.layers.{layer_index}.norm1.weight",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias",
                    f"encoder.layers.{layer_index}.norm1.bias",
                ],
                [
                    f"encoder.layer.{layer_index}.intermediate.dense.weight",
                    f"encoder.layers.{layer_index}.linear1.weight",
                    "transpose",
                ],
                [f"encoder.layer.{layer_index}.intermediate.dense.bias", f"encoder.layers.{layer_index}.linear1.bias"],
                [
                    f"encoder.layer.{layer_index}.output.dense.weight",
                    f"encoder.layers.{layer_index}.linear2.weight",
                    "transpose",
                ],
                [f"encoder.layer.{layer_index}.output.dense.bias", f"encoder.layers.{layer_index}.linear2.bias"],
                [f"encoder.layer.{layer_index}.output.LayerNorm.weight", f"encoder.layers.{layer_index}.norm2.weight"],
                [f"encoder.layer.{layer_index}.output.LayerNorm.bias", f"encoder.layers.{layer_index}.norm2.bias"],
            ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(model_mappings)
        # base-model prefix "ElectraModel"
        if "ElectraModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "electra." + mapping[0]
                mapping[1] = "electra." + mapping[1]

        # downstream mappings
        if "ElectraForQuestionAnswering" in config.architectures:
            model_mappings.extend(
                [["qa_outputs.weight", "classifier.weight", "transpose"], ["qa_outputs.bias", "classifier.bias"]]
            )

        if "ElectraForMultipleChoice" in config.architectures:
            model_mappings.extend(
                [
                    ["sequence_summary.summary.weight", "sequence_summary.dense.weight", "transpose"],
                    ["sequence_summary.summary.bias", "sequence_summary.dense.bias"],
                    ["classifier.weight", "classifier.weight", "transpose"],
                    ["classifier.bias", "classifier.bias"],
                ]
            )

        if "ElectraForSequenceClassification" in config.architectures:
            model_mappings.extend(
                [
                    ["classifier.dense.weight", "classifier.dense.weight", "transpose"],
                    ["classifier.dense.bias", "classifier.dense.bias"],
                    ["classifier.out_proj.weight", "classifier.out_proj.weight", "transpose"],
                    ["classifier.out_proj.bias", "classifier.out_proj.bias"],
                ]
            )

        if "ElectraForTokenClassification" in config.architectures:
            model_mappings.extend(
                [
                    ["classifier.weight", "classifier.weight", "transpose"],
                    "classifier.bias",
                ]
            )

        # TODO: need to tie weights
        if "ElectraForMaskedLM" in config.architectures:
            model_mappings.extend(
                [
                    ["generator_predictions.LayerNorm.weight", "generator_predictions.layer_norm.weight", "transpose"],
                    ["generator_predictions.LayerNorm.bias", "generator_predictions.layer_norm.bias"],
                    ["generator_predictions.dense.weight", None, "transpose"],
                    "generator_predictions.dense.bias",
                    ["generator_lm_head.bias", "generator_lm_head_bias"],
                ]
            )

        init_name_mappings(model_mappings)
        return [StateDictNameMapping(*mapping) for mapping in model_mappings]

    def _init_weights(self, layer):
        """Initialize the weights"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=layer.weight.shape,
                )
            )
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))
            layer._epsilon = getattr(self, "layer_norm_eps", 1e-12)
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            layer.bias.set_value(paddle.zeros_like(layer.bias))


@register_base_model
class ElectraModel(ElectraPretrainedModel):
    """
    The bare Electra Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig
    """

    def __init__(self, config: ElectraConfig):
        super(ElectraModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.layer_norm_eps = config.layer_norm_eps
        self.embeddings = ElectraEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            attn_dropout=config.attention_probs_dropout_prob,
            act_dropout=0,
        )
        self.encoder = TransformerEncoder(encoder_layer, config.num_hidden_layers)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The ElectraModel forward method, overrides the `__call__()` special method.

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
            inputs_embeds (Tensor, optional):
                Instead of passing input_ids you can choose to directly pass an embedded representation.
                This is useful for use cases such as P-Tuning, where you want more control over how to convert input_ids indices
                into the embedding space.
                Its data type should be `float32` and it has a shape of [batch_size, sequence_length, embedding_size].
            past_key_values (tuple(tuple(Tensor)), optional):
                Precomputed key and value hidden states of the attention blocks of each layer. This can be used to speedup
                auto-regressive decoding for generation tasks or to support use cases such as Prefix-Tuning where vectors are prepended
                to each attention layer. The length of tuple equals to the number of layers, and each tuple having 2 tensors of shape
                `(batch_size, num_heads, past_key_values_length, embed_size_per_head)`)
                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, optional):
                If set to `True`, `past_key_values` key value states are returned.
                Defaults to `None`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            Tensor: Returns tensor `encoder_outputs`, which is the output at the last layer of the model.
            Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ElectraModel, ElectraTokenizer

                tokenizer = ElectraTokenizer.from_pretrained('electra-small')
                model = ElectraModel.from_pretrained('electra-small')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)

        """
        past_key_values_length = None
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(paddle.get_default_dtype()) * -1e4, axis=[1, 2]
            )
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)
        else:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(axis=[1, 2])

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if hasattr(self, "embeddings_project"):
            embedding_output = self.embeddings_project(embedding_output)

        self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            cache=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return encoder_outputs


class ElectraDiscriminator(ElectraPretrainedModel):
    """
    The Electra Discriminator can detect the tokens that are replaced by the Electra Generator.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig

    """

    def __init__(self, config: ElectraConfig):
        super(ElectraDiscriminator, self).__init__(config)

        self.electra = ElectraModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ElectraModel`.
            token_type_ids (Tensor, optional):
                See :class:`ElectraModel`.
            position_ids (Tensor, optional):
                See :class:`ElectraModel`.
            attention_mask (Tensor, optional):
                See :class:`ElectraModel`.
            inputs_embeds (Tensor, optional):
                See :class:`ElectraModel`.

        Returns:
            Tensor: Returns tensor `logits`, the prediction result of replaced tokens.
            Its data type should be float32 and if batch_size>1, its shape is [batch_size, sequence_length],
            if batch_size=1, its shape is [sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ElectraDiscriminator, ElectraTokenizer

                tokenizer = ElectraTokenizer.from_pretrained('electra-small')
                model = ElectraDiscriminator.from_pretrained('electra-small')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        discriminator_sequence_output = self.electra(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        logits = self.discriminator_predictions(discriminator_sequence_output)
        return logits


class ElectraGenerator(ElectraPretrainedModel):
    """
    The Electra Generator will replace some tokens of the given sequence, it is trained as
    a masked language model.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig
    """

    def __init__(self, config: ElectraConfig):
        super(ElectraGenerator, self).__init__(config)

        self.electra = ElectraModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)

        if not self.tie_word_embeddings:
            self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        else:
            self.generator_lm_head_bias = self.create_parameter(
                shape=[config.vocab_size], dtype=paddle.get_default_dtype(), is_bias=True
            )

    def get_input_embeddings(self):
        return self.electra.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.electra.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ElectraModel`.
            token_type_ids (Tensor, optional):
                See :class:`ElectraModel`.
            position_ids (Tensor, optional):
                See :class:`ElectraModel`.
            attention_mask (Tensor, optional):
                See :class:`ElectraModel`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.
        Returns:
            Tensor: Returns tensor `prediction_scores`, the scores of Electra Generator.
            Its data type should be int64 and its shape is [batch_size, sequence_length, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ElectraGenerator, ElectraTokenizer

                tokenizer = ElectraTokenizer.from_pretrained('electra-small')
                model = ElectraGenerator.from_pretrained('electra-small')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                prediction_scores = model(**inputs)

        """
        generator_sequence_output = self.electra(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(generator_sequence_output, type(input_ids)):
            generator_sequence_output = (generator_sequence_output,)

        prediction_scores = self.generator_predictions(generator_sequence_output[0])
        if not self.tie_word_embeddings:
            prediction_scores = self.generator_lm_head(prediction_scores)
        else:
            prediction_scores = paddle.add(
                paddle.matmul(prediction_scores, self.get_input_embeddings().weight, transpose_y=True),
                self.generator_lm_head_bias,
            )
        loss = None
        # Masked language modeling softmax layer
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(prediction_scores.reshape([-1, self.electra.config["vocab_size"]]), labels.reshape([-1]))

        if not return_dict:
            output = (prediction_scores,) + generator_sequence_output[1:]
            return tuple_output(output, loss)

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_sequence_output.hidden_states,
            attentions=generator_sequence_output.attentions,
        )


class ElectraClassificationHead(nn.Layer):
    """
    Perform sentence-level classification tasks.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig

    """

    def __init__(self, config: ElectraConfig):
        super(ElectraClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.act = get_activation(config.hidden_act)

    def forward(self, features, **kwargs):
        r"""
        The ElectraClassificationHead forward method, overrides the __call__() special method.

        Args:
            features(Tensor):
               Input sequence, usually the `sequence_output` of electra model.
               Its data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

        Returns:
            Tensor: Returns a tensor of the input text classification logits.
            Shape as `[batch_size, num_labels]` and dtype as float32.
        """
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ErnieHealthDiscriminator(ElectraPretrainedModel):
    """
    The Discriminators in ERNIE-Health (https://arxiv.org/abs/2110.07244), including
        - token-level Replaced Token Detection (RTD) task
        - token-level Multi-Token Selection (MTS) task
        - sequence-level Contrastive Sequence Prediction (CSP) task.

    Args:
         config (:class:`ElectraConfig`):
            An instance of ElectraConfig to construct ErnieHealthDiscriminator

    """

    def __init__(self, config: ElectraConfig):
        super(ErnieHealthDiscriminator, self).__init__(config)

        self.electra = ElectraModel(config)
        self.discriminator_rtd = ElectraDiscriminatorPredictions(config)

        self.discriminator_mts = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation_mts = get_activation(config.hidden_act)
        self.bias_mts = nn.Embedding(config.vocab_size, 1)

        self.discriminator_csp = ElectraClassificationHead(config)

    def forward(self, input_ids, candidate_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ElectraModel`.
            candidate_ids (Tensor):
                The candidate indices of input sequence tokens in the vocabulary for MTS task.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                See :class:`ElectraModel`.
            position_ids (Tensor, optional):
                See :class:`ElectraModel`.
            attention_mask (Tensor, optional):
                See :class:`ElectraModel`.

        Returns:
            Tensor: Returns list of tensors, the prediction results of RTD, MTS and CSP.
            The logits' data type should be float32 and if batch_size > 1,
                - the shape of `logits_rtd` is [batch_size, sequence_length],
                - the shape of `logits_mts` is [batch_size, sequence_length, num_candidate],
                - the shape of `logits_csp` is [batch_size, 128].
            If batch_size=1, the shapes are [sequence_length], [sequence_length, num_cadidate],
            [128], separately.

        """

        discriminator_sequence_output = self.electra(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        logits_rtd = self.discriminator_rtd(discriminator_sequence_output)

        cands_embs = self.electra.embeddings.word_embeddings(candidate_ids)
        hidden_mts = self.discriminator_mts(discriminator_sequence_output)
        hidden_mts = self.activation_mts(hidden_mts)
        hidden_mts = paddle.matmul(hidden_mts.unsqueeze(2), cands_embs, transpose_y=True).squeeze(2)
        logits_mts = paddle.add(hidden_mts, self.bias_mts(candidate_ids).squeeze(3))

        logits_csp = self.discriminator_csp(discriminator_sequence_output)

        return logits_rtd, logits_mts, logits_csp


class ElectraForSequenceClassification(ElectraPretrainedModel):
    """
    Electra Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig to construct ElectraForSequenceClassification
    """

    def __init__(self, config: ElectraConfig):
        super(ElectraForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        r"""
        The ElectraForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ElectraModel`.
            token_type_ids (Tensor, optional):
                See :class:`ElectraModel`.
            position_ids(Tensor, optional):
                See :class:`ElectraModel`.
            attention_mask (list, optional):
                See :class:`ElectraModel`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_labels]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ElectraForSequenceClassification
                from paddlenlp.transformers import ElectraTokenizer

                tokenizer = ElectraTokenizer.from_pretrained('electra-small')
                model = ElectraForSequenceClassification.from_pretrained('electra-small')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        sequence_output = self.electra(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(sequence_output, type(input_ids)):
            sequence_output = (sequence_output,)

        logits = self.classifier(sequence_output[0])

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
            output = (logits,) + sequence_output[1:]
            return tuple_output(output, loss)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=sequence_output.hidden_states,
            attentions=sequence_output.attentions,
        )


class ElectraForTokenClassification(ElectraPretrainedModel):
    """
    Electra  Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig to construct ElectraForTokenClassification
    """

    def __init__(self, config: ElectraConfig):
        super(ElectraForTokenClassification, self).__init__(config)
        self.electra = ElectraModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob if config.classifier_dropout is None else config.classifier_dropout
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The ElectraForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ElectraModel`.
            token_type_ids (Tensor, optional):
                See :class:`ElectraModel`.
            position_ids(Tensor, optional):
                See :class:`ElectraModel`.
            attention_mask (list, optional):
                See :class:`ElectraModel`.
            labels (Tensor of shape `(batch_size, )`, optional):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_labels]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ElectraForTokenClassification
                from paddlenlp.transformers import ElectraTokenizer

                tokenizer = ElectraTokenizer.from_pretrained('electra-small')
                model = ElectraForTokenClassification.from_pretrained('electra-small')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        sequence_output = self.electra(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(sequence_output, type(input_ids)):
            sequence_output = (sequence_output,)

        logits = self.classifier(self.dropout(sequence_output[0]))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))

        if not return_dict:
            output = (logits,) + sequence_output[1:]
            return tuple_output(output, loss)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=sequence_output.hidden_states,
            attentions=sequence_output.attentions,
        )


class ElectraForTotalPretraining(ElectraPretrainedModel):
    """
    Electra Model for pretraining tasks.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig to construct ElectraForTotalPretraining

    """

    def __init__(self, config: ElectraConfig):
        super(ElectraForTotalPretraining, self).__init__(config)
        self.generator = ElectraGenerator(config)
        self.discriminator = ElectraDiscriminator(config)
        self.initializer_range = config.initializer_range
        self.tie_weights()

    def get_input_embeddings(self):
        if not self.untied_generator_embeddings:
            return self.generator.electra.embeddings.word_embeddings
        else:
            return None

    def get_output_embeddings(self):
        if not self.untied_generator_embeddings:
            return self.discriminator.electra.embeddings.word_embeddings
        else:
            return None

    def get_discriminator_inputs(self, inputs, raw_inputs, generator_logits, generator_labels, use_softmax_sample):
        # get generator token result
        sampled_tokens = (self.sample_from_softmax(generator_logits, use_softmax_sample)).detach()
        sampled_tokids = paddle.argmax(sampled_tokens, axis=-1)
        # update token only at mask position
        # generator_labels : [B, L], L contains -100(unmasked) or token value(masked)
        # mask_positions : [B, L], L contains 0(unmasked) or 1(masked)
        umask_positions = paddle.zeros_like(generator_labels)
        mask_positions = paddle.ones_like(generator_labels)
        mask_positions = paddle.where(generator_labels == -100, umask_positions, mask_positions)
        updated_inputs = self.update_inputs(inputs, sampled_tokids, mask_positions)
        # use inputs and updated_input to get discriminator labels
        labels = mask_positions * (paddle.ones_like(inputs) - paddle.equal(updated_inputs, raw_inputs).astype("int32"))
        return updated_inputs, labels, sampled_tokids

    def sample_from_softmax(self, logits, use_softmax_sample=True):
        if use_softmax_sample:
            # uniform_noise = paddle.uniform(logits.shape, dtype="float32", min=0, max=1)
            uniform_noise = paddle.rand(logits.shape, dtype=paddle.get_default_dtype())
            gumbel_noise = -paddle.log(-paddle.log(uniform_noise + 1e-9) + 1e-9)
        else:
            gumbel_noise = paddle.zeros_like(logits)
        # softmax_sample equal to sampled_tokids.unsqueeze(-1)
        softmax_sample = paddle.argmax(F.softmax(logits + gumbel_noise), axis=-1)
        # one hot
        return F.one_hot(softmax_sample, logits.shape[-1])

    def update_inputs(self, sequence, updates, positions):
        shape = sequence.shape
        assert len(shape) == 2, "the dimension of inputs should be [B, L]"
        B, L = shape
        N = positions.shape[1]
        assert N == L, "the dimension of inputs and mask should be same as [B, L]"

        updated_sequence = ((paddle.ones_like(sequence) - positions) * sequence) + (positions * updates)

        return updated_sequence

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        raw_input_ids=None,
        generator_labels=None,
    ):
        r"""
        The ElectraForPretraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ElectraModel`.
            token_type_ids (Tensor, optional):
                See :class:`ElectraModel`.
            position_ids(Tensor, optional):
                See :class:`ElectraModel`.
            attention_mask (list, optional):
                See :class:`ElectraModel`.
            raw_input_ids(Tensor, optional):
                Raw inputs used to get discriminator labels.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            generator_labels(Tensor, optional):
                Labels to compute the discriminator inputs.
                Its data type should be int64 and its shape is [batch_size, sequence_length].
                The value for unmasked tokens should be -100 and value for masked tokens should be 0.

        Returns:
            tuple: Returns tuple (generator_logits, disc_logits, disc_labels, attention_mask).

            With the fields:

            - `generator_logits` (Tensor):
                The scores of Electra Generator.
                Its data type should be int64 and its shape is [batch_size, sequence_length, vocab_size].

            - `disc_logits` (Tensor):
                The prediction result of replaced tokens.
                Its data type should be float32 and if batch_size>1, its shape is [batch_size, sequence_length],
                if batch_size=1, its shape is [sequence_length].

            - `disc_labels` (Tensor):
                The labels of electra discriminator. Its data type should be int32,
                and its shape is [batch_size, sequence_length].

            - `attention_mask` (Tensor):
                See :class:`ElectraModel`. Its data type should be bool.

        """

        assert (
            generator_labels is not None
        ), "generator_labels should not be None, please check DataCollatorForLanguageModeling"

        generator_logits = self.generator(input_ids, token_type_ids, position_ids, attention_mask)

        disc_inputs, disc_labels, generator_predict_tokens = self.get_discriminator_inputs(
            input_ids, raw_input_ids, generator_logits, generator_labels, self.use_softmax_sample
        )

        disc_logits = self.discriminator(disc_inputs, token_type_ids, position_ids, attention_mask)

        if attention_mask is None:
            attention_mask = input_ids != self.discriminator.electra.config["pad_token_id"]
        else:
            attention_mask = attention_mask.astype("bool")

        return generator_logits, disc_logits, disc_labels, attention_mask


class ElectraPooler(nn.Layer):
    def __init__(self, config: ElectraConfig):
        super(ElectraPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation(config.hidden_act)
        self.pool_act = config.hidden_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@dataclass
class ErnieHealthForPreTrainingOutput(ModelOutput):
    """
    Output type of [`ErnieHealthForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `paddle.Tensor` of shape `(1,)`):
            Total loss of the ELECTRA objective.
    """

    loss: Optional[paddle.Tensor] = None
    gen_loss: Optional[paddle.Tensor] = None
    disc_rtd_loss: Optional[paddle.Tensor] = None
    disc_mts_loss: Optional[paddle.Tensor] = None
    disc_csp_loss: Optional[paddle.Tensor] = None


class ErnieHealthForTotalPretraining(ElectraForTotalPretraining):
    """
    ERNIE-Health Model for pretraining task.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig to construct ElectraForMultipleChoice
    """

    def __init__(self, config: ElectraConfig):
        super(ErnieHealthForTotalPretraining, self).__init__(config)
        self.generator = ElectraGenerator(config)
        self.discriminator = ErnieHealthDiscriminator(config)
        self.initializer_range = config.initializer_range

    def get_discriminator_inputs_ernie_health(
        self, inputs, raw_inputs, generator_logits, generator_labels, use_softmax_sample
    ):
        updated_inputs, labels, sampled_tokids = self.get_discriminator_inputs(
            inputs, raw_inputs, generator_logits, generator_labels, use_softmax_sample
        )

        # Get negative samples to construct candidates.
        neg_samples_ids = self.sample_negatives_from_softmax(generator_logits, raw_inputs, use_softmax_sample)
        candidate_ids = paddle.concat([raw_inputs.unsqueeze(2), neg_samples_ids], axis=2).detach()
        return updated_inputs, labels, sampled_tokids, candidate_ids

    def sample_negatives_from_softmax(self, logits, raw_inputs, use_softmax_sample=True):
        r"""
        Sample K=5 non-original negative samples for candidate set.

        Returns:
            Tensor: Returns tensor `neg_samples_ids`, a tensor of the negative
            samples of original inputs.
            Shape as ` [batch_size, sequence_length, K, vocab_size]` and dtype
            as `int64`.
        """
        K = 5
        # Initialize excluded_ids by original inputs in one-hot encoding.
        # Its shape is [batch_size, sequence_length, vocab_size].
        excluded_ids = F.one_hot(raw_inputs, logits.shape[-1]) * -10000
        neg_sample_one_hot = None
        neg_samples_ids = None
        for sample in range(K):
            # Update excluded_ids.
            if neg_sample_one_hot is not None:
                excluded_ids = excluded_ids + neg_sample_one_hot * -10000
            if use_softmax_sample:
                uniform_noise = paddle.rand(logits.shape, dtype=paddle.get_default_dtype())
                gumbel_noise = -paddle.log(-paddle.log(uniform_noise + 1e-9) + 1e-9)
            else:
                gumbel_noise = paddle.zeros_like(logits)
            sampled_ids = paddle.argmax(F.softmax(logits + gumbel_noise + excluded_ids), axis=-1)
            # One-hot encoding of sample_ids.
            neg_sample_one_hot = F.one_hot(sampled_ids, logits.shape[-1])
            if neg_samples_ids is None:
                neg_samples_ids = sampled_ids.unsqueeze(2)
            else:
                neg_samples_ids = paddle.concat([neg_samples_ids, sampled_ids.unsqueeze(2)], axis=2)
        return neg_samples_ids

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        raw_input_ids=None,
        generator_labels=None,
        return_dict: Optional[bool] = None,
    ):
        assert generator_labels is not None, "generator_labels should not be None, please check DataCollator"

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        generator_logits = self.generator(input_ids, token_type_ids, position_ids, attention_mask)

        disc_input_list = self.get_discriminator_inputs_ernie_health(
            input_ids, raw_input_ids, generator_logits, generator_labels, self.use_softmax_sample
        )
        disc_inputs, disc_labels, _, disc_candidates = disc_input_list

        logits_rtd, logits_mts, logits_csp = self.discriminator(
            disc_inputs, disc_candidates, token_type_ids, position_ids, attention_mask
        )

        if attention_mask is None:
            pad_id = self.generator.electra.pad_token_id
            attention_mask = input_ids != pad_id
        else:
            attention_mask = attention_mask.astype("bool")

        total_loss = None
        gen_loss = None
        disc_rtd_loss = None
        disc_mts_loss = None
        disc_csp_loss = None

        if generator_labels is not None and disc_labels is not None:
            loss_fct = ErnieHealthPretrainingCriterion(self.config)
            total_loss, gen_loss, disc_rtd_loss, disc_mts_loss, disc_csp_loss = loss_fct(
                generator_logits, generator_labels, logits_rtd, logits_mts, logits_csp, disc_labels, attention_mask
            )

        if not return_dict:
            # return total_loss
            return total_loss, gen_loss, disc_rtd_loss, disc_mts_loss, disc_csp_loss

        return ErnieHealthForPreTrainingOutput(total_loss, gen_loss, disc_rtd_loss, disc_mts_loss, disc_csp_loss)


class ElectraForMultipleChoice(ElectraPretrainedModel):
    """
    Electra Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig to construct ElectraForMultipleChoice
    """

    def __init__(self, config: ElectraConfig):
        super(ElectraForMultipleChoice, self).__init__(config)
        self.num_choices = config.num_choices
        self.electra = ElectraModel(config)
        self.sequence_summary = ElectraPooler(config)
        dropout_p = config.hidden_dropout_prob if config.classifier_dropout is None else config.classifier_dropout
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The ElectraForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ElectraModel` and shape as [batch_size, num_choice, sequence_length].
            token_type_ids (Tensor, optional):
                See :class:`ElectraModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`ElectraModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`ElectraModel` and shape as [batch_size, num_choice, sequence_length].
            labels (Tensor of shape `(batch_size, )`, optional):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            Tensor: Returns tensor `reshaped_logits`, a tensor of the multiple choice classification logits.
            Shape as `[batch_size, num_choice]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ElectraForMultipleChoice, ElectraTokenizer
                from paddlenlp.data import Pad, Dict

                tokenizer = ElectraTokenizer.from_pretrained('electra-small')
                model = ElectraForMultipleChoice.from_pretrained('electra-small', num_choices=2)

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
                print(reshaped_logits.shape)
                # [2, 2]

        """
        input_ids = input_ids.reshape((-1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape((-1, token_type_ids.shape[-1]))
        if position_ids is not None:
            position_ids = position_ids.reshape((-1, position_ids.shape[-1]))
        if attention_mask is not None:
            attention_mask = attention_mask.reshape((-1, attention_mask.shape[-1]))

        sequence_output = self.electra(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(sequence_output, type(input_ids)):
            sequence_output = (sequence_output,)

        pooled_output = self.sequence_summary(sequence_output[0])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape((-1, self.num_choices))  # logits: (bs, num_choice)

        loss = None
        output = (reshaped_logits,) + sequence_output[1:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            output = (loss,) + output

        if not return_dict:
            output = (reshaped_logits,) + sequence_output[1:]
            return tuple_output(output, loss)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=sequence_output.hidden_states,
            attentions=sequence_output.attentions,
        )


class ElectraPretrainingCriterion(paddle.nn.Layer):
    """

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig

    """

    def __init__(self, config: ElectraConfig):
        super(ElectraPretrainingCriterion, self).__init__()

        self.vocab_size = config.vocab_size
        self.gen_weight = config.gen_weight
        self.disc_weight = config.disc_weight
        self.gen_loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.disc_loss_fct = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        generator_prediction_scores,
        discriminator_prediction_scores,
        generator_labels,
        discriminator_labels,
        attention_mask,
    ):
        """
        Args:
            generator_prediction_scores(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].
            discriminator_prediction_scores(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length] or [sequence length] if batch_size=1.
            generator_labels(Tensor):
                The labels of the generator, its dimensionality is equal to `generator_prediction_scores`.
                Its data type should be int64 and its shape is [batch_size, sequence_size, 1].
            discriminator_labels(Tensor):
                The labels of the discriminator, its dimensionality is equal to `discriminator_prediction_scores`.
                The labels should be numbers between 0 and 1.
                Its data type should be float32 and its shape is [batch_size, sequence_size] or [sequence length] if batch_size=1.
            attention_mask(Tensor):
                See :class:`ElectraModel`.

        Returns:
            Tensor: The pretraining loss, equals to weighted generator loss plus the weighted discriminator loss.
            Its data type should be float32 and its shape is [1].

        """
        # generator loss
        gen_loss = self.gen_loss_fct(
            paddle.reshape(generator_prediction_scores, [-1, self.vocab_size]), paddle.reshape(generator_labels, [-1])
        )
        # todo: we can remove 4 lines after when CrossEntropyLoss(reduction='mean') improved
        umask_positions = paddle.zeros_like(generator_labels).astype(paddle.get_default_dtype())
        mask_positions = paddle.ones_like(generator_labels).astype(paddle.get_default_dtype())
        mask_positions = paddle.where(generator_labels == -100, umask_positions, mask_positions)
        if mask_positions.sum() == 0:
            gen_loss = paddle.to_tensor([0.0])
        else:
            gen_loss = gen_loss.sum() / mask_positions.sum()

        # discriminator loss
        seq_length = discriminator_labels.shape[1]
        disc_loss = self.disc_loss_fct(
            paddle.reshape(discriminator_prediction_scores, [-1, seq_length]),
            discriminator_labels.astype(paddle.get_default_dtype()),
        )
        if attention_mask is not None:
            umask_positions = paddle.ones_like(discriminator_labels).astype(paddle.get_default_dtype())
            mask_positions = paddle.zeros_like(discriminator_labels).astype(paddle.get_default_dtype())
            use_disc_loss = paddle.where(attention_mask, disc_loss, mask_positions)
            umask_positions = paddle.where(attention_mask, umask_positions, mask_positions)
            disc_loss = use_disc_loss.sum() / umask_positions.sum()
        else:
            total_positions = paddle.ones_like(discriminator_labels).astype(paddle.get_default_dtype())
            disc_loss = disc_loss.sum() / total_positions.sum()

        return self.gen_weight * gen_loss + self.disc_weight * disc_loss


class ErnieHealthPretrainingCriterion(paddle.nn.Layer):
    """

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig

    """

    def __init__(self, config: ElectraConfig):
        super(ErnieHealthPretrainingCriterion, self).__init__()

        self.vocab_size = config.vocab_size
        self.gen_weight = config.gen_weight
        self.rtd_weight = 50.0
        self.mts_weight = 20.0
        self.csp_weight = 1.0
        self.gen_loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.disc_rtd_loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        self.disc_csp_loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.disc_mts_loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.temperature = 0.07

    def forward(
        self,
        generator_logits,
        generator_labels,
        logits_rtd,
        logits_mts,
        logits_csp,
        discriminator_labels,
        attention_mask,
    ):
        """
        Args:
            generator_logits(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].
            generator_labels(Tensor):
                The labels of the generator, its dimensionality is equal to `generator_prediction_scores`.
                Its data type should be int64 and its shape is [batch_size, sequence_size, 1].
            logits_rtd(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length] or [sequence length] if batch_size=1.
            discriminator_labels(Tensor):
                The labels of the discriminator, its dimensionality is equal to `discriminator_prediction_scores`.
                The labels should be numbers between 0 and 1.
                Its data type should be float32 and its shape is [batch_size, sequence_size] or [sequence length] if batch_size=1.
            attention_mask(Tensor):
                See :class:`ElectraModel`.

        Returns:
            Tensor: The pretraining loss, equals to weighted generator loss plus the weighted discriminator loss.
            Its data type should be float32 and its shape is [1].

        """
        # generator loss
        gen_loss = self.gen_loss_fct(
            paddle.reshape(generator_logits, [-1, self.vocab_size]), paddle.reshape(generator_labels, [-1])
        )
        # todo: we can remove 4 lines after when CrossEntropyLoss(reduction='mean') improved
        umask_positions = paddle.zeros_like(generator_labels).astype(paddle.get_default_dtype())
        mask_positions = paddle.ones_like(generator_labels).astype(paddle.get_default_dtype())
        mask_positions = paddle.where(generator_labels == -100, umask_positions, mask_positions)
        if mask_positions.sum() == 0:
            gen_loss = paddle.to_tensor([0.0])
        else:
            gen_loss = gen_loss.sum() / mask_positions.sum()

        # RTD discriminator loss
        seq_length = discriminator_labels.shape[1]
        rtd_labels = discriminator_labels

        disc_rtd_loss = self.disc_rtd_loss_fct(
            paddle.reshape(logits_rtd, [-1, seq_length]), rtd_labels.astype(logits_rtd.dtype)
        )
        if attention_mask is not None:
            umask_positions = paddle.ones_like(rtd_labels).astype(paddle.get_default_dtype())
            mask_positions = paddle.zeros_like(rtd_labels).astype(paddle.get_default_dtype())
            umask_positions = paddle.where(attention_mask, umask_positions, mask_positions)
            # Mask has different meanings here. It denotes [mask] token in
            # generator and denotes [pad] token in discriminator.
            disc_rtd_loss = paddle.where(attention_mask, disc_rtd_loss, mask_positions)
            disc_rtd_loss = disc_rtd_loss.sum() / umask_positions.sum()
        else:
            total_positions = paddle.ones_like(rtd_labels).astype(paddle.get_default_dtype())
            disc_rtd_loss = disc_rtd_loss.sum() / total_positions.sum()

        # MTS discriminator loss
        replaced_positions = discriminator_labels.astype("bool")
        mts_labels = paddle.zeros([logits_mts.shape[0] * logits_mts.shape[1]], dtype=generator_labels.dtype).detach()
        disc_mts_loss = self.disc_mts_loss_fct(paddle.reshape(logits_mts, [-1, logits_mts.shape[-1]]), mts_labels)
        disc_mts_loss = paddle.reshape(disc_mts_loss, [-1, seq_length])
        original_positions = paddle.zeros_like(replaced_positions).astype(paddle.get_default_dtype())
        disc_mts_loss = paddle.where(replaced_positions, disc_mts_loss, original_positions)
        if discriminator_labels.sum() == 0:
            disc_mts_loss = paddle.to_tensor([0.0])
        else:
            disc_mts_loss = disc_mts_loss.sum() / discriminator_labels.sum()

        # CSP discriminator loss
        logits_csp = F.normalize(logits_csp, axis=-1)
        # Gather from all devices (split first)
        logit_csp_0, logit_csp_1 = paddle.split(logits_csp, num_or_sections=2, axis=0)
        if paddle.distributed.get_world_size() > 1:
            csp_list_0, csp_list_1 = [], []
            paddle.distributed.all_gather(csp_list_0, logit_csp_0)
            paddle.distributed.all_gather(csp_list_1, logit_csp_1)
            logit_csp_0 = paddle.concat(csp_list_0, axis=0)
            logit_csp_1 = paddle.concat(csp_list_1, axis=0)
        batch_size = logit_csp_0.shape[0]
        logits_csp = paddle.concat([logit_csp_0, logit_csp_1], axis=0)
        # Similarity matrix
        logits_csp = paddle.matmul(logits_csp, logits_csp, transpose_y=True)
        # Temperature scale
        logits_csp = logits_csp / self.temperature
        # Mask self-contrast
        mask = -1e4 * paddle.eye(logits_csp.shape[0])
        logits_csp = logits_csp + mask
        # Create labels for bundle
        csp_labels = paddle.concat([paddle.arange(batch_size, 2 * batch_size), paddle.arange(batch_size)], axis=0)
        # Calculate SimCLR loss
        disc_csp_loss = self.disc_csp_loss_fct(logits_csp, csp_labels)
        disc_csp_loss = disc_csp_loss.sum() / (batch_size * 2)

        loss = (
            self.gen_weight * gen_loss
            + self.rtd_weight * disc_rtd_loss
            + self.mts_weight * disc_mts_loss
            + self.csp_weight * disc_csp_loss
        )

        return loss, gen_loss, disc_rtd_loss, disc_mts_loss, disc_csp_loss


class ElectraForQuestionAnswering(ElectraPretrainedModel):
    """
    Electra Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`ElectraConfig`):
            An instance of ElectraConfig used to construct ElectraForQuestionAnswering.

    """

    def __init__(self, config: ElectraConfig):
        super(ElectraForQuestionAnswering, self).__init__(config)
        self.electra = ElectraModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The ElectraForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ElectraModel`.
            token_type_ids (Tensor, optional):
                See :class:`ElectraModel`.
            position_ids(Tensor, optional):
                See :class:`ElectraModel`.
            attention_mask (list, optional):
                See :class:`ElectraModel`.
            start_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (Tensor of shape `(batch_size,)`, optional):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.
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
                from paddlenlp.transformers import ElectraForQuestionAnswering, ElectraTokenizer

                tokenizer = ElectraTokenizer.from_pretrained('electra-small')
                model = ElectraForQuestionAnswering.from_pretrained('electra-small')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits  = outputs[1]

        """
        sequence_output = self.electra(
            input_ids,
            token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(sequence_output, type(input_ids)):
            sequence_output = (sequence_output,)

        logits = self.classifier(sequence_output[0])
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if start_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = paddle.shape(start_logits)[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + sequence_output[2:]
            return tuple_output(output, total_loss)

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=sequence_output.hidden_states,
            attentions=sequence_output.attentions,
        )


# ElectraForMaskedLM is the same as ElectraGenerator
ElectraForMaskedLM = ElectraGenerator
ElectraForPretraining = ElectraForTotalPretraining
