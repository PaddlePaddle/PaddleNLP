# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""Modeling classes for ALBERT model."""

import math
from typing import List, Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer

from ...layers import Linear as TransposedLinear
from ...utils.converter import StateDictNameMapping, init_name_mappings
from ...utils.env import CONFIG_NAME
from .. import PretrainedModel, register_base_model
from ..activations import ACT2FN
from ..model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    tuple_output,
)
from .configuration import (
    ALBERT_PRETRAINED_INIT_CONFIGURATION,
    ALBERT_PRETRAINED_RESOURCE_FILES_MAP,
    AlbertConfig,
)

__all__ = [
    "AlbertPretrainedModel",
    "AlbertModel",
    "AlbertForPretraining",
    "AlbertForMaskedLM",
    "AlbertForSequenceClassification",
    "AlbertForTokenClassification",
    "AlbertForQuestionAnswering",
    "AlbertForMultipleChoice",
]

dtype_float = paddle.get_default_dtype()


class AlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`AlbertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `paddle.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`paddle.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(paddle.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `paddle.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(paddle.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `paddle.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[paddle.Tensor] = None
    prediction_logits: paddle.Tensor = None
    sop_logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


class AlbertEmbeddings(Layer):
    """
    Constructs the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: AlbertConfig):
        super(AlbertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        self.layer_norm = nn.LayerNorm(config.embedding_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", paddle.arange(config.max_position_embeddings, dtype="int64").expand((1, -1))
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
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

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertAttention(Layer):
    def __init__(self, config: AlbertConfig):
        super(AlbertAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    # Copied from transformers.models.bert.modeling_bert.BertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        context_layer = context_layer.reshape([0, 0, -1])

        # dense layer shape to be checked
        projected_context_layer = self.dense(context_layer)

        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layer_normed_context_layer = self.layer_norm(hidden_states + projected_context_layer_dropout)
        return (layer_normed_context_layer, attention_probs) if output_attentions else (layer_normed_context_layer,)


class AlbertLayer(Layer):
    def __init__(self, config: AlbertConfig):
        super(AlbertLayer, self).__init__()
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class AlbertLayerGroup(Layer):
    def __init__(self, config: AlbertConfig):
        super(AlbertLayerGroup, self).__init__()

        self.albert_layers = nn.LayerList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False
    ):
        layer_attentions = () if output_attentions else None
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(
                hidden_states,
                attention_mask,
                head_mask[layer_index],
                output_attentions=output_attentions,
            )
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)

        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        if output_attentions:
            outputs = outputs + (layer_attentions,)

        return outputs


class AlbertTransformer(Layer):
    def __init__(self, config: AlbertConfig):
        super(AlbertTransformer, self).__init__()

        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups

        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.LayerList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=False,
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i in range(self.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.num_hidden_layers / self.num_hidden_groups)
            # Index of the hidden group
            group_idx = int(i / (self.num_hidden_layers / self.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class AlbertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ALBERT models. It provides ALBERT related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = CONFIG_NAME
    config_class = AlbertConfig

    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "transformer"

    pretrained_init_configuration = ALBERT_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ALBERT_PRETRAINED_RESOURCE_FILES_MAP

    @classmethod
    def _get_name_mappings(cls, config: AlbertConfig) -> List[StateDictNameMapping]:
        model_mappings = [
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            ["embeddings.LayerNorm.weight", "embeddings.layer_norm.weight"],
            ["embeddings.LayerNorm.bias", "embeddings.layer_norm.bias"],
            ["encoder.embedding_hidden_mapping_in.weight", None, "transpose"],
            "encoder.embedding_hidden_mapping_in.bias",
        ]

        if config.add_pooling_layer:
            model_mappings.extend(
                [
                    ["pooler.weight", None, "transpose"],
                    ["pooler.bias"],
                ]
            )

        for group_index in range(config.num_hidden_groups):
            group_mappings = [
                f"encoder.albert_layer_groups.{group_index}.albert_layers.0.full_layer_layer_norm.weight",
                f"encoder.albert_layer_groups.{group_index}.albert_layers.0.full_layer_layer_norm.bias",
                [
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.query.weight",
                    None,
                    "transpose",
                ],
                f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.query.bias",
                [
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.key.weight",
                    None,
                    "transpose",
                ],
                f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.key.bias",
                [
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.value.weight",
                    None,
                    "transpose",
                ],
                f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.value.bias",
                [
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.dense.weight",
                    None,
                    "transpose",
                ],
                f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.dense.bias",
                [
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.LayerNorm.weight",
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.layer_norm.weight",
                ],
                [
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.LayerNorm.bias",
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.attention.layer_norm.bias",
                ],
                [
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.ffn.weight",
                    None,
                    "transpose",
                ],
                f"encoder.albert_layer_groups.{group_index}.albert_layers.0.ffn.bias",
                [
                    f"encoder.albert_layer_groups.{group_index}.albert_layers.0.ffn_output.weight",
                    None,
                    "transpose",
                ],
                f"encoder.albert_layer_groups.{group_index}.albert_layers.0.ffn_output.bias",
            ]
            model_mappings.extend(group_mappings)

        init_name_mappings(model_mappings)
        # base-model prefix "AlbertModel"
        if "AlbertModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "albert." + mapping[0]
                mapping[1] = "transformer." + mapping[1]

        # downstream mappings
        if "AlbertForQuestionAnswering" in config.architectures:
            model_mappings.extend(
                [["qa_outputs.weight", "qa_outputs.weight", "transpose"], ["qa_outputs.bias", "qa_outputs.bias"]]
            )
        if (
            "AlbertForMultipleChoice" in config.architectures
            or "AlbertForSequenceClassification" in config.architectures
            or "AlbertForTokenClassification" in config.architectures
        ):
            model_mappings.extend(
                [["classifier.weight", "classifier.weight", "transpose"], ["classifier.bias", "classifier.bias"]]
            )

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def _init_weights(self, layer):
        # Initialize the weights.
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=layer.weight.shape,
                )
            )
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=layer.weight.shape,
                )
            )
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(paddle.zeros_like(layer.weight[layer._padding_idx]))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))


@register_base_model
class AlbertModel(AlbertPretrainedModel):
    """
    The bare Albert Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`AlbertConfig`):
            An instance of AlbertConfig used to construct AlbertModel.
    """

    def __init__(self, config: AlbertConfig):
        super(AlbertModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.initializer_range = config.initializer_range
        self.num_hidden_layers = config.num_hidden_layers
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.config = config

        if config.add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = paddle.cast(head_mask, dtype=dtype_float)
        return head_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=False,
    ):
        r"""
        The AlbertModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
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
            head_mask (Tensor, optional):
                Mask to nullify selected heads of the self-attention modules. Masks values can either be 0 or 1:

                - 1 indicates the head is **not masked**,
                - 0 indicated the head is **masked**.
            inputs_embeds (Tensor, optional):
               If you want to control how to convert `inputs_ids` indices into associated vectors, you can
               pass an embedded representation directly instead of passing `inputs_ids`.
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
            tuple or Dict: Returns tuple (`sequence_output`, `pooled_output`) or a dict with
            `last_hidden_state`, `pooled_output`, `all_hidden_states`, `all_attentions` fields.

            With the fields:

            - `sequence_output` (Tensor):
               Sequence of hidden-states at the last layer of the model.
               It's data type should be float32 and has a shape of [`batch_size, sequence_length, hidden_size`].

            - `pooled_output` (Tensor):
               The output of first token (`[CLS]`) in sequence.
               We "pool" the model by simply taking the hidden state corresponding to the first token.
               Its data type should be float32 and
               has a shape of [batch_size, hidden_size].

            - `last_hidden_state` (Tensor):
               The output of the last encoder layer, it is also the `sequence_output`.
               It's data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].

            - `all_hidden_states` (Tensor):
               Hidden_states of all layers in the Transformer encoder. The length of `all_hidden_states` is `num_hidden_layers + 1`.
               For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `all_attentions` (Tensor):
               Attentions of all layers of in the Transformer encoder. The length of `all_attentions` is `num_hidden_layers`.
               For all element in the tuple, its data type should be float32 and its shape is
               [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import AlbertModel, AlbertTokenizer

                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
                model = AlbertModel.from_pretrained('albert-base-v1')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(shape=input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(shape=input_shape, dtype="int64")

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = paddle.cast(extended_attention_mask, dtype=dtype_float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AlbertForPretraining(AlbertPretrainedModel):
    """
    Albert Model with a `masked language modeling` head and a `sentence order prediction` head
    on top.

    Args:
        config (:class:`AlbertConfig`):
            An instance of AlbertConfig used to construct AlbertModel.

    """

    def __init__(self, config: AlbertConfig):
        super(AlbertForPretraining, self).__init__(config)

        self.transformer = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)
        self.config = config
        self.vocab_size = config.vocab_size

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sentence_order_label=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The AlbertForPretraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            sentence_order_label(Tensor, optional):
                Labels of the next sequence prediction. Input should be a sequence pair
                Indices should be 0 or 1. ``0`` indicates original order (sequence A, then sequence B),
                and ``1`` indicates switched order (sequence B, then sequence A). Defaults to `None`.
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
            tuple or Dict: Returns tuple (`prediction_scores`, `sop_scores`) or a dict with
            `prediction_logits`, `sop_logits`, `pooled_output`, `hidden_states`, `attentions` fields.

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `sop_scores` (Tensor):
                The scores of sentence order prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

            - `prediction_logits` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `sop_logits` (Tensor):
                The scores of sentence order prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        """

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        total_loss = None
        if labels is not None and sentence_order_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.reshape([-1, self.config.vocab_size]), labels.reshape([-1]))
            sentence_order_loss = loss_fct(sop_scores.reshape([-1, 2]), sentence_order_label.reshape([-1]))
            total_loss = masked_lm_loss + sentence_order_loss

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return tuple_output(output, total_loss)

        return AlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AlbertMLMHead(Layer):
    def __init__(self, config: AlbertConfig):
        super(AlbertMLMHead, self).__init__()

        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.bias = self.create_parameter(
            [config.vocab_size], is_bias=True, default_initializer=nn.initializer.Constant(value=0)
        )
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = TransposedLinear(config.embedding_size, config.vocab_size)

        self.activation = ACT2FN[config.hidden_act]

        # link bias
        self.bias = self.decoder.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states
        return prediction_scores


class AlbertSOPHead(Layer):
    def __init__(self, config: AlbertConfig):
        super(AlbertSOPHead, self).__init__()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class AlbertForMaskedLM(AlbertPretrainedModel):
    """
    Albert Model with a `masked language modeling` head on top.

    Args:
        config (:class:`AlbertConfig`):
            An instance of AlbertConfig used to construct AlbertModel.

    """

    def __init__(self, config: AlbertConfig):
        super(AlbertForMaskedLM, self).__init__(config)

        self.transformer = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.config = config
        self.tie_weights()

    def get_output_embeddings(self):
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=False,
    ):
        r"""
        The AlbertForPretraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
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
            Tensor or Dict: Returns tensor `prediction_scores` or a dict with `logits`,
            `hidden_states`, `attentions` fields.

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `logits` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                and its shape is [batch_size, sequence_length, vocab_size].

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        """

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(transformer_outputs, type(input_ids)):
            transformer_outputs = [transformer_outputs]

        hidden_states = transformer_outputs[0]
        logits = self.predictions(hidden_states)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(logits.reshape((-1, logits.shape[-1])), labels.reshape((-1,)))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return tuple_output(output, masked_lm_loss)

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class AlbertForSequenceClassification(AlbertPretrainedModel):
    """
    Albert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`AlbertConfig`):
            An instance of AlbertConfig used to construct AlbertModel.

    """

    def __init__(self, config: AlbertConfig):
        super(AlbertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=False,
    ):
        r"""
        The AlbertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            labels (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the sequence classification/regression loss.
                Indices should be in `[0, ..., num_labels - 1]`. If `num_labels == 1`
                a regression loss is computed (Mean-Square loss), If `num_labels > 1`
                a classification loss is computed (Cross-Entropy).
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
            Tensor or Dict: Returns tensor `logits`, or a dict with `logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `logits` (Tensor):
                A tensor of the input text classification logits.
                Shape as `[batch_size, num_labels]` and dtype as float32.

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import AlbertForSequenceClassification, AlbertTokenizer

                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
                model = AlbertForSequenceClassification.from_pretrained('albert-base-v1')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = transformer_outputs[1]
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
            output = (logits,) + transformer_outputs[2:]
            return tuple_output(output, loss)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class AlbertForTokenClassification(AlbertPretrainedModel):
    """
    Albert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`AlbertConfig`):
            An instance of AlbertConfig used to construct AlbertModel.
    """

    def __init__(self, config: AlbertConfig):
        super(AlbertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=False,
    ):
        r"""
        The AlbertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the token classification loss. Indices should be in `[0, ..., num_labels - 1]`.
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
            Tensor or Dict: Returns tensor `logits`, or a dict with `logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `logits` (Tensor):
                A tensor of the input token classification logits.
                Shape as `[batch_size, sequence_length, num_labels]` and dtype as `float32`.

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import AlbertForTokenClassification, AlbertTokenizer

                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
                model = AlbertForTokenClassification.from_pretrained('albert-base-v1')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = transformer_outputs[0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1,)))
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return tuple_output(output, loss)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class AlbertForQuestionAnswering(AlbertPretrainedModel):
    """
    Albert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`AlbertConfig`):
            An instance of AlbertConfig used to construct AlbertModel.

    """

    def __init__(self, config: AlbertConfig):
        super(AlbertForQuestionAnswering, self).__init__(config)
        self.config = config
        self.transformer = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=False,
    ):
        r"""
        The AlbertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
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
            tuple or Dict: Returns tuple (`start_logits, end_logits`)or a dict
            with `start_logits`, `end_logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].


        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import AlbertForQuestionAnswering, AlbertTokenizer

                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
                model = AlbertForQuestionAnswering.from_pretrained('albert-base-v1')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = transformer_outputs[0]
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = paddle.split(logits, num_or_sections=2, axis=-1)
        start_logits = start_logits.squeeze(axis=-1)
        end_logits = start_logits.squeeze(axis=-1)

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
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return tuple_output(output, total_loss)

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class AlbertForMultipleChoice(AlbertPretrainedModel):
    """
    Albert Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like SWAG tasks .

    Args:
        config (:class:`AlbertConfig`):
            An instance of AlbertConfig used to construct AlbertModel.

    """

    def __init__(self, config: AlbertConfig):
        super(AlbertForMultipleChoice, self).__init__(config)
        self.transformer = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=False,
    ):
        r"""
        The AlbertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`AlbertModel`.
            attention_mask (list, optional):
                See :class:`AlbertModel`.
            token_type_ids (Tensor, optional):
                See :class:`AlbertModel`.
            position_ids(Tensor, optional):
                See :class:`AlbertModel`.
            head_mask(Tensor, optional):
                See :class:`AlbertModel`.
            inputs_embeds(Tensor, optional):
                See :class:`AlbertModel`.
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
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.

        Returns:
            Tensor or Dict: Returns tensor `reshaped_logits` or a dict
            with `reshaped_logits`, `hidden_states`, `attentions` fields.

            With the fields:

            - `reshaped_logits` (Tensor):
                A tensor of the input multiple choice classification logits.
                Shape as `[batch_size, num_labels]` and dtype as `float32`.

            - `hidden_states` (Tensor):
                Hidden_states of all layers in the Transformer encoder. The length of `hidden_states` is `num_hidden_layers + 1`.
                For all element in the tuple, its data type should be float32 and its shape is [`batch_size, sequence_length, hidden_size`].

            - `attentions` (Tensor):
                Attentions of all layers of in the Transformer encoder. The length of `attentions` is `num_hidden_layers`.
                For all element in the tuple, its data type should be float32 and its shape is
                [`batch_size, num_attention_heads, sequence_length, sequence_length`].
        """

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape([-1, input_ids.shape[-1]]) if input_ids is not None else None
        attention_mask = attention_mask.reshape([-1, attention_mask.shape[-1]]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape([-1, token_type_ids.shape[-1]]) if token_type_ids is not None else None
        position_ids = position_ids.reshape([-1, position_ids.shape[-1]]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.reshape([-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]])
            if inputs_embeds is not None
            else None
        )
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = transformer_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape([-1, num_choices])

        loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[2:]
            return tuple_output(output, loss)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
