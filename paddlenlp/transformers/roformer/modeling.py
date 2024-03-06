# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional, Tuple

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.common_ops_import import convert_dtype

from ...utils.converter import StateDictNameMapping, init_name_mappings
from .. import PretrainedModel, register_base_model
from ..activations import get_activation
from ..model_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    tuple_output,
)

__all__ = [
    "RoFormerModel",
    "RoFormerPretrainedModel",
    "RoFormerForSequenceClassification",
    "RoFormerForTokenClassification",
    "RoFormerForQuestionAnswering",
    "RoFormerForMaskedLM",
    "RoFormerForMultipleChoice",
    "RoFormerForCausalLM",
]
from .configuration import (
    ROFORMER_PRETRAINED_INIT_CONFIGURATION,
    ROFORMER_PRETRAINED_RESOURCE_FILES_MAP,
    RoFormerConfig,
)


class RoFormerEmbeddings(nn.Layer):
    """
    Include embeddings from word and token_type embeddings
    """

    def __init__(
        self,
        vocab_size,
        embedding_size=768,
        hidden_dropout_prob=0.1,
        type_vocab_size=2,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids_shape = paddle.shape(inputs_embeds)[:-1]
            token_type_ids = paddle.zeros(token_type_ids_shape, dtype="int64")

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RotaryPositionEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2, dtype=paddle.get_default_dtype()) / dim))
        t = paddle.arange(max_position_embeddings, dtype=paddle.get_default_dtype())
        freqs = paddle.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
        self.register_buffer("sin", freqs.sin(), persistable=False)
        self.register_buffer("cos", freqs.cos(), persistable=False)

    def forward(self, x, offset=0):
        # x shape [batch_size, num_heads, seqlen, head_dim]
        seqlen = paddle.shape(x)[-2]
        sin, cos = (
            self.sin[offset : offset + seqlen, :],
            self.cos[offset : offset + seqlen, :],
        )
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # [cos_nθ, -sin_nθ] [x1]
        # [sin_nθ,  cos_nθ] [x2]
        # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
        return paddle.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).flatten(-2, -1)


class MultiHeadAttentionWithRotary(nn.MultiHeadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
        rotary_value=False,
        max_position_embeddings=512,
    ):
        super().__init__(embed_dim, num_heads, dropout, kdim, vdim, need_weights)
        self.rotary_value = rotary_value
        self.rotary = RotaryPositionEmbedding(self.head_dim, max_position_embeddings)

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)
        q = paddle.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])

        k, v = self.compute_kv(key, value)

        offset = 0 if cache is None else cache.k.shape[2]

        # rotary q,k,v
        q = self.rotary(q, offset=offset)
        k = self.rotary(k, offset=offset)
        if self.rotary_value:
            v = self.rotary(v, offset=offset)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = paddle.concat([cache.k, k], axis=2)
            v = paddle.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)

        return (q, k, v) if cache is None else (q, k, v, cache)


class TransformerEncoderLayerWithRotary(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=False,
        rotary_value=False,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            activation=activation,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            normalize_before=normalize_before,
        )
        self.self_attn = MultiHeadAttentionWithRotary(
            d_model,
            nhead,
            dropout=attn_dropout,
            rotary_value=rotary_value,
            max_position_embeddings=max_position_embeddings,
        )
        self._config.update({"rotary_value": rotary_value, "max_position_embeddings": max_position_embeddings})


class RoFormerPooler(nn.Layer):
    def __init__(self, hidden_size, pool_act="tanh"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = get_activation(pool_act)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RoFormerLMPredictionHead(nn.Layer):
    def __init__(self, embedding_size, hidden_size, vocab_size, activation, embedding_weights=None):
        super().__init__()
        self.transform = nn.Linear(hidden_size, embedding_size)
        self.activation = get_activation(activation)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.decoder_weight = (
            self.create_parameter(
                shape=[vocab_size, embedding_size],
                dtype=self.transform.weight.dtype,
                is_bias=False,
            )
            if embedding_weights is None
            else embedding_weights
        )
        self.decoder_bias = self.create_parameter(shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.matmul(hidden_states, self.decoder_weight, transpose_y=True) + self.decoder_bias
        return hidden_states


class RoFormerOnlyMLMHead(nn.Layer):
    def __init__(self, embedding_size, hidden_size, vocab_size, activation, embedding_weights):
        super().__init__()
        self.predictions = RoFormerLMPredictionHead(
            embedding_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            activation=activation,
            embedding_weights=embedding_weights,
        )

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RoFormerPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained RoFormer models. It provides RoFormer related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """
    config_class = RoFormerConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "roformer"
    pretrained_init_configuration = ROFORMER_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ROFORMER_PRETRAINED_RESOURCE_FILES_MAP

    @classmethod
    def _get_name_mappings(cls, config: RoFormerConfig) -> List[StateDictNameMapping]:
        mappings: List[StateDictNameMapping] = []
        model_mappings = [
            "embeddings.word_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            ["embeddings.LayerNorm.weight", "embeddings.layer_norm.weight"],
            ["embeddings.LayerNorm.bias", "embeddings.layer_norm.bias"],
            ["pooler.dense.weight", None, "transpose"],
            "pooler.dense.bias",
            # for TokenClassification
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
                    f"encoder.layer.{layer_index}.intermediate.dense.weight",
                    f"encoder.layers.{layer_index}.linear1.weight",
                    "transpose",
                ],
                [f"encoder.layer.{layer_index}.intermediate.dense.bias", f"encoder.layers.{layer_index}.linear1.bias"],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.weight",
                    f"encoder.layers.{layer_index}.norm1.weight",
                ],
                [
                    f"encoder.layer.{layer_index}.attention.output.LayerNorm.bias",
                    f"encoder.layers.{layer_index}.norm1.bias",
                ],
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

        # base-model prefix "RoFormerModel"
        if "RoFormerModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "roformer." + mapping[0]
                mapping[1] = "roformer." + mapping[1]

        if "RoFormerForMaskedLM" in config.architectures:
            model_mappings.extend(
                [
                    ["cls.predictions.transform.dense.weight", "cls.predictions.transform.weight", "transpose"],
                    ["cls.predictions.transform.dense.bias", "cls.predictions.transform.bias"],
                    ["cls.predictions.transform.LayerNorm.weight", "cls.predictions.layer_norm.weight"],
                    ["cls.predictions.transform.LayerNorm.bias", "cls.predictions.layer_norm.bias"],
                    ["cls.predictions.decoder.bias", "cls.predictions.decoder_bias"],
                ]
            )
        # downstream mappings
        if "RoFormerForQuestionAnswering" in config.architectures:
            model_mappings.extend(
                [["qa_outputs.weight", "classifier.weight", "transpose"], ["qa_outputs.bias", "classifier.bias"]]
            )
        if (
            "RoFormerForMultipleChoice" in config.architectures
            or "RoFormerForSequenceClassification" in config.architectures
            or "RoFormerForTokenClassification" in config.architectures
        ):
            model_mappings.extend([["classifier.weight", None, "transpose"]])

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
                    paddle.normal(
                        mean=0.0,
                        std=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )

        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = self.config.layer_norm_eps


@register_base_model
class RoFormerModel(RoFormerPretrainedModel):
    """
    The bare RoFormerModel outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`RoFormerConfig`):
            An instance of RoFormerConfig used to construct RoFormerModel.
    """

    def __init__(
        self,
        config: RoFormerConfig,
    ):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.initializer_range = config.initializer_range
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        self.embeddings = RoFormerEmbeddings(
            config.vocab_size,
            config.embedding_size,
            config.hidden_dropout_prob,
            config.type_vocab_size,
        )
        encoder_layer = TransformerEncoderLayerWithRotary(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            attn_dropout=config.attention_probs_dropout_prob,
            act_dropout=0,
            rotary_value=config.rotary_value,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        self.pooler = RoFormerPooler(config.hidden_size, config.pool_act)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The RoFormerModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor, optional):
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
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.ModelOutput` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPoolingAndCrossAttentions`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerModel, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-char-base')
                model = RoFormerModel.from_pretrained('roformer-chinese-char-base')

                tokenized_inputs = tokenizer("欢迎使用百度飞桨!", return_tensors="pd")
                output = model(**tokenized_inputs)

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
            )
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)

        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if hasattr(self, "embeddings_project"):
            embedding_output = self.embeddings_project(embedding_output)

        self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
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


class RoFormerForQuestionAnswering(RoFormerPretrainedModel):
    r"""
    RoFormer Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
     and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`RoFormerConfig`):
            An instance of RoFormerConfig used to construct RoFormerForQuestionAnswering.
    """

    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The RoFormerForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel`.
            inputs_embeds(Tensor, optional):
                See :class:`RoFormerModel`.
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
            An instance of :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.QuestionAnsweringModelOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForQuestionAnswering, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-char-base')
                model = RoFormerForQuestionAnswering.from_pretrained('roformer-chinese-char-base')

                tokenized_inputs = tokenizer("欢迎使用百度飞桨!", return_tensors="pd")
                outputs = model(**tokenized_inputs)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)
        start_logits, end_logits = paddle.unstack(x=logits, axis=-1)

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
            output = (start_logits, end_logits) + outputs[2:]
            return tuple_output(output, total_loss)

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RoFormerForSequenceClassification(RoFormerPretrainedModel):
    r"""
    RoFormer Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`RoFormerConfig`):
            An instance of RoFormerConfig used to construct RoFormerForSequenceClassification.
    """

    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roformer = RoFormerModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The RoFormerForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel`.
            inputs_embeds(Tensor, optional):
                See :class:`RoFormerModel`.
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
            An instance of :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.SequenceClassifierOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForSequenceClassification, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-char-base')
                model = RoFormerForSequenceClassification.from_pretrained('roformer-chinese-char-base')

                tokenized_inputs = tokenizer("欢迎使用百度飞桨!", return_tensors="pd")
                logits = model(**tokenized_inputs)

        """
        outputs = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
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
            return tuple_output(output, loss)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RoFormerForTokenClassification(RoFormerPretrainedModel):
    r"""
    RoFormer Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`RoFormerConfig`):
            An instance of RoFormerConfig used to construct RoFormerForTokenClassification.
    """

    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The RoFormerForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel`.
            inputs_embeds(Tensor, optional):
                See :class:`RoFormerModel`.
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
            An instance of :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.TokenClassifierOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForTokenClassification, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-char-base')
                model = RoFormerForTokenClassification.from_pretrained('roformer-chinese-char-base')

                tokenized_inputs = tokenizer("欢迎使用百度飞桨!", return_tensors="pd")
                logits = model(**tokenized_inputs)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
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
            return tuple_output(output, loss)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RoFormerForMultipleChoice(RoFormerPretrainedModel):
    """
    RoFormerModel with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        config (:class:`RoFormerConfig`):
            An instance of RoFormerConfig used to construct RoFormerForMultipleChoice.
    """

    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.num_choices = config.num_choices
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The RoFormerForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel` and shape as [batch_size, num_choice, sequence_length].
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel` and shape as [batch_size, num_choice, sequence_length].
            inputs_embeds(Tensor, optional):
                See :class:`RoFormerModel`.
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
            An instance of :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.MultipleChoiceModelOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForMultipleChoice, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-char-base')
                model = RoFormerForMultipleChoice.from_pretrained('roformer-chinese-char-base')

                data = [
                    {
                        "question": "如何打开ipad屏幕？",
                        "answer1": "按音量按钮。",
                        "answer2": "按下锁定按钮。",
                        "label": 1,
                    },
                    {
                        "question": "如何缩进一些文本？",
                        "answer1": "在开始写之前留一些空格。",
                        "answer2": "按空格键。",
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

                tokenized_inputs = tokenizer(text, text_pair, padding=True, return_tensors="pd")
                reshaped_logits = model(**tokenized_inputs)
                print(reshaped_logits.shape)
                # [2, 2]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids = input_ids.reshape((-1, input_ids.shape[-1])) if input_ids is not None else None
        token_type_ids = token_type_ids.reshape((-1, token_type_ids.shape[-1])) if token_type_ids is not None else None
        attention_mask = attention_mask.reshape((-1, attention_mask.shape[-1])) if attention_mask is not None else None

        outputs = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape((-1, self.num_choices))

        loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return tuple_output(output, loss)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RoFormerForMaskedLM(RoFormerPretrainedModel):
    """
    RoFormer Model with a `masked language modeling` head on top.

    Args:
        config (:class:`RoFormerConfig`):
            An instance of RoFormerConfig used to construct RoFormerForMaskedLM.

    """

    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.cls = RoFormerOnlyMLMHead(
            config.embedding_size,
            config.hidden_size,
            config.vocab_size,
            config.hidden_act,
            embedding_weights=self.roformer.embeddings.word_embeddings.weight,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The RoFormerForMaskedLM forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel`.
            inputs_embeds(Tensor, optional):
                See :class:`RoFormerModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., vocab_size]`
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.


        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.MaskedLMOutput`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForMaskedLM, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-char-base')
                model = RoFormerForMaskedLM.from_pretrained('roformer-chinese-char-base')

                tokenized_inputs = tokenizer("欢迎使用百度飞桨!", return_tensors="pd")
                logits = model(**tokenized_inputs)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.reshape((-1, prediction_scores.shape[-1])), labels.reshape((-1,))
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return tuple_output(output, masked_lm_loss)

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RoFormerForCausalLM(RoFormerPretrainedModel):
    """
    RoFormer Model with a `Causal language modeling` head on top.

    Args:
        config (:class:`RoFormerConfig`):
            An instance of RoFormerConfig used to construct RoFormerForCausalLM.

    """

    def __init__(self, config: RoFormerConfig):
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.cls = RoFormerOnlyMLMHead(
            config.embedding_size,
            config.hidden_size,
            config.vocab_size,
            config.hidden_act,
            embedding_weights=self.roformer.embeddings.word_embeddings.weight,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The RoFormerForCausalLM forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerModel`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerModel`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerModel`.
            inputs_embeds(Tensor, optional):
                See :class:`RoFormerModel`.
            labels (Tensor of shape `(batch_size, sequence_length)`, optional):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., vocab_size]`.
            past_key_values (tuple(tuple(Tensor)), optional):
                See :class:`RoFormerModel`.
            use_cache (Tensor, optional):
                See :class:`RoFormerModel`.
            output_hidden_states (bool, optional):
                Whether to return the hidden states of all layers.
                Defaults to `False`.
            output_attentions (bool, optional):
                Whether to return the attentions tensors of all attention layers.
                Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithCrossAttentions` object. If
                `False`, the output will be a tuple of tensors. Defaults to `False`.


        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithCrossAttentions` if `return_dict=True`.
            Otherwise it returns a tuple of tensors corresponding to ordered and
            not None (depending on the input arguments) fields of :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithCrossAttentions`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerForCausalLM, RoFormerTokenizer

                tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-sim-char-ft-base')
                model = RoFormerForCausalLM.from_pretrained('roformer-chinese-sim-char-ft-base')

                tokenized_inputs = tokenizer("欢迎使用百度飞桨!", return_tensors="pd")
                logits = model(**tokenized_inputs)
                print(logits.shape)
                # [1, 11, 12000]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            loss_fct = paddle.nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shifted_prediction_scores.reshape((-1, prediction_scores.shape[-1])), labels.reshape((-1,))
            )
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return tuple_output(output, lm_loss)

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, cache=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None:
            if "int" in convert_dtype(attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4

        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            if attention_mask.ndim == 4:
                attention_mask = attention_mask[:, -1, -1, :].unsqueeze([1, 2])

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "use_cache": use_cache,
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple):
            model_kwargs["cache"] = outputs[1]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            # token type id = 1
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, paddle.ones_like(token_type_ids[:, -1:])], axis=-1
            )

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == "bool":
                attention_mask = paddle.cast(attention_mask, "int64")
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask.expand((-1, -1, attention_mask.shape[-1], -1))
                attention_mask = nn.Pad2D([0, 0, 0, 1], mode="replicate")(attention_mask)
                attention_mask = nn.Pad2D([0, 1, 0, 0], value=-1e4)(attention_mask)
                dtype = convert_dtype(attention_mask.dtype)
                if "int" in dtype:
                    attention_mask[:, :, -1, -1] = 1
                elif "float" in dtype:
                    attention_mask[:, :, -1, -1] = 0.0
                else:
                    raise ValueError("The data type of input `attention_mask` must " "be bool, int or float")
            else:
                # convert to 4D attention_mask
                attention_mask = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="int64")], axis=-1
                )
                if "int" in convert_dtype(attention_mask.dtype):
                    attention_mask = (1.0 - attention_mask) * -1e4
                attention_mask = attention_mask.unsqueeze([1, 2]).expand((-1, -1, attention_mask.shape[-1], -1))

            token_type_ids = model_kwargs["token_type_ids"]
            mask = token_type_ids[:, None, :] > token_type_ids[:, :, None]
            # we need expand attention_mask
            attention_mask = paddle.where(mask.unsqueeze(1), paddle.to_tensor(-1e4), attention_mask)
            model_kwargs["attention_mask"] = attention_mask

        return model_kwargs
