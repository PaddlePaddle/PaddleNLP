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
"""Modeling classes for UNIMO model."""

from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from ...utils.env import CONFIG_NAME
from ...utils.log import logger
from .. import PretrainedModel, register_base_model
from ..model_outputs import CausalLMOutputWithCrossAttentions
from .configuration import (
    UNIMO_PRETRAINED_INIT_CONFIGURATION,
    UNIMO_PRETRAINED_RESOURCE_FILES_MAP,
    UNIMOConfig,
)

__all__ = [
    "UNIMOPretrainedModel",
    "UNIMOModel",
    "UNIMOLMHeadModel",
    "UNIMOForMaskedLM",
    "UNIMOForConditionalGeneration",
]


class UNIMOPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained UNIMO models. It provides UNIMO related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading
    and loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = CONFIG_NAME
    pretrained_init_configuration = UNIMO_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = UNIMO_PRETRAINED_RESOURCE_FILES_MAP
    base_model_prefix = "unimo"
    config_class = UNIMOConfig

    def _init_weights(self, layer):
        # Initialization hook
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


class UNIMOEmbeddings(nn.Layer):
    # Include embeddings from word, position and token_type.

    def __init__(self, config: UNIMOConfig):
        super(UNIMOEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pad_token_id = config.pad_token_id

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        input_embeddings: Optional[Tensor] = None,
    ):
        if input_ids is None and input_embeddings is None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            inputs_shape = paddle.shape(input_ids)
        elif input_embeddings is not None:
            inputs_shape = paddle.shape(input_embeddings)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_embeddings is None:
            input_embeddings = self.word_embeddings(input_ids)

        if position_ids is None:
            if self.pad_token_id is None:
                position_ids = paddle.expand_as(paddle.arange(end=inputs_shape[1], dtype="int64"), inputs_shape)
            else:
                if input_ids is not None:
                    num_pad = paddle.sum((input_ids == self.pad_token_id).astype("float32"), axis=-1, keepdim=True)
                    position_ids = F.relu(
                        paddle.expand_as(paddle.arange(end=inputs_shape[1], dtype="int64"), inputs_shape) - num_pad
                    ).astype("int64")
                else:
                    logger.warning(
                        "Position_ids or pad_token_ids should be provided when input_embeds is specified, "
                        "otherwise an unexpected result may be returned since `[0, 1, ..., sequence length - 1]` will be generated as a default position_ids."
                    )
                    position_ids = paddle.expand_as(paddle.arange(end=inputs_shape[1], dtype="int64"), inputs_shape)
            position_ids.stop_gradient = True
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")
            token_type_ids.stop_gradient = True
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        return embeddings


@register_base_model
class UNIMOModel(UNIMOPretrainedModel):
    """
    The bare UNIMO Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the  superclass documentation for the generic methods.

    This model is also a `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass.
    Use it as a regular Paddle Layer and refer to the Paddle
    documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`UNIMOConfig`):
            An instance of UNIMOConfig used to construct UNIMOModel.
    """

    def __init__(self, config: UNIMOConfig):
        super(UNIMOModel, self).__init__(config)
        self.unk_token_id = config.unk_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.mask_token_id = config.mask_token_id
        self.initializer_range = config.initializer_range

        self.embeddings = UNIMOEmbeddings(config)
        encoder_layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            attn_dropout=config.attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=config.normalize_before,
        )

        self.encoder_norm = nn.LayerNorm(config.hidden_size)
        # post_encoder_norm = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            config.num_hidden_layers,
            # post_encoder_norm,
        )

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[Tuple[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The UNIMOModel forward method, overrides the special :meth:`__call__` method.

        Args:
            input_ids (Tensor, optional):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of  [batch_size, sequence_length].
            token_type_ids (Tensor):
                Segment token indices to indicate first and second portions of
                the inputs. Indices can be either 0 or 1:

                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.

                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to None, which means no segment embeddings is added to token embeddings.
            position_ids (Tensor):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range ``[0, max_position_embeddings - 1]``.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.
            attention_mask (Tensor):
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
            use_cache: (bool, optional):
                Whether or not use the model cache to speed up decoding.
                Defaults to `False`.
            cache (list, optional):
                It is a list, and each element in the list is `incremental_cache`
                produced by :meth:`paddle.nn.TransformerEncoderLayer.gen_cache`
                method. See :meth:`paddle.nn.TransformerEncoder.gen_cache`
                method for more details. It is only used for inference and
                should be None for training. Defaults to `None`.
            inputs_embeds (Tensor, optional):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation
                of shape `(batch_size, sequence_length, hidden_size)`. This is useful if you want more control over
                how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
                Default to None.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail. Defaults to `False`.
            output_hidden_states (bool, optional):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail. Defaults to `False`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.BaseModelOutputWithPastAndCrossAttentions`.
            Especially, When `return_dict=output_hidden_states=output_attentions=False` and `cache=None`,
            returns tensor `Sequence_output` of shape [batch_size, sequence_length, hidden_size],
            which is the output at the last layer of the model.

        Example:
            .. code-block::

                from paddlenlp.transformers import UNIMOModel
                from paddlenlp.transformers import UNIMOTokenizer

                model = UNIMOModel.from_pretrained('unimo-text-1.0')
                tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0')

                inputs = tokenizer.gen_encode("Welcome to use PaddlePaddle and PaddleNLP!", return_tensors=True)
                outputs = model(**inputs)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = (
                    (input_ids == self.pad_token_id).astype(paddle.get_default_dtype()) * -1e4
                ).unsqueeze([1, 2])
            else:
                logger.warning(
                    "Provided inputs_embeds while attention_mask is None, attention weights will not be masked during forwarding."
                )

        if attention_mask is not None:
            attention_mask.stop_gradient = True

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, inputs_embeds)

        embedding_output = self.encoder_norm(embedding_output)
        embedding_output = self.dropout(embedding_output)

        if use_cache and cache is None:
            cache = self.encoder.gen_cache(embedding_output)

        outputs = self.encoder(
            embedding_output,
            attention_mask,
            cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class UNIMOLMHead(nn.Layer):
    def __init__(self, hidden_size, vocab_size, activation, embedding_weights=None):
        super(UNIMOLMHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = (
            self.create_parameter(shape=[vocab_size, hidden_size], dtype=self.transform.weight.dtype, is_bias=False)
            if embedding_weights is None
            else embedding_weights
        )
        self.decoder_bias = self.create_parameter(shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states: Tensor, masked_positions: Optional[Tensor] = None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states, [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states, masked_positions)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = paddle.tensor.matmul(hidden_states, self.decoder_weight, transpose_y=True) + self.decoder_bias
        return logits


class UNIMOLMHeadModel(UNIMOPretrainedModel):
    """
    The UNIMO Model with a `language modeling` head on top designed for generation tasks.

    Args:
        unimo (:class:`UNIMOModel`):
            An instance of :class:`UNIMOModel`.
    """

    def __init__(self, config: UNIMOConfig):
        super(UNIMOLMHeadModel, self).__init__(config)
        self.unimo = UNIMOModel(config)
        self.lm_head = UNIMOLMHead(
            config.hidden_size,
            config.vocab_size,
            config.hidden_act,
            self.unimo.embeddings.word_embeddings.weight,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        masked_positions: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        cache: Optional[Tuple[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The UNIMOLMHeadModel forward method, overrides the special
        :meth:`__call__` method.

        Args:
            input_ids (Tensor, optional):
                See :class:`UNIMOModel`.
            token_type_ids (Tensor):
                See :class:`UNIMOModel`.
            position_ids (Tensor):
                See :class:`UNIMOModel`.
            attention_mask (Tensor):
                See :class:`UNIMOModel`.
            use_cache: (bool, optional):
                See :class:`UNIMOModel`.
            cache (list, optional):
                See :class:`UNIMOModel`.
            inputs_embeds (Tensor, optional):
                See :class:`UNIMOModel`.
            labels (Tensor, optional):
                Labels for computing the left-to-right language modeling loss. Indices should be in
                `[-100, 0, ..., vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., vocab_size]`
            output_attentions (bool, optional):
                See :class:`UNIMOModel`.
            output_hidden_states (bool, optional):
                See :class:`UNIMOModel`.
            return_dict (bool, optional):
                Whether to return a :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithPastAndCrossAttentions` object. If `False`, the output
                will be a tuple of tensors. Defaults to `False`.

        Returns:
            An instance of :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithPastAndCrossAttentions` if
            `return_dict=True`. Otherwise it returns a tuple of tensors corresponding
            to ordered and not None (depending on the input arguments) fields of
            :class:`~paddlenlp.transformers.model_outputs.CausalLMOutputWithPastAndCrossAttentions`.
            Especially, When `return_dict=output_hidden_states=output_attentions=False` and `cache=labels=None`,
            returns tensor `logits` of shape [batch_size, sequence_length, hidden_size],
            which is the output at the last layer of the model.

        Example:
            .. code-block::

                from paddlenlp.transformers import UNIMOLMHeadModel
                from paddlenlp.transformers import UNIMOTokenizer

                model = UNIMOLMHeadModel.from_pretrained('unimo-text-1.0')
                tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0')

                inputs = tokenizer.gen_encode(
                    "Welcome to use PaddlePaddle and PaddleNLP!",
                    return_tensors=True,
                    is_split_into_words=False)
                logits = model(**inputs)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.unimo(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            use_cache,
            cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        input_type = type(input_ids) if input_ids is not None else type(inputs_embeds)
        sequence_output = outputs if isinstance(outputs, input_type) else outputs[0]

        logits = self.lm_head(sequence_output, masked_positions)

        lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(logits.reshape((-1, self.unimo.config.vocab_size)), labels.reshape((-1,)))

        if not return_dict:
            if isinstance(outputs, input_type):
                return (lm_loss, logits) if lm_loss is not None else logits
            else:
                outputs = (logits,) + outputs[1:]
                return ((lm_loss,) + outputs) if lm_loss is not None else outputs

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_fast_entry(self, kwargs):
        from paddlenlp.ops import FasterMIRO, FasterUNIMOText

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decode_strategy = kwargs.get("decode_strategy")
        if decode_strategy == "sampling" and kwargs.get("top_k") != 0 and kwargs.get("top_p") != 1:
            raise AttributeError(
                "Only topk sampling or topp sampling are supported. "
                "Topk sampling and topp sampling cannot be both applied in the fast version."
            )
        if kwargs["repetition_penalty"] != 1.0:
            # not support for repetition_penalty yet in the fast version
            raise AttributeError("'repetition_penalty != 1' is not supported yet in the fast version")
        if kwargs["forced_bos_token_id"] is not None:
            # not support for min_length yet in the fast version
            raise AttributeError(
                "Only topk sampling or topp sampling are supported. "
                "Topk sampling and topp sampling cannot be both applied in the fast version."
            )

        if getattr(self.encoder, "norm", None) is None:
            self._fast_entry = FasterUNIMOText(self, use_fp16_decoding=use_fp16_decoding).forward
        else:
            self._fast_entry = FasterMIRO(self, use_fp16_decoding=use_fp16_decoding).forward
        return self._fast_entry

    def adjust_logits_during_generation(self, logits):
        # pre-process distribution
        logits[:, self.unimo.unk_token_id] = -1e9
        logits[:, self.unimo.pad_token_id] = -1e9
        logits[:, self.unimo.bos_token_id] = -1e9
        return logits

    def prepare_inputs_for_generation(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        use_cache=False,
        cache=None,
        **kwargs
    ):

        if position_ids is None:
            if self.pad_token_id is None:
                position_ids = paddle.expand_as(
                    paddle.arange(end=paddle.shape(input_ids)[1], dtype="int64"), input_ids
                )
            else:
                num_pad = paddle.sum((input_ids == self.pad_token_id).astype("float32"), axis=-1, keepdim=True)
                position_ids = F.relu(
                    paddle.expand_as(paddle.arange(end=paddle.shape(input_ids)[1], dtype="float32"), input_ids)
                    - num_pad
                ).astype("int64")
            position_ids.stop_gradient = True

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")
            token_type_ids.stop_gradient = True

        if attention_mask is None:
            attention_mask = ((input_ids == self.pad_token_id).astype(paddle.get_default_dtype()) * -1e4).unsqueeze(
                [1, 2]
            )
            attention_mask.stop_gradient = True

        # only last token for inputs_ids if cache is defined in kwargs
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, :, -1:, :]

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(getattr(self, self.base_model_prefix), name)


UNIMOForMaskedLM = UNIMOLMHeadModel
UNIMOForConditionalGeneration = UNIMOLMHeadModel
