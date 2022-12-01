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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import TransformerEncoder

from .. import PretrainedModel, register_base_model
from ..model_outputs import CausalLMOutputWithCrossAttentions

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

    pretrained_init_configuration = {
        "unimo-text-1.0": {
            "vocab_size": 18000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": False,
            "max_position_embeddings": 513,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "unk_token_id": 17963,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "mask_token_id": 3,
        },
        "unimo-text-1.0-lcsts-new": {
            "vocab_size": 18000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": False,
            "max_position_embeddings": 513,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "unk_token_id": 17963,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "mask_token_id": 3,
        },
        "unimo-text-1.0-summary": {
            "vocab_size": 18000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": False,
            "max_position_embeddings": 513,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "unk_token_id": 17963,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "mask_token_id": 3,
        },
        "unimo-text-1.0-large": {
            "vocab_size": 12800,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": False,
            "max_position_embeddings": 512,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "unk_token_id": 12088,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "mask_token_id": 3,
        },
        "unimo-text-1.0-dureader_qg": {
            "vocab_size": 18000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": False,
            "max_position_embeddings": 513,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "unk_token_id": 17963,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "mask_token_id": 3,
        },
        "unimo-text-1.0-question-generation": {
            "vocab_size": 18000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": False,
            "max_position_embeddings": 513,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "unk_token_id": 17963,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "mask_token_id": 3,
        },
        "unimo-text-1.0-question-generation-full_domain": {
            "vocab_size": 18000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": False,
            "max_position_embeddings": 513,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "unk_token_id": 17963,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "mask_token_id": 3,
        },
        "unimo-text-1.0-question-generation-dureader_qg": {
            "vocab_size": 18000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": False,
            "max_position_embeddings": 513,
            "type_vocab_size": 4,
            "initializer_range": 0.02,
            "unk_token_id": 17963,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 3,
            "mask_token_id": 3,
        },
    }
    pretrained_resource_files_map = {
        "model_state": {
            "unimo-text-1.0": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0.pdparams",
            "unimo-text-1.0-lcsts-new": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-lcsts-new.pdparams",
            "unimo-text-1.0-large": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-large.pdparams",
            "unimo-text-1.0-summary": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-summary.pdparams",
            "unimo-text-1.0-dureader_qg": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-dureader_qg.pdparams",
            "unimo-text-1.0-question-generation": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-question-generation.pdparams",
            "unimo-text-1.0-question-generation-v2": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-question-generation-full_domain.pdparams",
            "unimo-text-1.0-question-generation-dureader_qg": "https://bj.bcebos.com/paddlenlp/models/transformers/unimo/unimo-text-1.0-question-generation-dureader_qg.pdparams",
        }
    }
    base_model_prefix = "unimo"

    def init_weights(self, layer):
        # Initialization hook
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.unimo.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )


class UNIMOEmbeddings(nn.Layer):
    # Include embeddings from word, position and token_type.

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=4,
        pad_token_id=None,
    ):
        super(UNIMOEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        input_embedings = self.word_embeddings(input_ids)

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
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")
            token_type_ids.stop_gradient = True
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        return embeddings


@register_base_model
class UNIMOModel(UNIMOPretrainedModel):
    """
    The bare UNIMO Model outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the  superclass documentation for the generic methods.

    This model is also a `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass.
    Use it as a regular Paddle Layer and refer to the Paddle
    documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `UNIMOModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `UNIMOModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layers and encoder layers. Defaults to `768`.
        num_hidden_layers (int, optional):
            The number of hidden layers in the Transformer encoder. Defaults to `12`.
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
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob(float, optional):
            The dropout probability used in pre-process and post-precess of MHA
            and FFN sub-layer. Defaults to 0.1.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        normalize_before (bool, optional):
            Indicate whether to put layer normalization into preprocessing of
            MHA and FFN sub-layers. If True, pre-process is layer normalization
            and post-precess includes dropout, residual connection. Otherwise,
            no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Defaults to `True`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.UNIMOModel`.
            Defaults to `2`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`UNIMOPretrainedModel._init_weights()` for how weights are initialized in `UNIMOModel`.

        unk_token_id (int, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` in order to be converted to an ID.
            Defaults to `17963`.
        pad_token_id (int, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to `0`.
        bos_token_id (int, optional):
            A special token representing the beginning of a sequence that was used during pretraining.
            Defaults to `1`.
        eos_token_id (int, optional):
            A special token representing the end of a sequence that was used during pretraining.
            Defaults to `3`.
        mask_token_id (int, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to `3`.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        normalize_before=False,
        max_position_embeddings=513,
        type_vocab_size=4,
        initializer_range=0.02,
        unk_token_id=17963,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=3,
        mask_token_id=3,
    ):
        super(UNIMOModel, self).__init__()
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.initializer_range = initializer_range

        self.embeddings = UNIMOEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob, max_position_embeddings, type_vocab_size, self.pad_token_id
        )
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=normalize_before,
        )

        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_hidden_layers,
        )

        self.apply(self.init_weights)

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
        use_cache=False,
        cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The UNIMOModel forward method, overrides the special :meth:`__call__` method.

        Args:
            input_ids (Tensor):
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
        if attention_mask is None:
            attention_mask = ((input_ids == self.pad_token_id).astype(paddle.get_default_dtype()) * -1e4).unsqueeze(
                [1, 2]
            )
            attention_mask.stop_gradient = True

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)

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

    def forward(self, hidden_states, masked_positions=None):
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

    def __init__(self, unimo):
        super(UNIMOLMHeadModel, self).__init__()
        self.unimo = unimo
        self.lm_head = UNIMOLMHead(
            self.unimo.config["hidden_size"],
            self.unimo.config["vocab_size"],
            self.unimo.config["hidden_act"],
            self.unimo.embeddings.word_embeddings.weight,
        )
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
        use_cache=False,
        cache=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        r"""
        The UNIMOLMHeadModel forward method, overrides the special
        :meth:`__call__` method.

        Args:
            input_ids (Tensor):
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

        outputs = self.unimo(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            use_cache,
            cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs if isinstance(outputs, type(input_ids)) else outputs[0]

        logits = self.lm_head(sequence_output, masked_positions)

        lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(logits.reshape((-1, self.unimo.config["vocab_size"])), labels.reshape((-1,)))

        if not return_dict:
            if isinstance(outputs, type(input_ids)):
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

    def prepare_faster_entry(self, kwargs):
        from paddlenlp.ops import FasterUNIMOText

        use_fp16_decoding = kwargs.get("use_fp16_decoding", False)
        decode_strategy = kwargs.get("decode_strategy")
        if decode_strategy == "sampling" and kwargs.get("top_k") != 0 and kwargs.get("top_p") != 1:
            raise AttributeError(
                "Only topk sampling or topp sampling are supported. "
                "Topk sampling and topp sampling cannot be both applied in the faster version."
            )
        if kwargs["repetition_penalty"] != 1.0:
            # not support for repetition_penalty yet in the faster version
            raise AttributeError("'repetition_penalty != 1' is not supported yet in the faster version")
        if kwargs["forced_bos_token_id"] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError("'forced_bos_token_id != None' is not supported yet in the faster version")
        self._faster_entry = FasterUNIMOText(self, use_fp16_decoding=use_fp16_decoding).forward
        return self._faster_entry

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
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e


UNIMOForMaskedLM = UNIMOLMHeadModel
UNIMOForConditionalGeneration = UNIMOLMHeadModel
