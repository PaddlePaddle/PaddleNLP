# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 Huawei Technologies Co., Ltd.
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
from paddle import Tensor
import paddle
import paddle.nn as nn

from ..bert.modeling import BertPooler, BertEmbeddings
from .. import PretrainedModel, register_base_model
from ..configuration_utils import PretrainedConfig

from ..model_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    QuestionAnsweringModelOutput,
    MultipleChoiceModelOutput,
    tuple_output,
)

__all__ = [
    "TinyBertModel",
    "TinyBertPretrainedModel",
    "TinyBertForPretraining",
    "TinyBertForSequenceClassification",
    "TinyBertForQuestionAnswering",
    "TinyBertForMultipleChoice",
]


class TinyBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained TinyBERT models. It provides TinyBERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading
    and loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    pretrained_init_configuration = {
        "tinybert-4l-312d": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-v2": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-v2": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-zh": {
            "vocab_size": 21128,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-zh": {
            "vocab_size": 21128,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    pretrained_resource_files_map = {
        "model_state": {
            "tinybert-4l-312d": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d.pdparams",
            "tinybert-6l-768d": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d.pdparams",
            "tinybert-4l-312d-v2": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-v2.pdparams",
            "tinybert-6l-768d-v2": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-v2.pdparams",
            "tinybert-4l-312d-zh": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-4l-312d-zh.pdparams",
            "tinybert-6l-768d-zh": "http://bj.bcebos.com/paddlenlp/models/transformers/tinybert/tinybert-6l-768d-zh.pdparams",
        }
    }
    base_model_prefix = "tinybert"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range")
                        else self.tinybert.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class TinyBertModel(TinyBertPretrainedModel):
    """
    The bare TinyBERT Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `TinyBertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `TinyBertModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layers and pooler layer. Defaults to `768`.
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
            The maximum value of the dimensionality of position encoding. The dimensionality of position encoding
            is the dimensionality of the sequence in `TinyBertModel`.
            Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids` passed when calling `~ transformers.TinyBertModel`.
            Defaults to `16`.

        initializer_range (float, optional):
            The standard deviation of the normal initializer.
            Defaults to `0.02`.

            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`TinyBertPretrainedModel.init_weights()` for how weights are initialized in `TinyBertModel`.

        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        fit_size (int, optional):
            Dimensionality of the output layer of `fit_dense(s)`, which is the hidden size of the teacher model.
            `fit_dense(s)` means a hidden states' transformation from student to teacher.
            `fit_dense(s)` will be generated when bert model is distilled during the training, and will not be generated
            during the prediction process.
            `fit_denses` is used in v2 models and it has `num_hidden_layers+1` layers.
            `fit_dense` is used in other pretraining models and it has one linear layer.
            Defaults to `768`.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0,
        fit_size=768,
    ):
        super(TinyBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range

        # TODO(wj-Mcat): construct config temporary
        # to be removed when TinyBertConfig is completed
        config = PretrainedConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            # the default pool_act is `tanh`
            pool_act="tanh",
        )
        self.embeddings = BertEmbeddings(config)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)

        self.pooler = BertPooler(config)
        # fit_dense(s) means a hidden states' transformation from student to teacher.
        # `fit_denses` is used in v2 model, and `fit_dense` is used in other pretraining models.
        self.fit_denses = nn.LayerList([nn.Linear(hidden_size, fit_size) for i in range(num_hidden_layers + 1)])
        self.fit_dense = nn.Linear(hidden_size, fit_size)
        self.apply(self.init_weights)

    def get_input_embeddings(self) -> nn.Embedding:
        """get input embedding of TinyBert Pretrained Model

        Returns:
            nn.Embedding: the input embedding of tiny bert
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, embedding: nn.Embedding) -> None:
        """set the input embedding with the new embedding value

        Args:
            embedding (nn.Embedding): the new embedding value
        """
        self.embeddings.word_embeddings = embedding

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
        The TinyBertModel forward method, overrides the `__call__()` special method.

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

            tuple: Returns tuple (`encoder_output`, `pooled_output`).

            With the fields:

            - `encoder_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TinyBertModel, TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-4l-312d')
                model = TinyBertModel.from_pretrained('tinybert-4l-312d')

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
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
            )

            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = paddle.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = paddle.concat([past_mask, attention_mask], axis=-1)
        elif attention_mask.ndim == 2:
            # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
            attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4

        # TODO(wj-Mcat): in current branch, not support `inputs_embeds`
        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids, past_key_values_length=past_key_values_length
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


class TinyBertForPretraining(TinyBertPretrainedModel):
    """
    TinyBert Model with pretraining tasks on top.

    Args:
        tinybert (:class:`TinyBertModel`):
            An instance of :class:`TinyBertModel`.

    """

    def __init__(self, tinybert):
        super(TinyBertForPretraining, self).__init__()
        self.tinybert: TinyBertModel = tinybert
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The TinyBertForPretraining forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`TinyBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            position_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            attention_mask (Tensor, optional):
                See :class:`TinyBertModel`.

        Returns:
            Tensor: Returns tensor `sequence_output`, sequence of hidden-states at the last layer of the model.
            It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.tinybert.modeling import TinyBertForPretraining
                from paddlenlp.transformers.tinybert.tokenizer import TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-4l-312d')
                model = TinyBertForPretraining.from_pretrained('tinybert-4l-312d')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]


        """
        outputs = self.tinybert(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # return the sequence presentation
        if not return_dict:
            return outputs[0]
        return outputs


class TinyBertForSequenceClassification(TinyBertPretrainedModel):
    """
    TinyBert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        tinybert (:class:`TinyBertModel`):
            An instance of TinyBertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of TinyBert.
            If None, use the same value as `hidden_dropout_prob` of `TinyBertModel`
            instance `tinybert`. Defaults to `None`.
    """

    def __init__(self, tinybert, num_classes=2, dropout=None):
        super(TinyBertForSequenceClassification, self).__init__()
        self.tinybert = tinybert
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout if dropout is not None else self.tinybert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.tinybert.config["hidden_size"], num_classes)
        self.activation = nn.ReLU()
        self.apply(self.init_weights)

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
        The TinyBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`TinyBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            position_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            attention_mask_list (list, optional):
                See :class:`TinyBertModel`.
            labels (Tensor of shape `(batch_size,)`, optional):
                Labels for computing the sequence classification/regression loss.
                Indices should be in `[0, ..., num_classes - 1]`. If `num_classes == 1`
                a regression loss is computed (Mean-Square loss), If `num_classes > 1`
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
                from paddlenlp.transformers.tinybert.modeling import TinyBertForSequenceClassification
                from paddlenlp.transformers.tinybert.tokenizer import TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-4l-312d')
                model = TinyBertForSequenceClassification.from_pretrained('tinybert-4l-312d')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP! ")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """

        outputs = self.tinybert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classifier(self.activation(outputs[1]))

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                loss_fct = paddle.nn.MSELoss()
                loss = loss_fct(logits, labels)
            elif labels.dtype == paddle.int64 or labels.dtype == paddle.int32:
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, self.num_classes)), labels.reshape((-1,)))
            else:
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


class TinyBertForQuestionAnswering(TinyBertPretrainedModel):
    """
    TinyBert Model with a linear layer on top of the hidden-states
    output to compute `span_start_logits` and `span_end_logits`,
    designed for question-answering tasks like SQuAD.

    Args:
        tinybert (`TinyBertModel`):
            An instance of `TinyBertModel`.
    """

    def __init__(self, tinybert):
        super(TinyBertForQuestionAnswering, self).__init__()
        self.tinybert = tinybert  # allow tinybert to be config
        self.classifier = nn.Linear(self.tinybert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`TinyBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            position_ids (Tensor, optional):
                See :class:`TinyBertModel`.
            attention_mask (Tensor, optional):
                See :class:`TinyBertModel`.
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
                from paddlenlp.transformers import TinyBertForQuestionAnswering, TinyBertTokenizer

                tokenizer = TinyBertTokenizer.from_pretrained('tinybert-6l-768d-zh')
                model = TinyBertForQuestionAnswering.from_pretrained('tinybert-6l-768d-zh')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """

        outputs = self.tinybert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.classifier(outputs[0])
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
            output = (start_logits, end_logits) + outputs[2:]
            return tuple_output(output, total_loss)

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TinyBertForMultipleChoice(TinyBertPretrainedModel):
    """
    TinyBERT Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        tinybert (:class:`TinyBertModel`):
            An instance of TinyBertModel.
        num_choices (int, optional):
            The number of choices. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of Tinybert.
            If None, use the same value as `hidden_dropout_prob` of `TinyBertModel`
            instance `tinybert`. Defaults to None.
    """

    def __init__(self, tinybert, num_choices=2, dropout=None):
        super(TinyBertForMultipleChoice, self).__init__()
        self.num_choices = num_choices
        self.tinybert = tinybert
        self.dropout = nn.Dropout(dropout if dropout is not None else self.tinybert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.tinybert.config["hidden_size"], 1)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        The TinyBertForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`TinyBertModel` and shape as [batch_size, num_choice, sequence_length].
            token_type_ids(Tensor, optional):
                See :class:`TinyBertModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`TinyBertModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`TinyBertModel` and shape as [batch_size, num_choice, sequence_length].
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
            Tensor: Returns tensor `reshaped_logits`, a tensor of the multiple choice classification logits.
            Shape as `[batch_size, num_choice]` and dtype as `float32`.

        """
        # input_ids: [bs, num_choice, seq_l]
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids and inputs_embeds should not be None at the same time.")
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]])
        else:
            input_ids = input_ids.reshape(shape=(-1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(shape=(-1, token_type_ids.shape[-1]))

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1, position_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(shape=(-1, attention_mask.shape[-1]))

        outputs = self.tinybert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self.dropout(outputs[1])

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(shape=(-1, self.num_choices))  # logits: (bs, num_choice)

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
