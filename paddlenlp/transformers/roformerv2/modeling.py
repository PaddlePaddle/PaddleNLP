# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import tensor
from paddle.nn import Layer

from .. import PretrainedModel, register_base_model

__all__ = [
    "RoFormerv2Model",
    "RoFormerv2ForMaskedLM",
    "RoFormerv2PretrainedModel",
    "RoFormerv2ForSequenceClassification",
    "RoFormerv2ForTokenClassification",
    "RoFormerv2ForQuestionAnswering",
    "RoFormerv2ForMultipleChoice",
]


class Norm(Layer):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self._epsilon = epsilon

    def forward(self, x):
        variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
        return x / paddle.sqrt(variance + self._epsilon)


def initializer(tensor, num_hidden_layers=12, order=2, gain=1.0):
    """
    https://github.com/bojone/bert4keras/blob/5572ed481a14f5a62be7107e3846c88a5d6b617d/bert4keras/models.py#L1226-L1235
    """
    shape = paddle.shape(tensor)
    if shape[0] > 10000 or shape[0] < 10:
        hidden_size = shape[1]
    else:
        hidden_size = shape[0]
    gain *= num_hidden_layers ** (-1.0 / order)
    std = 1.13684723 / hidden_size**0.5 * gain

    return nn.initializer.TruncatedNormal(std=std)


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = attn_mask.dtype
        if attn_mask_dtype in [paddle.bool, paddle.int8, paddle.int16, paddle.int32, paddle.int64]:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e4
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class RotaryPositionEmbedding(Layer):
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


class MultiHeadAttentionWithRotary(Layer):
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
        super(MultiHeadAttentionWithRotary, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.need_weights = need_weights
        self.rotary_value = rotary_value

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rotary = RotaryPositionEmbedding(self.head_dim, max_position_embeddings)

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        q, k = self.rotary(q), self.rotary(k)
        if self.rotary_value:
            v = self.rotary(v)

        product = tensor.matmul(x=q, y=k, transpose_y=True) * self.scale
        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask

        weights = F.softmax(product)
        weights = self.dropout(weights)
        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


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
        self.norm1 = Norm()
        self.norm2 = Norm()
        self._config.update({"rotary_value": rotary_value, "max_position_embeddings": max_position_embeddings})


class RoFormerv2Embeddings(Layer):
    """
    Include embeddings from word and token_type embeddings
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        type_vocab_size=2,
    ):
        super(RoFormerv2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.norm = Norm()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids)

        input_embedings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + token_type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RoFormerv2PretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained RoFormerv2 models. It provides RoFormerv2 related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    pretrained_init_configuration = {
        "roformer_v2_chinese_char_small": {
            "vocab_size": 12000,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "pad_token_id": 0,
            "rotary_value": False,
            "use_bias": False,
        },
        "roformer_v2_chinese_char_base": {
            "vocab_size": 12000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "pad_token_id": 0,
            "rotary_value": False,
            "use_bias": False,
        },
        "roformer_v2_chinese_char_large": {
            "vocab_size": 12000,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "pad_token_id": 0,
            "rotary_value": False,
            "use_bias": False,
        },
    }

    pretrained_resource_files_map = {
        "model_state": {
            "roformer_v2_chinese_char_small": "https://bj.bcebos.com/paddlenlp/models/transformers/roformerv2/roformer_v2_chinese_char_small/model_state.pdparams",
            "roformer_v2_chinese_char_base": "https://bj.bcebos.com/paddlenlp/models/transformers/roformerv2/roformer_v2_chinese_char_base/model_state.pdparams",
            "roformer_v2_chinese_char_large": "https://bj.bcebos.com/paddlenlp/models/transformers/roformerv2/roformer_v2_chinese_char_large/model_state.pdparams",
        }
    }

    base_model_prefix = "roformerv2"

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                num_hidden_layers = (
                    self.num_hidden_layers
                    if hasattr(self, "num_hidden_layers")
                    else self.roformerv2.config["num_hidden_layers"]
                )
                initializer(layer.weight, num_hidden_layers, order=2, gain=1.0)
            if isinstance(layer, nn.Linear):
                use_bias = self.use_bias if hasattr(self, "use_bias") else self.roformerv2.config["use_bias"]
                if layer.bias is not None and not use_bias:
                    layer.bias = None
        elif isinstance(layer, Norm):
            layer._epsilon = 1e-12


@register_base_model
class RoFormerv2Model(RoFormerv2PretrainedModel):
    """
    The bare RoFormerv2 Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `RoFormerv2Model`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `RoFormerv2Model`.
        hidden_size (int, optional):
            Dimensionality of the, encoder layers and pooler layer. Defaults to `768`.
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
            are supported. Defaults to `"relu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`.
            Defaults to `2`.
        pad_token_id (int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        rotary_value (`bool`, optional):
            Whether or not apply rotay position embeddings to value.
            Defaults to `False`.
        use_bias (`bool`, optional):
            Whether or not use bias.
            Defaults to `False`.
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
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        rotary_value=False,
        use_bias=False,
    ):
        super(RoFormerv2Model, self).__init__()
        self.pad_token_id = pad_token_id
        self.num_hidden_layers = num_hidden_layers
        self.use_bias = use_bias
        self.embeddings = RoFormerv2Embeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            type_vocab_size,
        )
        encoder_layer = TransformerEncoderLayerWithRotary(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            rotary_value=rotary_value,
            max_position_embeddings=max_position_embeddings,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_hidden_states=False):
        r"""
        The RoFormerv2Model forward method, overrides the `__call__()` special method.

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
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `0.0` values and the others have `1.0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Currently, we only support 2D attention_mask.
                Defaults to `None`, which means `pad_token_id` will be ignored.
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns `sequence_output` or `encoder_outputs`.

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and its shape is [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerv2Model, RoFormerv2Tokenizer

                tokenizer = RoFormerv2Tokenizer.from_pretrained('roformer_v2_chinese_char_base')
                model = RoFormerv2Model.from_pretrained('roformer_v2_chinese_char_base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                output = model(**inputs)
        """

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(paddle.get_default_dtype()) * -1e4, axis=[1, 2]
            )
        else:
            if attention_mask.ndim == 2:
                # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
                attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(paddle.get_default_dtype())
                attention_mask = (1.0 - attention_mask) * -1e4
            else:
                raise ValueError("Currently we only support 2D attention_mask.")

        attention_mask.stop_gradient = True

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)

        outputs = encoder_outputs if output_hidden_states else sequence_output

        return outputs

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, embedding: nn.Embedding):
        self.embeddings.word_embeddings = embedding


class RoFormerv2ForQuestionAnswering(RoFormerv2PretrainedModel):
    """
    RoFormerv2 with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        roformerv2 (:class:`RoFormerv2Model`):
            An instance of RoFormerv2Model.
        dropout (float, optional):
            The dropout probability for output of RoFormerv2.
            If None, use the same value as `hidden_dropout_prob` of `RoFormerv2Model`
            instance `roformerv2`. Defaults to `None`.
    """

    def __init__(self, roformerv2, dropout=None):
        super(RoFormerv2ForQuestionAnswering, self).__init__()
        self.roformerv2 = roformerv2
        self.dropout = nn.Dropout(dropout if dropout is not None else self.roformerv2.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roformerv2.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        The RoFormerv2ForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerv2Model`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerv2Model`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerv2Model`.

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
                from paddlenlp.transformers import RoFormerv2ForQuestionAnswering, RoFormerv2Tokenizer

                tokenizer = RoFormerv2Tokenizer.from_pretrained('roformer_v2_chinese_char_base')
                model = RoFormerv2ForQuestionAnswering.from_pretrained('roformer_v2_chinese_char_base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits = outputs[1]
        """
        sequence_output = self.roformerv2(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        start_logits, end_logits = paddle.unstack(logits, axis=-1)

        return start_logits, end_logits


class RoFormerv2ForSequenceClassification(RoFormerv2PretrainedModel):
    """
    RoFormerv2 Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        roformerv2 (`RoFormerv2Model`):
            An instance of `paddlenlp.transformers.RoFormerv2Model`.
        num_classes (int, optional):
            The number of classes. Default to `2`.
        dropout (float, optional):
            The dropout probability for output of RoFormerv2.
            If None, use the same value as `hidden_dropout_prob`
            of `paddlenlp.transformers.RoFormerv2Model` instance. Defaults to `None`.
    """

    def __init__(self, roformerv2, num_classes=2, dropout=None):
        super(RoFormerv2ForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.roformerv2 = roformerv2
        self.dropout = nn.Dropout(dropout if dropout is not None else self.roformerv2.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roformerv2.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RoFormerv2Model`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerv2Model`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerv2Model`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerv2ForSequenceClassification, RoFormerv2Tokenizer

                tokenizer = RoFormerv2Tokenizer.from_pretrained('roformer_v2_chinese_char_base')
                model = RoFormerv2ForSequenceClassification.from_pretrained('roformer_v2_chinese_char_base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        sequence_output = self.roformerv2(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = sequence_output[:, 0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RoFormerv2ForTokenClassification(RoFormerv2PretrainedModel):
    """
    RoFormerv2 Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        roformerv2 (`RoFormerv2Model`):
            An instance of `paddlenlp.transformers.RoFormerv2Model`.
        num_classes (int, optional):
            The number of classes. Default to `2`.
        dropout (float, optional):
            The dropout probability for output of RoFormerv2.
            If None, use the same value as `hidden_dropout_prob`
            of `paddlenlp.transformers.RoFormerv2Model` instance. Defaults to `None`.
    """

    def __init__(self, roformerv2, num_classes=2, dropout=None):
        super(RoFormerv2ForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.roformerv2 = roformerv2  # allow roformerv2 to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.roformerv2.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roformerv2.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RoFormerv2Model`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerv2Model`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerv2Model`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerv2ForTokenClassification, RoFormerv2Tokenizer

                tokenizer = RoFormerv2Tokenizer.from_pretrained('roformer_v2_chinese_char_base')
                model = RoFormerv2ForTokenClassification.from_pretrained('roformer_v2_chinese_char_base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        sequence_output = self.roformerv2(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class RoFormerv2ForMultipleChoice(RoFormerv2PretrainedModel):
    """
    RoFormerv2 Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        roformerv2 (:class:`RoFormerv2Model`):
            An instance of RoFormerv2Model.
        num_choices (int, optional):
            The number of choices. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of RoFormerv2.
            If None, use the same value as `hidden_dropout_prob` of `RoFormerv2Model`
            instance `roformerv2`. Defaults to None.
    """

    def __init__(self, roformerv2, num_choices=2, dropout=None):
        super(RoFormerv2ForMultipleChoice, self).__init__()
        self.num_choices = num_choices
        self.roformerv2 = roformerv2
        self.dropout = nn.Dropout(dropout if dropout is not None else self.roformerv2.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roformerv2.config["hidden_size"], 1)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""
        The RoFormerv2ForMultipleChoice forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`RoFormerv2Model` and shape as [batch_size, num_choice, sequence_length].
            token_type_ids(Tensor, optional):
                See :class:`RoFormerv2Model` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`RoFormerv2Model` and shape as [batch_size, num_choice, sequence_length].

        Returns:
            Tensor: Returns tensor `reshaped_logits`, a tensor of the multiple choice classification logits.
            Shape as `[batch_size, num_choice]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerv2ForMultipleChoice, RoFormerv2Tokenizer
                from paddlenlp.data import Pad

                tokenizer = RoFormerv2Tokenizer.from_pretrained('roformer_v2_chinese_char_base')
                model = RoFormerv2ForMultipleChoice.from_pretrained('roformer_v2_chinese_char_base', num_choices=2)

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

                inputs = tokenizer(text, text_pair)
                input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id)(inputs["input_ids"])
                token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)(inputs["token_type_ids"])

                reshaped_logits = model(
                    input_ids=paddle.to_tensor(input_ids, dtype="int64"),
                    token_type_ids=paddle.to_tensor(token_type_ids, dtype="int64"),
                )
                print(reshaped_logits.shape)
                # [2, 2]

        """
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(-1, paddle.shape(input_ids)[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(shape=(-1, paddle.shape(token_type_ids)[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(shape=(-1, paddle.shape(attention_mask)[-1]))

        sequence_output = self.roformerv2(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        return reshaped_logits


class RoFormerv2LMPredictionHead(Layer):
    def __init__(self, hidden_size, vocab_size, embedding_weights=None, use_bias=False):
        super(RoFormerv2LMPredictionHead, self).__init__()
        self.use_bias = use_bias
        self.decoder_weight = (
            self.create_parameter(shape=[vocab_size, hidden_size], dtype=self.transform.weight.dtype)
            if embedding_weights is None
            else embedding_weights
        )
        if use_bias:
            self.decoder_bias = self.create_parameter(
                shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True
            )

    def forward(self, hidden_states):
        hidden_states = paddle.matmul(hidden_states, self.decoder_weight, transpose_y=True)
        if self.use_bias:
            hidden_states = hidden_states + self.decoder_bias

        return hidden_states


class RoFormerv2ForMaskedLM(RoFormerv2PretrainedModel):
    """
    RoFormerv2 Model with a `masked language modeling` head on top.

    Args:
        roformerv2 (:class:`RoFormerv2Model`):
            An instance of :class:`RoFormerv2Model`.

    """

    def __init__(self, roformerv2):
        super(RoFormerv2ForMaskedLM, self).__init__()
        self.roformerv2 = roformerv2
        self.cls = RoFormerv2LMPredictionHead(
            self.roformerv2.config["hidden_size"],
            self.roformerv2.config["vocab_size"],
            embedding_weights=self.roformerv2.embeddings.word_embeddings.weight,
            use_bias=self.roformerv2.config["use_bias"],
        )

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`RoFormerv2Model`.
            token_type_ids (Tensor, optional):
                See :class:`RoFormerv2Model`.
            attention_mask (Tensor, optional):
                See :class:`RoFormerv2Model`.

        Returns:
            Tensor: Returns tensor `prediction_scores`, The scores of masked token prediction.
            Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import RoFormerv2ForMaskedLM, RoFormerv2Tokenizer

                tokenizer = RoFormerv2Tokenizer.from_pretrained('roformer_v2_chinese_char_base')
                model = RoFormerv2ForMaskedLM.from_pretrained('roformer_v2_chinese_char_base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 11, 12000]

        """
        sequence_output = self.roformerv2(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        prediction_scores = self.cls(sequence_output)
        return prediction_scores
