# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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

import itertools
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...utils.env import CONFIG_NAME
from .. import PretrainedModel, register_base_model
from ..activations import ACT2FN
from .configuration import (
    XLM_PRETRAINED_INIT_CONFIGURATION,
    XLM_PRETRAINED_RESOURCE_FILES_MAP,
    XLMConfig,
)

__all__ = [
    "XLMModel",
    "XLMPretrainedModel",
    "XLMWithLMHeadModel",
    "XLMForSequenceClassification",
    "XLMForTokenClassification",
    "XLMForQuestionAnsweringSimple",
    "XLMForMultipleChoice",
]

INF = 1e4


class SinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        n_pos, dim = paddle.shape(out)
        out.stop_gradient = True
        position_ids = paddle.arange(0, n_pos, dtype=out.dtype).unsqueeze(1)
        indices = paddle.arange(0, dim // 2, dtype=out.dtype).unsqueeze(0)
        indices = 10000.0 ** (-2 * indices / dim)
        embeddings = paddle.matmul(position_ids, indices)
        out[:, 0::2] = paddle.sin(embeddings)
        out[:, 1::2] = paddle.cos(embeddings)
        return out

    @paddle.no_grad()
    def forward(self, position_ids):
        return super().forward(position_ids)


def get_masks(seqlen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = paddle.arange(0, seqlen, dtype="int64")
    if padding_mask is not None:
        mask = padding_mask
    else:
        mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    bs = paddle.shape(lengths)[0]
    if causal:
        attn_mask = paddle.tile(alen[None, None, :], (bs, seqlen, 1)) <= alen[None, :, None]
    else:
        attn_mask = mask

    return mask, attn_mask


class MultiHeadAttention(nn.Layer):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config: XLMConfig):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        assert self.dim % self.n_heads == 0
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dim_per_head = self.dim // self.n_heads

    def shape(self, x):
        """projection"""
        return x.reshape([0, 0, self.n_heads, self.dim_per_head]).transpose([0, 2, 1, 3])

    def unshape(self, x):
        """compute context"""
        return x.transpose([0, 2, 1, 3]).reshape([0, 0, self.n_heads * self.dim_per_head])

    def forward(self, input, mask, kv=None, cache=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = paddle.shape(input)
        if kv is None:
            klen = qlen if cache is None else cache["seqlen"] + qlen
        else:
            klen = paddle.shape(kv)[1]

        mask_reshape = (bs, 1, qlen, klen) if mask.ndim == 3 else (bs, 1, 1, klen)

        q = self.shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = self.shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = self.shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = self.shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = self.shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = paddle.concat([k_, k], axis=2)  # (bs, n_heads, klen, dim_per_head)
                    v = paddle.concat([v_, v], axis=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(self.dim_per_head)  # (bs, n_heads, qlen, dim_per_head)

        scores = paddle.matmul(q, k, transpose_y=True)  # (bs, n_heads, qlen, klen)

        mask = mask.reshape(mask_reshape)  # (bs, n_heads, qlen, klen)

        scores = scores + (mask.astype(scores.dtype) - 1) * INF

        weights = F.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        weights = self.dropout(weights)  # (bs, n_heads, qlen, klen)

        context = paddle.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = self.unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs


class TransformerFFN(nn.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config: XLMConfig):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class XLMPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained XLM models. It provides XLM related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    pretrained_init_configuration = XLM_PRETRAINED_INIT_CONFIGURATION
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = XLM_PRETRAINED_RESOURCE_FILES_MAP
    model_config_file = CONFIG_NAME
    config_class = XLMConfig
    base_model_prefix = "xlm"

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.Embedding):
            new_weight = paddle.normal(
                mean=0.0,
                std=self.embed_init_std if hasattr(self, "embed_init_std") else self.xlm.config["embed_init_std"],
                shape=layer.weight.shape,
            )
            if layer._padding_idx is not None:
                new_weight[layer._padding_idx] = paddle.zeros_like(new_weight[layer._padding_idx])
            layer.weight.set_value(new_weight)
        elif isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.init_std if hasattr(self, "init_std") else self.xlm.config["init_std"],
                    shape=layer.weight.shape,
                )
            )
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))


@register_base_model
class XLMModel(XLMPretrainedModel):
    """
    The bare XLM Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`XLMConfig`):
            An instance of :class:`XLMConfig`.
    """

    def __init__(self, config: XLMConfig):
        super().__init__(config)
        self.causal = config.causal
        self.num_hidden_layers = config.num_hidden_layers
        self.pad_token_id = config.pad_token_id
        self.hidden_size = config.hidden_size
        self.embed_init_std = config.embed_init_std
        self.init_std = config.init_std
        self.use_lang_embeddings = config.use_lang_embeddings
        self.n_langs = config.n_langs
        if not config.is_encoder:
            raise NotImplementedError("Currently XLM can only be used as an encoder")
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), "xlm model's hidden_size must be a multiple of num_attention_heads"

        # embeddings
        if config.use_sinusoidal_embeddings:
            self.position_embeddings = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.hidden_size
            )
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.n_langs > 1 and config.use_lang_embeddings:
            self.lang_embeddings = nn.Embedding(config.n_langs, config.hidden_size)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer_norm_emb = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        self.attentions = nn.LayerList()
        self.layer_norm1 = nn.LayerList()
        self.ffns = nn.LayerList()
        self.layer_norm2 = nn.LayerList()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        for _ in range(self.num_hidden_layers):
            self.attentions.append(MultiHeadAttention(config.num_attention_heads, config.hidden_size, config))
            self.layer_norm1.append(nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps))

            self.ffns.append(
                TransformerFFN(
                    config.hidden_size,
                    config.hidden_size * 4,
                    config.hidden_size,
                    config,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps))

        self.register_buffer(
            "position_ids",
            paddle.arange(0, config.max_position_embeddings).reshape((1, -1)),
            persistable=False,
        )

    def forward(
        self,
        input_ids=None,
        langs=None,
        attention_mask=None,
        position_ids=None,
        lengths=None,
        cache=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        r"""
        The XLMModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            langs (Tensor, optional):
                A parallel sequence of tokens to be used to indicate the language of each token in the input. Indices are
                languages ids which can be obtained from the language names by using two conversion mappings provided in
                the configuration of the model (only provided for multilingual models). More precisely, the *language name
                to language id* mapping is in `model.config['lang2id']` (which is a dictionary string to int).
                Shape as [batch_size, sequence_length] and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some
                unwanted positions, usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others
                have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `0.0` values and the others have `1.0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                Defaults to `None`, which means nothing needed to be prevented attention to.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected
                in the range `[0, max_position_embeddings - 1]`.
                Shape as [batch_size, sequence_length] and dtype as int64. Defaults to `None`.
            lengths (Tensor, optional):
                Length of each sentence that can be used to avoid performing attention on padding token indices. You can
                also use *attention_mask* for the same result (see above), kept here for compatibility. Indices selected in
                `[0, ..., sequence_length]`.
                Shape as [batch_size] and dtype as int64. Defaults to `None`.
            cache (Tuple[Tuple[Tensor]], optional):
                Contains pre-computed hidden-states (key and values in the attention blocks)
                as computed by the model. Can be used to speed up sequential decoding.
                The `input_ids` which have their past given to this model should not be
                passed as input ids as they have already been computed.
                Defaults to `None`.
            output_attentions (bool, optional):
                Whether or not to return the attentions tensors of all attention layers.
                Defaults to `False`.
            output_hidden_states (bool, optional):
                Whether or not to return the output of all hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (`last_hidden_state`, `hidden_states`, `attentions`)

            With the fields:

            - `last_hidden_state` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `hidden_states` (tuple(Tensor), optional):
                returned when `output_hidden_states=True` is passed.
                Tuple of `Tensor` (one for the output of the embeddings + one for the output of
                each layer). Each Tensor has a data type of float32 and its shape is
                [batch_size, sequence_length, hidden_size].

            - `attentions` (tuple(Tensor), optional):
                returned when `output_attentions=True` is passed.
                Tuple of `Tensor` (one for each layer) of shape. Each Tensor has a data type of
                float32 and its shape is [batch_size, num_heads, sequence_length, sequence_length].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import XLMModel, XLMTokenizer

                tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
                model = XLMModel.from_pretrained("xlm-mlm-tlm-xnli15-1024")

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", lang="en")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                inputs["langs"] = paddle.ones_like(inputs["input_ids"]) * tokenizer.lang2id["en"]

                last_hidden_state = model(**inputs)[0]

        """
        bs, seqlen = paddle.shape(input_ids)

        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_token_id).sum(axis=1).astype("int64")
            else:
                lengths = paddle.to_tensor([seqlen] * bs, dtype="int64")

        # generate masks
        mask, attn_mask = get_masks(seqlen, lengths, self.causal, padding_mask=attention_mask)

        # position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seqlen]

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _seqlen = seqlen - cache["seqlen"]
            input_ids = input_ids[:, -_seqlen:]
            position_ids = position_ids[:, -_seqlen:]
            if langs is not None:
                langs = langs[:, -_seqlen:]
            mask = mask[:, -_seqlen:]
            attn_mask = attn_mask[:, -_seqlen:]

        # embeddings
        tensor = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        if langs is not None and self.use_lang_embeddings and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)

        tensor = self.layer_norm_emb(tensor)
        tensor = self.dropout(tensor)
        tensor = tensor * mask.unsqueeze(-1).astype(tensor.dtype)

        # transformer layers
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for i in range(self.num_hidden_layers):
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)
            # self attention
            attn_outputs = self.attentions[i](
                tensor,
                attn_mask,
                cache=cache,
                output_attentions=output_attentions,
            )
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = self.dropout(attn)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor = tensor * mask.unsqueeze(-1).astype(tensor.dtype)

        # Add last hidden state
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache["seqlen"] += paddle.shape(tensor)[1]

        return tuple(v for v in [tensor, hidden_states, attentions] if v is not None)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value


class XLMPredLayer(nn.Layer):
    """
    Prediction layer with cross_entropy.
    """

    def __init__(
        self,
        config: XLMConfig,
        embedding_weights=None,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        if embedding_weights is None:
            self.proj = nn.Linear(config.hidden_size, config.vocab_size)
        else:
            self.bias = self.create_parameter(shape=[config.vocab_size], is_bias=True)
            self.proj = lambda x: paddle.matmul(x, embedding_weights, transpose_y=True) + self.bias

    def forward(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        outputs = ()
        scores = self.proj(x)
        outputs = (scores,) + outputs
        if y is not None:
            loss = F.cross_entropy(scores.reshape([-1, self.vocab_size]), y.flatten(), reduction="mean")
            outputs = (loss,) + outputs
        return outputs


class XLMWithLMHeadModel(XLMPretrainedModel):
    """
    The XLM Model transformer with a masked language modeling head on top (linear
    layer with weights tied to the input embeddings).

    Args:
        config (:class:`XLMConfig`):
            An instance of :class:`XLMConfig`.

    """

    def __init__(self, config: XLMConfig):
        super().__init__(config)
        self.xlm = XLMModel(config)
        self.pred_layer = XLMPredLayer(
            config,
            embedding_weights=self.xlm.embeddings.weight,
        )

    def forward(
        self, input_ids=None, langs=None, attention_mask=None, position_ids=None, lengths=None, cache=None, labels=None
    ):
        r"""
        The XLMWithLMHeadModel forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`XLMModel`.
            langs (Tensor, optional):
                See :class:`XLMModel`.
            attention_mask (Tensor, optional):
                See :class:`XLMModel`.
            position_ids (Tensor, optional):
                See :class:`XLMModel`.
            lengths (Tensor, optional):
                See :class:`XLMModel`.
            cache (Dict[str, Tensor], optional):
                See :class:`XLMModel`.
            labels (Tensor, optional):
                The Labels for computing the masked language modeling loss. Indices are selected in
                `[-100, 0, ..., vocab_size-1]` All labels set to `-100` are ignored (masked), the loss is
                only computed for labels in `[0, ..., vocab_size-1]`
                Shape as [batch_size, sequence_length] and dtype as int64. Defaults to `None`.

        Returns:
            tuple: Returns tuple `(loss, logits)`.
            With the fields:

            - `loss` (Tensor):
                returned when `labels` is provided.
                Language modeling loss (for next-token prediction).
                It's data type should be float32 and its shape is [1,].

            - `logits` (Tensor):
                Prediction scores of the language modeling head (scores for each vocabulary
                token before SoftMax).
                It's data type should be float32 and
                its shape is [batch_size, sequence_length, vocab_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import XLMWithLMHeadModel, XLMTokenizer

                tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-tlm-xnli15-1024')
                model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-tlm-xnli15-1024')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", lang="en")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                inputs["langs"] = paddle.ones_like(inputs["input_ids"]) * tokenizer.lang2id["en"]
                inputs["labels"] = inputs["input_ids"]

                loss, logits = model(**inputs)


        """
        xlm_outputs = self.xlm(
            input_ids,
            langs=langs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
        )

        output = xlm_outputs[0]
        outputs = self.pred_layer(output, labels)
        return outputs + xlm_outputs[1:]


class XLMForSequenceClassification(XLMPretrainedModel):
    """
    The XLMModel with a sequence classification head on top (linear layer).
    `XLMForSequenceClassification` uses the first token in order to do the classification.

    Args:
        config (:class:`XLMConfig`):
            An instance of :class:`XLMConfig`.

    """

    def __init__(self, config: XLMConfig):
        super().__init__(config)
        self.num_classes = config.num_classes
        self.xlm = XLMModel(config)
        dropout_prob = config.dropout if config.dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids=None, langs=None, attention_mask=None, position_ids=None, lengths=None):
        r"""
        The XLMForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`XLMModel`.
            langs (Tensor, optional):
                See :class:`XLMModel`.
            attention_mask (Tensor, optional):
                See :class:`XLMModel`.
            position_ids (Tensor, optional):
                See :class:`XLMModel`.
            lengths (Tensor, optional):
                See :class:`XLMModel`.

        Returns:
            logits (Tensor):
                A tensor of the input text classification logits.
                Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import XLMForSequenceClassification, XLMTokenizer

                tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
                model = XLMForSequenceClassification.from_pretrained("xlm-mlm-tlm-xnli15-1024", num_classes=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", lang="en")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                inputs["langs"] = paddle.ones_like(inputs["input_ids"]) * tokenizer.lang2id["en"]

                logits = model(**inputs)

        """

        sequence_output = self.xlm(
            input_ids, langs=langs, attention_mask=attention_mask, position_ids=position_ids, lengths=lengths
        )[0]
        sequence_output = self.dropout(sequence_output)
        pooled_output = sequence_output[:, 0]
        logits = self.classifier(pooled_output)

        return logits


class XLMForTokenClassification(XLMPretrainedModel):
    """
    XLMModel with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`XLMConfig`):
            An instance of :class:`XLMConfig`.
    """

    def __init__(self, config: XLMConfig):
        super(XLMForTokenClassification, self).__init__(config)
        self.num_classes = config.num_classes
        self.xlm = XLMModel(config)  # allow xlm to be config
        self.dropout = nn.Dropout(config.dropout if config.dropout is not None else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids=None, langs=None, attention_mask=None, position_ids=None, lengths=None):
        r"""
        The XLMForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`XLMModel`.
            langs (Tensor, optional):
                See :class:`XLMModel`.
            attention_mask (Tensor, optional):
                See :class:`XLMModel`.
            position_ids (Tensor, optional):
                See :class:`XLMModel`.
            lengths (Tensor, optional):
                See :class:`XLMModel`.

        Returns:
            logits (Tensor):
                A tensor of the input token classification logits.
                Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import XLMForTokenClassification, XLMTokenizer

                tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
                model = XLMForTokenClassification.from_pretrained("xlm-mlm-tlm-xnli15-1024", num_classes=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", lang="en")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                inputs["langs"] = paddle.ones_like(inputs["input_ids"]) * tokenizer.lang2id["en"]

                logits = model(**inputs)

        """

        sequence_output = self.xlm(
            input_ids, langs=langs, attention_mask=attention_mask, position_ids=position_ids, lengths=lengths
        )[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


class XLMForQuestionAnsweringSimple(XLMPretrainedModel):
    """
    XLMModel with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

    Args:
        config (:class:`XLMConfig`):
            An instance of :class:`XLMConfig`.
    """

    def __init__(self, config: XLMConfig):
        super(XLMForQuestionAnsweringSimple, self).__init__(config)
        self.xlm = XLMModel(config)  # allow xlm to be config
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids=None, langs=None, attention_mask=None, position_ids=None, lengths=None):
        r"""
        The XLMForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`XLMModel`.
            langs (Tensor, optional):
                See :class:`XLMModel`.
            attention_mask (Tensor, optional):
                See :class:`XLMModel`.
            position_ids (Tensor, optional):
                See :class:`XLMModel`.
            lengths (Tensor, optional):
                See :class:`XLMModel`.

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
                from paddlenlp.transformers import XLMForQuestionAnswering, XLMTokenizer

                tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
                model = XLMForQuestionAnswering.from_pretrained("xlm-mlm-tlm-xnli15-1024", num_classes=2)

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!", lang="en")
                inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
                inputs["langs"] = paddle.ones_like(inputs["input_ids"]) * tokenizer.lang2id["en"]

                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits = outputs[1]

        """

        sequence_output = self.xlm(
            input_ids, langs=langs, attention_mask=attention_mask, position_ids=position_ids, lengths=lengths
        )[0]
        logits = self.classifier(sequence_output)
        start_logits, end_logits = paddle.unstack(x=logits, axis=-1)

        return start_logits, end_logits


class XLMForMultipleChoice(XLMPretrainedModel):
    """
    XLMModel with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.

    Args:
        config (:class:`XLMConfig`):
            An instance of :class:`XLMConfig`.
    """

    def __init__(self, config: XLMConfig):
        super(XLMForMultipleChoice, self).__init__(config)
        # self.num_choices = num_choices
        self.xlm = XLMModel(config)
        self.dropout = nn.Dropout(config.dropout if config.dropout is not None else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, langs=None, attention_mask=None, position_ids=None, lengths=None):
        r"""
        The XLMForMultipleChoice forward method, overrides the __call__() special method.
        Args:
            input_ids (Tensor):
                See :class:`XLMModel` and shape as [batch_size, num_choice, sequence_length].
            langs(Tensor, optional):
                See :class:`XLMModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (Tensor, optional):
                See :class:`XLMModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids (Tensor, optional):
                See :class:`XLMModel` and shape as [batch_size, num_choice, sequence_length].
            lengths (Tensor, optional):
                See :class:`XLMModel` and shape as [batch_size, num_choice].

        Returns:
            reshaped_logits (Tensor):
                A tensor of the multiple choice classification logits.
                Shape as `[batch_size, num_choice]` and dtype as `float32`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import XLMForMultipleChoice, XLMTokenizer
                from paddlenlp.data import Pad

                tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
                model = XLMForMultipleChoice.from_pretrained("xlm-mlm-tlm-xnli15-1024", num_choices=2)

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

                inputs = tokenizer(text, text_pair, lang="en")
                input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id)(inputs["input_ids"])
                input_ids = paddle.to_tensor(input_ids, dtype="int64")
                langs = paddle.ones_like(input_ids) * tokenizer.lang2id["en"]

                reshaped_logits = model(
                    input_ids=input_ids,
                    langs=langs,
                )
        """
        num_choices = input_ids.shape[1]
        # input_ids: [bs, num_choice, seqlen]
        input_ids = input_ids.reshape(
            shape=(-1, paddle.shape(input_ids)[-1])
        )  # flat_input_ids: [bs*num_choice, seqlen]

        if langs is not None:
            langs = langs.reshape(shape=(-1, paddle.shape(langs)[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(shape=(-1, paddle.shape(attention_mask)[-1]))

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1, paddle.shape(position_ids)[-1]))

        if lengths is not None:
            lengths = lengths.reshape(shape=(-1,))

        sequence_output = self.xlm(
            input_ids, langs=langs, attention_mask=attention_mask, position_ids=position_ids, lengths=lengths
        )[0]
        sequence_output = self.dropout(sequence_output)
        pooled_output = sequence_output[:, 0]

        logits = self.classifier(pooled_output)  # logits: [bs*num_choice, 1]
        reshaped_logits = logits.reshape(shape=(-1, num_choices))  # logits: [bs, num_choice]

        return reshaped_logits
