# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 ShannonAI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import PretrainedModel, register_base_model

from .configuration import (
    CHINESEBERT_PRETRAINED_INIT_CONFIGURATION,
    CHINESEBERT_PRETRAINED_RESOURCE_FILES_MAP,
    ChineseBertConfig,
)

__all__ = [
    "ChineseBertModel",
    "ChineseBertPretrainedModel",
    "ChineseBertForPretraining",
    "ChineseBertPretrainingCriterion",
    "ChineseBertForSequenceClassification",
    "ChineseBertForTokenClassification",
    "ChineseBertForQuestionAnswering",
]


class PinyinEmbedding(nn.Layer):
    def __init__(self, config: ChineseBertConfig):
        """Pinyin Embedding Layer"""
        super(PinyinEmbedding, self).__init__()
        self.embedding = nn.Embedding(config.pinyin_map_len, config.pinyin_embedding_size)
        self.pinyin_out_dim = config.hidden_size
        self.conv = nn.Conv1D(
            in_channels=config.pinyin_embedding_size,
            out_channels=self.pinyin_out_dim,
            kernel_size=2,
            stride=1,
            padding=0,
        )

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids (Tensor): Its shape is (bs*sentence_length*pinyin_locs).

        Returns:
            pinyin_embed (Tensor): Its shape is (bs,sentence_length,pinyin_out_dim).

        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length*pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.reshape(
            shape=[-1, pinyin_locs, embed_size]
        )  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.transpose([0, 2, 1])  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.reshape(
            shape=[bs, sentence_length, self.pinyin_out_dim]
        )  # [bs,sentence_length,pinyin_out_dim]


class GlyphEmbedding(nn.Layer):
    """Glyph2Image Embedding."""

    def __init__(self, config: ChineseBertConfig):
        super(GlyphEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.glyph_embedding_dim)

    def forward(self, input_ids):
        """
        Get glyph images for batch inputs.

        Args:
            input_ids (Tensor): Its shape is [batch, sentence_length].

        Returns:
            images (Tensor): Its shape is [batch, sentence_length, self.font_num*self.font_size*self.font_size].

        """
        return self.embedding(input_ids)


class FusionBertEmbeddings(nn.Layer):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """

    def __init__(self, config: ChineseBertConfig):
        super(FusionBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(config)
        self.glyph_embeddings = GlyphEmbedding(config)

        self.glyph_map = nn.Linear(config.glyph_embedding_dim, config.hidden_size)
        self.map_fc = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            paddle.expand(paddle.arange(config.max_position_embeddings, dtype="int64"), shape=[1, -1]),
        )

    def forward(self, input_ids, pinyin_ids, token_type_ids=None, position_ids=None):

        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = self.word_embeddings(input_ids)  # [bs,l,hidden_size]

        pinyin_embeddings = self.pinyin_embeddings(
            pinyin_ids.reshape(shape=[input_shape[0], seq_length, 8])
        )  # [bs,l,hidden_size]

        glyph_embeddings = self.glyph_map(self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]
        # fusion layer
        concat_embeddings = paddle.concat((word_embeddings, pinyin_embeddings, glyph_embeddings), axis=2)
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Same as BertLMPredictionHead
class ChineseBertLMPredictionHead(nn.Layer):
    """
    Language Modeling head
    """

    def __init__(self, config: ChineseBertConfig, embedding_weights=None):
        super(ChineseBertLMPredictionHead, self).__init__()

        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = getattr(nn.functional, config.hidden_act)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.decoder_weight = (
            self.create_parameter(
                shape=[config.vocab_size, config.hidden_size], dtype=self.transform.weight.dtype, is_bias=False
            )
            if embedding_weights is None
            else embedding_weights
        )

        self.decoder_bias = self.create_parameter(
            shape=[config.vocab_size], dtype=self.decoder_weight.dtype, is_bias=True
        )

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states, [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states, masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(hidden_states, self.decoder_weight, transpose_y=True) + self.decoder_bias
        return hidden_states


# Same as BertPretrainingHeads
class ChineseBertPretrainingHeads(nn.Layer):
    """
    Perform language modeling task and next sentence classification task.

    Args:
        config (:class:`ChineseBertConfig`):
            An instance of ChineseBertConfig used to construct ChineseBertPretrainingHeads.
        embedding_weights (Tensor, optional):
            Decoding weights used to map hidden_states to logits of the masked token prediction.
            Its data type should be float32 and its shape is [vocab_size, hidden_size].
            Defaults to `None`, which means use the same weights of the embedding layer.

    """

    def __init__(self, config: ChineseBertConfig, embedding_weights=None):
        super(ChineseBertPretrainingHeads, self).__init__()
        self.predictions = ChineseBertLMPredictionHead(config, embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        """
        Args:
            sequence_output(Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
            pooled_output(Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
            masked_positions(Tensor, optional):
                A tensor indicates positions to be masked in the position embedding.
                Its data type should be int64 and its shape is [batch_size, mask_token_num].
                `mask_token_num` is the number of masked tokens. It should be no bigger than `sequence_length`.
                Defaults to `None`, which means we output hidden-states of all tokens in masked token prediction.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# Same as BertPooler
class ChineseBertPooler(nn.Layer):
    """
    Pool the result of ChineseBertEncoder.
    """

    def __init__(self, config):
        """init the bert pooler with config & args/kwargs

        Args:
            config (:class:`ChineseBertConfig`): An instance of ChineseBertConfig.
        """
        super(ChineseBertPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = config.pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class ChineseBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ChineseBert models. It provides ChineseBert related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    base_model_prefix = "chinesebert"
    pretrained_resource_files_map = CHINESEBERT_PRETRAINED_RESOURCE_FILES_MAP
    pretrained_init_configuration = CHINESEBERT_PRETRAINED_INIT_CONFIGURATION
    config_class = ChineseBertConfig

    def _init_weights(self, layer):
        """Initialize the weights."""

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
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = self.config.layer_norm_eps


@register_base_model
class ChineseBertModel(ChineseBertPretrainedModel):
    """
    The bare ChineseBert Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/zh/api/paddle/nn/Layer_cn.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        config (:class:`ChineseBertConfig`):
            An instance of ChineseBertConfig used to construct ChineseBertModel.

    """

    def __init__(self, config: ChineseBertConfig):
        super(ChineseBertModel, self).__init__(config)
        self.pad_token_id = config.pad_token_id
        self.layer_norm_eps = config.layer_norm_eps
        self.initializer_range = config.initializer_range
        self.embeddings = FusionBertEmbeddings(config)
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
        self.pooler = ChineseBertPooler(config)

    def forward(
        self,
        input_ids,
        pinyin_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        r"""
        The ChineseBert forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            pinyin_ids (Tensor, optional):
                Indices of input sequence tokens pinyin. We apply a CNN model with width 2 on the pinyin
                sequence, followed by max-pooling to derive the resulting pinyin embedding. This makes output
                dimensionality immune to the length of the input pinyin sequence. The length of the input pinyin
                sequence is fixed at 8.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length, 8].
                Defaults to `None`, which means we don't add pinyin embeddings.
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
            output_hidden_states (bool, optional):
                Whether to return the output of each hidden layers.
                Defaults to `False`.

        Returns:
            tuple: Returns tuple (`sequence_output`, `pooled_output`) or (`encoder_outputs`, `pooled_output`).

            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].

            - `encoder_outputs` (List(Tensor)):
                A list of Tensor containing hidden-states of the model at each hidden layer in the Transformer encoder.
                The length of the list is `num_hidden_layers`.
                Each Tensor has a data type of float32 and its shape is [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ChineseBertModel, ChineseBertTokenizer

                tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')
                model = ChineseBertModel.from_pretrained('ChineseBERT-base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)
        """

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(paddle.get_default_dtype()) * -1e4,
                axis=[1, 2],
            )
        elif attention_mask.ndim == 2:
            # attention_mask [batch_size, sequence_length] -> [batch_size, 1, 1, sequence_length]
            attention_mask = attention_mask.unsqueeze(axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4

        embedding_output = self.embeddings(
            input_ids=input_ids,
            pinyin_ids=pinyin_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        print(embedding_output.shape)

        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, src_mask=attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class ChineseBertForQuestionAnswering(ChineseBertPretrainedModel):
    """
    ChineseBert Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
    and `span_end_logits`, designed for question-answering tasks like SQuAD.

    Args:
        config (:class:`ChineseBertConfig`):
            An instance of ChineseBertConfig used to construct ChineseBertForQuestionAnswering.
    """

    def __init__(self, config: ChineseBertConfig):
        super(ChineseBertForQuestionAnswering, self).__init__(config)
        self.chinesebert = ChineseBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, pinyin_ids=None, token_type_ids=None, attention_mask=None):
        r"""
        The ChineseBertForQuestionAnswering forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ChineseBertModel`.
            pinyin_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            attention_mask (Tensor, optional):
                See :class:`ChineseBertModel`.

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
                from paddlenlp.transformers.chinesebert.modeling import ChineseBertForQuestionAnswering
                from paddlenlp.transformers.chinesebert.tokenizer import ChineseBertTokenizer

                tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')
                model = ChineseBertForQuestionAnswering.from_pretrained('ChineseBERT-base')

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                start_logits = outputs[0]
                end_logits = outputs[1]
        """
        sequence_output, _ = self.chinesebert(
            input_ids, pinyin_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=None
        )

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class ChineseBertForSequenceClassification(ChineseBertPretrainedModel):
    """
    ChineseBert Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        config (:class:`ChineseBertConfig`):
            An instance of ChineseBertConfig used to construct ChineseBertForSequenceClassification.e.
    """

    def __init__(self, config: ChineseBertConfig):
        super(ChineseBertForSequenceClassification, self).__init__(config)
        self.chinesebert = ChineseBertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, pinyin_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The ChineseBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ChineseBertModel`.
            pinyin_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            position_ids(Tensor, optional):
                See :class:`ChineseBertModel`.
            attention_mask (list, optional):
                See :class:`ChineseBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.chinesebert.modeling import ChineseBertForSequenceClassification
                from paddlenlp.transformers.chinesebert.tokenizer import ChineseBertTokenizer

                tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')
                model = ChineseBertForSequenceClassification.from_pretrained('ChineseBERT-base', num_classes=2)

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 2]

        """

        _, pooled_output = self.chinesebert(
            input_ids,
            pinyin_ids=pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ChineseBertForTokenClassification(ChineseBertPretrainedModel):
    """
    ChineseBert Model with a linear layer on top of the hidden-states output layer,
    designed for token classification tasks like NER tasks.

    Args:
        config (:class:`ChineseBertConfig`):
            An instance of ChineseBertConfig used to construct ChineseBertForTokenClassification.e.
    """

    def __init__(self, config: ChineseBertConfig):
        super(ChineseBertForTokenClassification, self).__init__(config)
        self.chinesebert = ChineseBertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, pinyin_ids=None, token_type_ids=None, position_ids=None, attention_mask=None):
        r"""
        The ChineseBertForTokenClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`ChineseBertModel`.
            pinyin_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            position_ids(Tensor, optional):
                See :class:`ChineseBertModel`.
            attention_mask (list, optional):
                See :class:`ChineseBertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
            Shape as `[batch_size, sequence_length, num_classes]` and dtype as `float32`.

        Example:
             .. code-block::

                import paddle
                from paddlenlp.transformers.chinesebert.modeling import ChineseBertForSequenceClassification
                from paddlenlp.transformers.chinesebert.tokenizer import ChineseBertTokenizer

                tokenizer = ChineseBertTokenizer.from_pretrained('ChineseBERT-base')
                model = ChineseBertForSequenceClassification.from_pretrained('ChineseBERT-base', num_classes=2)

                inputs = tokenizer("欢迎使用百度飞桨!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                logits = model(**inputs)
                print(logits.shape)
                # [1, 13, 2]

        """
        sequence_output, _ = self.chinesebert(
            input_ids,
            pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class ChineseBertForPretraining(ChineseBertPretrainedModel):
    """
    ChineseBert Model with pretraining tasks on top.

    Args:
        config (:class:`ChineseBertConfig`):
            An instance of ChineseBertConfig used to construct ChineseBertForPretraining.e.

    """

    def __init__(self, config: ChineseBertConfig):
        super(ChineseBertForPretraining, self).__init__(config)
        self.chinesebert = ChineseBertModel(config)
        self.cls = ChineseBertPretrainingHeads(
            config,
            embedding_weights=self.chinesebert.embeddings.word_embeddings.weight,
        )

    def forward(
        self,
        input_ids,
        pinyin_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
    ):
        r"""

        Args:
            input_ids (Tensor):
                See :class:`ChineseBertModel`.
            pinyin_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            token_type_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            position_ids (Tensor, optional):
                See :class:`ChineseBertModel`.
            attention_mask (Tensor, optional):
                See :class:`ChineseBertModel`.
            masked_positions(Tensor, optional):
                See :class:`ChineseBertPretrainingHeads`.

        Returns:
            tuple: Returns tuple (``prediction_scores``, ``seq_relationship_score``).

            With the fields:

            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size].

            - `seq_relationship_score` (Tensor):
                The scores of next sentence prediction.
                Its data type should be float32 and its shape is [batch_size, 2].

        """
        with paddle.static.amp.fp16_guard():
            outputs = self.chinesebert(
                input_ids,
                pinyin_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output, masked_positions)
            return prediction_scores, seq_relationship_score


class ChineseBertPretrainingCriterion(nn.Layer):
    """

    Args:
        vocab_size(int):
            Vocabulary size of `inputs_ids` in `ChineseBertModel`. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling `ChineseBertBertModel`.

    """

    def __init__(self, vocab_size):
        super(ChineseBertPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(
        self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels, masked_lm_scale
    ):
        """
        Args:
            prediction_scores(Tensor):
                The scores of masked token prediction. Its data type should be float32.
                If `masked_positions` is None, its shape is [batch_size, sequence_length, vocab_size].
                Otherwise, its shape is [batch_size, mask_token_num, vocab_size]
            seq_relationship_score(Tensor):
                The scores of next sentence prediction. Its data type should be float32 and
                its shape is [batch_size, 2]
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, its dimensionality is equal to `prediction_scores`.
                Its data type should be int64. If `masked_positions` is None, its shape is [batch_size, sequence_length, 1].
                Otherwise, its shape is [batch_size, mask_token_num, 1]
            next_sentence_labels(Tensor):
                The labels of the next sentence prediction task, the dimensionality of `next_sentence_labels`
                is equal to `seq_relation_labels`. Its data type should be int64 and
                its shape is [batch_size, 1]
            masked_lm_scale(Tensor or int):
                The scale of masked tokens. Used for the normalization of masked language modeling loss.
                If it is a `Tensor`, its data type should be int64 and its shape is equal to `prediction_scores`.

        Returns:
            Tensor: The pretraining loss, equals to the sum of `masked_lm_loss` plus the mean of `next_sentence_loss`.
            Its data type should be float32 and its shape is [1].


        """
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(prediction_scores, masked_lm_labels, reduction="none", ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(seq_relationship_score, next_sentence_labels, reduction="none")
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)
