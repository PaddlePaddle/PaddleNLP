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

import paddle
import paddle.nn as nn

from paddlenlp.layers.crf import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.utils.log import logger

from .. import PretrainedModel, register_base_model

__all__ = [
    'SkepModel', 'SkepPretrainedModel', 'SkepForSequenceClassification',
    'SkepForTokenClassification', 'SkepCrfForTokenClassification'
]


class SkepEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 pad_token_id=0):
        super(SkepEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.type_vocab_size = type_vocab_size
        if self.type_vocab_size != 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                      hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings
        if self.type_vocab_size != 0:
            if token_type_ids is None:
                token_type_ids = paddle.zeros_like(input_ids, dtype="int64")
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        elif token_type_ids is not None:
            logger.warning(
                "There is no need to pass the token type ids to SKEP based on RoBERTa model."
                "The input token type ids will be ignored.")

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SkepPooler(nn.Layer):
    """
    The pooling layer on skep model.
    """

    def __init__(self, hidden_size):
        super(SkepPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SkepPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained Skep models. It provides SKEP related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "skep_ernie_1.0_large_ch": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,  # special for ernie-large
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 12800,
            "pad_token_id": 0,
        },
        "skep_ernie_2.0_large_en": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,  # special for ernie-large
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "skep_roberta_large_en": {
            "attention_probs_dropout_prob": 0.1,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "max_position_embeddings": 514,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 0,
            "vocab_size": 50265,
            "pad_token_id": 1,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "skep_ernie_1.0_large_ch":
            "https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_1.0_large_ch.pdparams",
            "skep_ernie_2.0_large_en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_2.0_large_en.pdparams",
            "skep_roberta_large_en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_roberta_large_en.pdparams",
        }
    }
    base_model_prefix = "skep"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.skep.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5


@register_base_model
class SkepModel(SkepPretrainedModel):
    r"""
    The bare SKEP Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Check the superclass documentation for the generic methods and the library implements for all its model.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    More details refer to `SKEP <https://www.aclweb.org/anthology/2020.acl-main.374>`.

    Args:

        vocab_size (`int`):
            Vocabulary size of the SKEP model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling SKEP.
        hidden_size (`int`, optional):
            Dimension of the encoder layers and the pooler layer. Defaults to ``768``.
        num_hidden_layers (`int`, optional):
            Number of hidden layers in the Transformer encoder. Defaults to ``12``.
        num_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to ``12``.
        intermediate_size (`int`, optional):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to ``3072``.
        hidden_act (`str`, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to ``0.1``.
        max_position_embeddings (`int`, optional):
            The max position index of an input sequence. Defaults to ``512``.
        type_vocab_size (`int`, optional):
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.ErnieModel`.
            Defaults to 2
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
        pad_token_id(`int`, optional):
            The pad token index in the token vocabulary.

    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(SkepModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = SkepEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = SkepPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (`Tensor`):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using :class:`~transformers.BertTokenizer`. See
                :meth:`paddlenlp.transformers.PreTrainedTokenizer.tokenize` and :meth:`transformers.PreTrainedTokenizer.__call__` for
                details.
            token_type_ids (`Tensor`, optional):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
                1]``:
                - 0 corresponds to a `sentence A` token,
                - 1 corresponds to a `sentence B` token.
                Defaults to `None`.
            position_ids (`Tensor`, `optional`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`.
            attention_mask (`Tensor`, optional):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                Defaults to `None`.

        Returns:
            A tuple of shape (``sequence_output``, ``pooled_output``).

            With the fields:
            - sequence_output (`Tensor`):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and has a shape of (batch_size, seq_lens, hidden_size].
                ``seq_lens`` corresponds to the length of input sequence.
            - pooled_output (`Tensor`):
                A Tensor of the first token representation.
                We "pool" the model by simply taking the hidden state corresponding to the first token.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import SkepModel, SkepTokenizer

                tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')
                model = SkepModel.from_pretrained('skep_ernie_1.0_large_ch')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor(v) for (k, v) in inputs.items()}
                sequence_output, pooled_output = model(**inputs)

        """
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class SkepForSequenceClassification(SkepPretrainedModel):
    r"""
    Model for sentence (pair) classification task with SKEP.

    Args:
        skep (`SkepModel`): 
            An instance of `SkepModel`.
        num_classes (`int`, optional): 
            The number of classes. Default to `2`.
        dropout (`float`, optional): 
            The dropout probability for output of SKEP. 
            If None, use the same value as `hidden_dropout_prob` 
            of `SkepModel` instance `Ernie`. Defaults to `None`.
    """

    def __init__(self, skep, num_classes=2, dropout=None):
        super(SkepForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.skep = skep  # allow skep to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.skep.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.skep.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (`Tensor`):
                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using :class:`~transformers.BertTokenizer`. See
                :meth:`paddlenlp.transformers.PreTrainedTokenizer.tokenize` and :meth:`transformers.PreTrainedTokenizer.__call__` for
                details.
            token_type_ids (`Tensor`, optional):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
                1]``:
                - 0 corresponds to a `sentence A` token,
                - 1 corresponds to a `sentence B` token.
                Defaults to `None`.
            position_ids (`Tensor`, `optional`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`.
            attention_mask (`Tensor`, optional):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                Defaults to `None`.

        Returns:
            logits (`Tensor`):
                A Tensor of the input text classification logits, shape as (batch_size, `num_classes`).

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

                tokenizer = SkepTokenizer.from_pretrained('ernie-1.0')
                model = SkepForSequenceClassification.from_pretrained('ernie-1.0')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor(v) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        _, pooled_output = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SkepForTokenClassification(SkepPretrainedModel):
    def __init__(self, skep, num_classes=2, dropout=None):
        super(SkepForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.skep = skep  # allow skep to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.skep.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.skep.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class SkepCrfForTokenClassification(nn.Layer):
    def __init__(self, skep, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.skep = skep  # allow skep to be config
        gru_hidden_size = 128

        self.gru = nn.GRU(self.skep.config["hidden_size"],
                          gru_hidden_size,
                          num_layers=2,
                          direction='bidirect')
        self.fc = nn.Linear(
            gru_hidden_size * 2,
            self.num_classes,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-0.1, high=0.1),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)))
        self.crf = LinearChainCrf(
            self.num_classes, crf_lr=0.2, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(
            self.crf.transitions, with_start_stop_tag=False)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                seq_lens=None,
                labels=None):
        sequence_output, _ = self.skep(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        bigru_output, _ = self.gru(
            sequence_output)  #, sequence_length=seq_lens)
        emission = self.fc(bigru_output)

        if labels is not None:
            loss = self.crf_loss(emission, seq_lens, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(emission, seq_lens)
            return prediction
