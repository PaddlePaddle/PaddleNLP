# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .. import PretrainedModel, register_base_model

__all__ = [
    'ErnieModel', 'ErniePretrainedModel', 'ErnieForSequenceClassification',
    'ErnieForTokenClassification', 'ErnieForQuestionAnswering',
    'ErnieForPretraining', 'ErniePretrainingCriterion'
]


class ErnieEmbeddings(nn.Layer):
    r"""
    Include embeddings from word, position and token_type embeddings.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 pad_token_id=0):
        super(ErnieEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            #seq_length = input_ids.shape[1]
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")
        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErniePooler(nn.Layer):
    def __init__(self, hidden_size):
        super(ErniePooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErniePretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained ERNIE models. It provides ERNIE related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. 
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-1.0": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 513,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 18000,
            "pad_token_id": 0,
        },
        "ernie-tiny": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 600,
            "num_attention_heads": 16,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 50006,
            "pad_token_id": 0,
        },
        "ernie-2.0-en": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "ernie-2.0-en-finetuned-squad": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "ernie-2.0-large-en": {
            "attention_probs_dropout_prob": 0.1,
            "intermediate_size": 4096,  # special for ernie-2.0-large-en
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-1.0":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams",
            "ernie-tiny":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/ernie_tiny.pdparams",
            "ernie-2.0-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_base/ernie_v2_eng_base.pdparams",
            "ernie-2.0-en-finetuned-squad":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_base/ernie_v2_eng_base_finetuned_squad.pdparams",
            "ernie-2.0-large-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_large/ernie_v2_eng_large.pdparams",
        }
    }
    base_model_prefix = "ernie"

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
                        self.ernie.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5


@register_base_model
class ErnieModel(ErniePretrainedModel):
    r"""
    The bare ERNIE Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:

        vocab_size (int):
            Vocabulary size of the ERNIE model. Also is the vocab size of token embedding matrix.
        hidden_size (int, optional):
            Dimension of the encoder layers and the pooler layer. Defaults to ``768``.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to ``12``.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to ``12``.
        intermediate_size (int, optional):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to ``3072``.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to ``0.1``.
        max_position_embeddings (int, optional):
            The max position index of an input sequence. Defaults to ``512``.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.ErnieModel`.
            Defaults to ``2``.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to 0.02.
            
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`ErniePretrainedModel._init_weights()` for how weights are initialized in `ErnieModel`.

        pad_token_id(int, optional):
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
        super(ErnieModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = ErnieEmbeddings(
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
        self.pooler = ErniePooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
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
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`. Shape as `(batch_sie, num_tokens)` and dtype as `int32` or `int64`.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.

        Returns:
            A tuple of shape (``sequence_output``, ``pooled_output``).

            With the fields:
            - sequence_output (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be `float` and has a shape of `(batch_size, seq_lens, hidden_size)`.
                ``seq_lens`` corresponds to the length of input sequence.
            - pooled_output (Tensor):
                A Tensor of the first token representation.
                It's data type should be `float` and has a shape of `(batch_size, hidden_size]`.
                We "pool" the model by simply taking the hidden state corresponding to the first token.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieModel, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = ErnieModel.from_pretrained('ernie-1.0')

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


class ErnieForSequenceClassification(ErniePretrainedModel):
    r"""
    Model for sentence (pair) classification task with ERNIE.

    Args:
        ernie (ErnieModel): 
            An instance of `paddlenlp.transformers.ErnieModel`.
        num_classes (int, optional): 
            The number of classes. Default to `2`.
        dropout (float, optional): 
            The dropout probability for output of ERNIE. 
            If None, use the same value as `hidden_dropout_prob` 
            of `paddlenlp.transformers.ErnieModel` instance. Defaults to `None`.
    """

    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
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
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`. Shape as `(batch_sie, num_tokens)` and dtype as `int32` or `int64`.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.


        Returns:
            logits (Tensor):
                A Tensor of the input text classification logits.
                Shape as `(batch_size, num_classes)` and dtype as `float`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = ErnieForSequenceClassification.from_pretrained('ernie-1.0')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        _, pooled_output = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ErnieForQuestionAnswering(ErniePretrainedModel):
    """
    Model for Question and Answering task with ERNIE.


    Args:
        ernie (`ErnieModel`): 
            An instance of `ErnieModel`.
    """

    def __init__(self, ernie):
        super(ErnieForQuestionAnswering, self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
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
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`. Shape as `(batch_sie, num_tokens)` and dtype as `int32` or `int64`.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.


        Returns:
            A tuple of shape (``start_logits``, ``end_logits``).

            With the fields:
            - start_logits(Tensor): The logits of start position of prediction answer.
            - end_logits(Tensor): The logits of end position of prediction answer.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieForQuestionAnswering, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = ErnieForQuestionAnswering.from_pretrained('ernie-1.0')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """

        sequence_output, _ = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class ErnieForTokenClassification(ErniePretrainedModel):
    r"""
    ERNIE Model transformer with a sequence classification/regression head on top 
    (a linear layer on top of the pooledoutput) e.g. for GLUE tasks.


    Args:
        ernie (`ErnieModel`): 
            An instance of `ErnieModel`.
        num_classes (int, optional): 
            The number of classes. Default to `2`.
        dropout (float, optional): 
            The dropout probability for output of ERNIE. 
            If None, use the same value as `hidden_dropout_prob` 
            of `ErnieModel` instance `ernie`. Defaults to `None`.
    """

    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        Args:
            input_ids (Tensor):
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
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`. Shape as `(batch_sie, num_tokens)` and dtype as `int32` or `int64`.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.


        Returns:
            logits (Tensor):
                A Tensor of the input text classification logits, shape as (batch_size, seq_lens, `num_classes`).
                seq_lens mean the number of tokens of the input sequence.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieForTokenClassification, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = ErnieForTokenClassification.from_pretrained('ernie-1.0')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)
        """
        sequence_output, _ = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class ErnieLMPredictionHead(nn.Layer):
    r"""
    Bert Model with a `language modeling` head on top.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(ErnieLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[vocab_size, hidden_size],
            dtype=self.transform.weight.dtype,
            is_bias=False) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class ErniePretrainingHeads(nn.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(ErniePretrainingHeads, self).__init__()
        self.predictions = ErnieLMPredictionHead(hidden_size, vocab_size,
                                                 activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class ErnieForPretraining(ErniePretrainedModel):
    r"""
    Bert Model with two heads on top as done during the pretraining: 
    a `masked language modeling` head and a `next sentence prediction (classification)` head.

    """

    def __init__(self, ernie):
        super(ErnieForPretraining, self).__init__()
        self.ernie = ernie
        self.cls = ErniePretrainingHeads(
            self.ernie.config["hidden_size"],
            self.ernie.config["vocab_size"],
            self.ernie.config["hidden_act"],
            embedding_weights=self.ernie.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        r"""
        Args:
            input_ids (Tensor):
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
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                config.max_position_embeddings - 1]``.
                Defaults to `None`. Shape as `(batch_sie, num_tokens)` and dtype as `int32` or `int64`.
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.


        Returns:
            A tuple of shape (``prediction_scores``, ``seq_relationship_score``).

            With the fields:
            - prediction_scores(Tensor): The scores of prediction on masked token.
            - seq_relationship_score(Tensor): The scores of next sentence prediction.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import ErnieForTokenClassification, ErnieTokenizer

                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
                model = ErnieForTokenClassification.from_pretrained('ernie-1.0')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        with paddle.static.amp.fp16_guard():
            outputs = self.ernie(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask)
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions)
            return prediction_scores, seq_relationship_score


class ErniePretrainingCriterion(paddle.nn.Layer):
    r"""
    The loss output of Bert Model during the pretraining: 
    a `masked language modeling` head and a `next sentence prediction (classification)` head.

    """

    def __init__(self, vocab_size):
        super(ErniePretrainingCriterion, self).__init__()
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels, masked_lm_scale):
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(
                prediction_scores,
                masked_lm_labels,
                ignore_index=-1,
                reduction='none')
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score, next_sentence_labels, reduction='none')
            return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)
