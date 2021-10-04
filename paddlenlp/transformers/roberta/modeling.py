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

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .. import PretrainedModel, register_base_model

__all__ = [
    'RobertaModel', 'RobertaPretrainedModel',
    'RobertaForSequenceClassification', 'RobertaForTokenClassification',
    'RobertaForQuestionAnswering', 'RobertaForMultipleChoice',
    'RobertaForMaskedLM', 'RobertaForCausalLM'
]


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


class RobertaEmbeddings(nn.Layer):
    r"""
    Include embeddings from word, position and token_type embeddings.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 pad_token_id=0):
        super(RobertaEmbeddings, self).__init__()
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
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
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


class RobertaPooler(nn.Layer):
    def __init__(self, hidden_size):
        super(RobertaPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained RoBERTa models. It provides RoBERTa related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. 
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "roberta-wwm-ext": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "roberta-wwm-ext-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "rbt3": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0,
        },
        "rbtl3": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "roberta-base": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 514,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": 50265,
            "pad_token_id": 1
        },
        "roberta-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 514,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": 50265,
            "pad_token_id": 1
        },
        "roberta-base-squad2": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 514,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": 50265,
            "pad_token_id": 1
        },
        "roberta-base-finetuned-chinanews-chinese": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "tiny-distilroberta-base": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 2,
            "initializer_range": 0.02,
            "intermediate_size": 2,
            "max_position_embeddings": 514,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "type_vocab_size": 1,
            "vocab_size": 50265,
            "pad_token_id": 1
        },
        "roberta-base-finetuned-cluener2020-chinese": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": 21128,
            "pad_token_id": 0
        },
        "roberta-base-chinese-extractive-qa": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": 21128,
            "pad_token_id": 0
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "roberta-wwm-ext":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roberta_base/roberta_chn_base.pdparams",
            "roberta-wwm-ext-large":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roberta_large/roberta_chn_large.pdparams",
            "rbt3":
            "https://paddlenlp.bj.bcebos.com/models/transformers/rbt3/rbt3_chn_large.pdparams",
            "rbtl3":
            "https://paddlenlp.bj.bcebos.com/models/transformers/rbtl3/rbtl3_chn_large.pdparams",
            "roberta-base": "roberta-base.pdparams",  # 从百度网盘下载
            "roberta-large": "roberta-large.pdparams",  # 从百度网盘下载
            "roberta-base-squad2": "roberta-base-squad2.pdparams",  # 从百度网盘下载
            "roberta-base-finetuned-chinanews-chinese":
            "roberta-base-finetuned-chinanews-chinese.pdparams",  # 从百度网盘下载
            "tiny-distilroberta-base":
            "tiny-distilroberta-base.pdparams",  # 从百度网盘下载
            "roberta-base-finetuned-cluener2020-chinese":
            "roberta-base-finetuned-cluener2020-chinese.pdparams",  # 从百度网盘下载
            "roberta-base-chinese-extractive-qa":
            "roberta-base-chinese-extractive-qa.pdparams",  # 从百度网盘下载
        }
    }
    base_model_prefix = "roberta"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.roberta.config["initializer_range"],
                    shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class RobertaModel(RobertaPretrainedModel):
    r"""
    The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:

        vocab_size (int):
            Vocabulary size of the RoBERTa model. Also is the vocab size of token embedding matrix.
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
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.RobertaModel`.
            Defaults to ``2``.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to 0.02.
            
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`RobertaPretrainedModel._init_weights()` for how weights are initialized in `RobertaModel`.

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
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(RobertaModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = RobertaEmbeddings(
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
        self.pooler = RobertaPooler(hidden_size)
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
                from paddlenlp.transformers import RobertaModel, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaModel.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
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


class RobertaForQuestionAnswering(RobertaPretrainedModel):
    r"""
    Model for sentence (pair) classification task with RoBERTa.

    Args:
        roberta (RobertaModel): 
            An instance of `paddlenlp.transformers.RobertaModel`.
        num_classes (int, optional): 
            The number of classes. Default to `2`.
        dropout (float, optional): 
            The dropout probability for output of RoBERTa. 
            If None, use the same value as `hidden_dropout_prob` 
            of `paddlenlp.transformers.RobertaModel` instance. Defaults to `None`.
    """

    def __init__(self, roberta, dropout=None):
        super(RobertaForQuestionAnswering, self).__init__()
        self.roberta = roberta  # allow roberta to be config
        self.classifier = nn.Linear(self.roberta.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None):
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
                from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForSequenceClassification.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        sequence_output, _ = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class RobertaForSequenceClassification(RobertaPretrainedModel):
    r"""
    RoBERTa Model transformer with a sequence classification/regression head on top 
    (a linear layer on top of the pooledoutput) e.g. for GLUE tasks.


    Args:
        roberta (`RobertaModel`): 
            An instance of `RobertaModel`.
        num_classes (int, optional): 
            The number of classes. Default to `2`.
        dropout (float, optional): 
            The dropout probability for output of RoBERTa. 
            If None, use the same value as `hidden_dropout_prob` 
            of `RobertaModel` instance `roberta`. Defaults to `None`.
    """

    def __init__(self, roberta, num_classes=2, dropout=None):
        super(RobertaForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.roberta = roberta  # allow roberta to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"],
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
                from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForSequenceClassification.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        _, pooled_output = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RobertaForTokenClassification(RobertaPretrainedModel):
    r"""
    RoBERTa Model transformer with a sequence classification/regression head on top 
    (a linear layer on top of the pooledoutput) e.g. for GLUE tasks.


    Args:
        roberta (`RobertaModel`): 
            An instance of `RobertaModel`.
        num_classes (int, optional): 
            The number of classes. Default to `2`.
        dropout (float, optional): 
            The dropout probability for output of RoBERTa. 
            If None, use the same value as `hidden_dropout_prob` 
            of `RobertaModel` instance `roberta`. Defaults to `None`.
    """

    def __init__(self, roberta, num_classes=2, dropout=None):
        super(RobertaForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.roberta = roberta  # allow roberta to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"],
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
                from paddlenlp.transformers import RobertaForTokenClassification, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-wwm-ext')
                model = RobertaForTokenClassification.from_pretrained('roberta-wwm-ext')

                inputs = tokenizer("这是个测试样例")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        sequence_output, _ = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class RobertaForMultipleChoice(RobertaPretrainedModel):
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    Args:
        roberta (:class:`RobertaModel`):
            An instance of RobertaModel.
        num_choices (int, optional):
            The number of choices. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of ROBERTA.
            If None, use the same value as `hidden_dropout_prob` of `RobertaModel`
            instance `roberta`. Defaults to None.
    """

    def __init__(self, roberta, num_choices=2, dropout=None):
        super(RobertaForMultipleChoice, self).__init__()
        self.roberta = roberta  # allow roberta to be config
        self.num_choices = num_choices
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"], 1)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        r"""
        The RobertaForMultipleChoice forward method, overrides the __call__() special method.
        Args:
            input_ids (Tensor):
                See :class:`RobertaModel` and shape as [batch_size, num_choice, sequence_length].
            position_ids(Tensor, optional):
                See :class:`RobertaModel` and shape as [batch_size, num_choice, sequence_length].
            attention_mask (list, optional):
                See :class:`RobertaModel` and shape as [batch_size, num_choice, sequence_length].

        Returns:
            Tensor: Returns tensor `reshaped_logits`, a tensor of the multiple choice classification logits.
            Shape as `[batch_size, num_choice]` and dtype as float32.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import RobertaForMultipleChoice, RobertaTokenizer

                tokenizer = RobertaTokenizer.from_pretrained('roberta-base-uncased')
                model = RobertaForTokenClassification.from_pretrained('roberta-base-uncased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

                outputs = model(**inputs)
        """
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(
            -1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1,
                                                       position_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(
                shape=(-1, attention_mask.shape[-1]))

        _, pooled_output = self.roberta(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(
            shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        return reshaped_logits


class RobertaLMHead(nn.Layer):
    """
    Roberta Model with a `language modeling` head on top.
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation="gelu",
                 embedding_weights=None):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = ACT2FN[activation]
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.decoder_weight = embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = (paddle.matmul(
            hidden_states, self.decoder_weight, transpose_y=True) +
                         self.decoder_bias)

        return hidden_states


class RobertaForMaskedLM(RobertaPretrainedModel):
    """
    Roberta Model with pretraining tasks on top.
    Args:
        Roberta (:class:`RobertaModel`):
            An instance of :class:`RobertaModel`.
    """

    def __init__(self, roberta):
        super(RobertaForMaskedLM, self).__init__()
        self.roberta = roberta
        self.lm_head = RobertaLMHead(
            self.roberta.config["hidden_size"],
            self.roberta.config["vocab_size"],
            self.roberta.config["hidden_act"],
            self.roberta.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                labels=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            position_ids(Tensor, optional):
                See :class:`RobertaModel`.
            attention_mask (list, optional):
                See :class:`RobertaModel`.
            labels (Tensor, optional):
                The Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ..., vocab_size]`` Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., vocab_size]`` Its shape is [batch_size, sequence_length].
        Returns:
            tuple: Returns tuple (`masked_lm_loss`, `prediction_scores`, ``sequence_output`).
            With the fields:
            - `masked_lm_loss` (Tensor):
                The masked lm loss. Its data type should be float32 and its shape is [1].
            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32. Its shape is [batch_size, sequence_length, vocab_size].
            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model. Its data type should be float32. Its shape is `[batch_size, sequence_length, hidden_size]`.
        """
        sequence_output, _ = self.roberta(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(shape=(
                    -1, self.roberta.config["vocab_size"])),
                labels.reshape(shape=(-1, )), )
            return masked_lm_loss, prediction_scores, sequence_output

        return prediction_scores, sequence_output


class RobertaForCausalLM(RobertaPretrainedModel):
    """
    Roberta Model for casual language model task.
    Args:
        Roberta (:class:`RobertaModel`):
            An instance of :class:`RobertaModel`.
    """

    def __init__(self, roberta):
        super(RobertaForCausalLM, self).__init__()
        self.roberta = roberta
        self.lm_head = RobertaLMHead(
            self.roberta.config["hidden_size"],
            self.roberta.config["vocab_size"],
            self.roberta.config["hidden_act"],
            self.roberta.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                labels=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`RobertaModel`.
            position_ids(Tensor, optional):
                See :class:`RobertaModel`.
            attention_mask (list, optional):
                See :class:`RobertaModel`.
            labels (Tensor, optional):
                The Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ..., vocab_size]`` Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., vocab_size]`` Its shape is [batch_size, sequence_length].
        Returns:
            tuple: Returns tuple (`masked_lm_loss`, `prediction_scores`, ``sequence_output`).
            With the fields:
            - `masked_lm_loss` (Tensor):
                The masked lm loss. Its data type should be float32 and its shape is [1].
            - `prediction_scores` (Tensor):
                The scores of masked token prediction. Its data type should be float32. Its shape is [batch_size, sequence_length, vocab_size].
            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model. Its data type should be float32. Its shape is `[batch_size, sequence_length, hidden_size]`.
        """
        sequence_output, _ = self.roberta(
            input_ids, position_ids=position_ids, attention_mask=attention_mask)
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                paddle.reshape(shifted_prediction_scores,
                               [-1, self.roberta.config['vocab_size']]),
                paddle.reshape(labels, [-1]))

            return lm_loss, prediction_scores, sequence_output

        return prediction_scores, sequence_output
