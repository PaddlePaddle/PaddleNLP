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
import paddle.tensor as tensor
from paddle.nn import TransformerEncoder, Linear, Layer, Embedding, LayerNorm, Tanh
from paddlenlp.layers.crf import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss

from .. import PretrainedModel, register_base_model

__all__ = [
    'ErnieCtmModel', 'ErnieCtmWordtagModel', 'ErnieCtmForTokenClassification'
]


class ErnieCtmEmbeddings(Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 padding_idx=0,
                 cls_num=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.cls_num = cls_num

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            content_len = paddle.shape(input_ids)[1] - self.cls_num
            position_ids = paddle.concat([
                paddle.zeros(
                    shape=[self.cls_num], dtype="int64"), paddle.linspace(
                        1, content_len, content_len, dtype="int64")
            ])
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + token_type_embeddings + position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ErnieCtmPooler(Layer):
    """
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieCtmPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ErnieCtm models. It provides ErnieCtm related `model_config_file`,
    `resource_files_names`, `pretrained_resource_files_map`, `pretrained_init_configuration` and
    `base_model_prefix` for downloading and loading pretrained models.

    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-ctm": {
            "vocab_size": 23000,
            "embedding_size": 128,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "use_content_summary": True,
            "content_summary_index": 1,
            "cls_num": 2,
        },
        "wordtag": {
            "vocab_size": 23000,
            "embedding_size": 128,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "use_content_summary": True,
            "content_summary_index": 1,
            "cls_num": 2,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-ctm":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/ernie_ctm_base_pos.pdparams",
            "wordtag":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/wordtag_pos.pdparams"
        }
    }
    base_model_prefix = "ernie_ctm"

    def init_weights(self, layer):
        # Initialize weights
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie_ctm.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class ErnieCtmModel(ErnieCtmPretrainedModel):
    """
    The bare ErnieCtm Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Check the superclass documentation for the generic methods and the library implements for all its model.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (`int`):
            Vocabulary size of the ErnieCtm model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling ErnieCtmModel.
        embedding_size (`int`, optional):
            Dimensionality of the embedding layer.
            Defaults to ``128``.
        hidden_size (`int`, optional):
            Dimensionality of the encoder layers and the pooler layer.
            Defaults to ``768``.
        num_hidden_layers (`int`, optional):
            The number of encoder layers to be stacked.
            Defaults to ``12``.
        num_attention_heads (`int`, optional):
            The number of heads in multi-head attention(MHA).
            Defaults to ``12``.
        intermediate_size (`int`, optional):
            The hidden layer size in the feedforward network(FFN).
            Defaults to ``3072``.
        hidden_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (`float`, optional):
            The dropout probability used in MHA to drop some attention target.
            Defaults to ``0.1``.
        max_position_embeddings (`int`, optional):
            The size position embeddings of matrix, which dictates the maximum length
            for which the model can be run.
            Defaults to ``512``.
        type_vocab_size (`int`, optional):
            The vocabulary size of the `token_type_ids`. 
            Defaults to ``16``.
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
        pad_token_id (`int`, optional):
            The index of padding token for BigBird embedding.
            Defaults to ``0``.
        use_content_summary (`bool`, optional):
            If adding content summary tokens. 
            Defaults to ``True``.
        content_summary_index (`int`, optional):
            The number of the content summary tokens. Only valid when use_content_summary is True.
            Defaults to ``1``.
        cls_num (`int`, optional):
            The number of the CLS tokens. Only valid when use_content_summary is True.
            Defaults to ``2``.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 use_content_summary=True,
                 content_summary_index=1,
                 cls_num=2):
        super(ErnieCtmModel, self).__init__()

        self.pad_token_id = pad_token_id
        self.content_summary_index = content_summary_index
        self.initializer_range = initializer_range
        self.embeddings = ErnieCtmEmbeddings(
            vocab_size,
            embedding_size,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            padding_idx=pad_token_id,
            cls_num=cls_num)
        self.embedding_hidden_mapping_in = nn.Linear(embedding_size,
                                                     hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation="gelu",
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        encoder_layer.activation = nn.GELU(approximate=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = ErnieCtmPooler(hidden_size)

        self.use_content_summary = use_content_summary
        self.content_summary_index = content_summary_index
        if use_content_summary is True:
            self.feature_fuse = nn.Linear(hidden_size * 2, intermediate_size)
            self.feature_output = nn.Linear(intermediate_size, hidden_size)

        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                content_clone=False):
        r"""
        The ErnieCtmModel forward method, overrides the __call__() special method.
        
        Args:
            input_ids (`Tensor`):
                Indices of input sequence tokens in the vocabulary.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (`Tensor`, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can either be 0 or 1:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to ``None``, which means we don't add segment embeddings.
            attention_mask_list (`list`, optional):
                A list which contains some tensors used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. The tensors' shape will be
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`
            content_clone (`bool`, optional):
                Whether the content_output is clone from sequence_output. If set to `True`, the content_output is
                clone from sequence_output, which may cause the classification task impact on the sequence labeling task.
        Returns:
            A tuple of shape (``sequence_output``, ``pooled_output``, ``content_output``).
            
            With the fields:
            - sequence_output (`Tensor`):
                Sequence of output at the last layer of the model. Its data type should be float32 and
                has a shape of [batch_size, sequence_length, hidden_size].
            - pooled_output (`Tensor`):
                The output of first token (`[CLS]`) in sequence. Its data type should be float32 and
                has a shape of [batch_size, hidden_size].
            - content_output (`Tensor`):
                The output of content summary token (`[CLS1]` in sequence). Its data type should be float32 and
                has a shape of [batch_size, hidden_size].
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

        embedding_output = self.embedding_hidden_mapping_in(embedding_output)

        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs

        pooled_output = self.pooler(sequence_output)
        content_output = (sequence_output[:, self.content_summary_index]
                          if self.use_content_summary else None)

        if self.use_content_summary is True:
            if content_clone is True:
                sequence_output = paddle.concat(
                    (sequence_output,
                     sequence_output[:, self.content_summary_index].clone(
                     ).unsqueeze([1]).expand_as(sequence_output)), 2)
            else:
                sequence_output = paddle.concat(
                    (sequence_output,
                     sequence_output[:, self.content_summary_index].unsqueeze(
                         [1]).expand_as(sequence_output)), 2)

            sequence_output = self.feature_fuse(sequence_output)

            sequence_output = self.feature_output(sequence_output)

        return sequence_output, pooled_output, content_output


class ErnieCtmWordtagModel(ErnieCtmPretrainedModel):
    """
    ErnieCtmWordtag Model with a token classification head on top (a crf layer on top of the hidden-states output) .
    e.g. for Named-Entity-Recognition (NER) tasks.

    Args:
        ernie_ctm (:clss:`ErnieCtmModel`):
            An instance of :class:`ErnieCtmModel`.
        num_tag (`int`):
            The number of tags.
        num_cls_label (`int`):
            The number of sentence classification label.
        crf_lr (`float`):
            The learning rate of the crf.
        ignore_index (`index`):
            The ignore prediction index when calculating the cross entropy loss.
    """

    def __init__(self,
                 ernie_ctm,
                 num_tag,
                 num_cls_label,
                 crf_lr=100,
                 ignore_index=0):
        super(ErnieCtmWordtagModel, self).__init__()
        self.num_tag = num_tag
        self.num_cls_label = num_cls_label
        self.ernie_ctm = ernie_ctm
        self.tag_classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                        self.num_tag)
        self.sent_classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                         self.num_cls_label)
        self.crf = LinearChainCrf(
            self.num_tag, crf_lr, with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(
            self.crf.transitions, with_start_stop_tag=False)
        self.ignore_index = ignore_index

        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                lengths=None,
                tag_labels=None,
                cls_label=None):
        outputs = self.ernie_ctm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids, )
        sequence_output, pooled_output = outputs[0], outputs[1]
        sequence_output = sequence_output
        pooled_output = pooled_output

        cls_logits = self.sent_classifier(pooled_output)

        seq_logits = self.tag_classifier(sequence_output)
        seq_logits = seq_logits

        total_loss = None
        if tag_labels is not None and cls_label is not None:
            loss_fct = nn.loss.CrossEntropyLoss(ignore_index=self.ignore_index)
            cls_loss = loss_fct(cls_logits, cls_label.reshape([-1]))
            seq_crf_loss = self.crf_loss(seq_logits, lengths, tag_labels)
            total_loss = cls_loss + seq_crf_loss
            return total_loss, seq_logits, cls_logits
        else:
            return seq_logits, cls_logits


class ErnieCtmForTokenClassification(ErnieCtmPretrainedModel):
    def __init__(self, ernie_ctm, num_classes=2, dropout=None):
        super(ErnieCtmForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie_ctm = ernie_ctm  # allow ernie_ctm to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie_ctm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _, _ = self.ernie_ctm(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
