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

__all__ = ['ErnieCtmModel', 'ErnieCtmWordtagModel']


class ErnieCtmEmbeddings(Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 padding_idx=0):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
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
    """An abstract class to handle weights initialzation and a simple interface for loading pretrained models.
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
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-ctm":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_ctm_base.pdparams"
        }
    }
    base_model_prefix = "ernie_ctm"

    def init_weights(self, layer):
        """ Initialization hook """
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
                 content_summary_index=1):
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
            padding_idx=pad_token_id)
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
        """Forward process.
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
    """Wordtag task model.
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
            seq_crf_loss = self.crf_loss(seq_logits, lengths, None, tag_labels)
            total_loss = cls_loss + seq_crf_loss
            return total_loss, seq_logits, cls_logits
        else:
            return seq_logits, cls_logits
