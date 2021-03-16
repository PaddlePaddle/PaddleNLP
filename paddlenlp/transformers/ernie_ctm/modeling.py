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
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddle.nn import TransformerEncoder, Linear, Layer, Embedding, LayerNorm, Tanh
from paddlenlp.layers.crf import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss

from .. import PretrainedModel, register_base_model

__all__ = [
    'ErnieCtmModel', "ErnieCtmPretrainedModel", 'ErnieCtmForPreTraining',
    'ErnieCtmPretrainingCriterion', 'ErnieCtmPretrainingHeads',
    'ErnieCtmForSequenceClassification', 'ErnieCtmForTokenClassification',
    'ErnieCtmForQuestionAnswering', 'ErnieCtmWordtagModel'
]


class ErnieCtmEmbeddings(Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(
            self,
            vocab_size,
            embedding_size=128,
            # hidden_size=768,
            hidden_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=16):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        # self.embedding_hidden_mapping_in = paddle.Linear(embedding_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
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

        embeddings = input_embedings + position_embeddings + token_type_embeddings
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


class ErnieCtmPretrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialzation and a simple interface for loading pretrained models.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "bert-base-uncased": { # TODO: fill it
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
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
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "bert-base-uncased":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bert-base-uncased.pdparams"
        }
    }
    base_model_prefix = "albert"

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
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 content_summary_index=0):
        super(ErnieCtmModel, self).__init__(config)

        self.pad_token_id = pad_token_id
        self.content_summary_index = content_summary_index
        self.initializer_range = initializer_range
        self.embeddings = ErnieCtmEmbeddings(
            vocab_size, embedding_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)
        self.embedding_hidden_mapping_in = paddle.Linear(embedding_size,
                                                         hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = ErnieCtmPooler(hidden_size)

        if self.config.use_content_summary is True:
            self.feature_fuse = paddle.Linear(hidden_size * 2,
                                              intermediate_size)
            self.feature_output = paddle.Linear(intermediate_size, hidden_size)

        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            use_content_summary=None,
            content_clone=False, ):
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
                          if use_content_summary else None)

        if use_content_summary is True:
            if content_clone is True:
                sequence_output = paddle.cat(
                    (sequence_output,
                     sequence_output[:, self.content_summary_index].clone(
                     ).unsqueeze(1).repeat(1, sequence_output.size(1), 1)), 2)
            else:
                sequence_output = paddle.cat(
                    (sequence_output,
                     sequence_output[:, self.content_summary_index].unsqueeze(
                         1).repeat(1, sequence_output.size(1), 1)), 2)
            sequence_output = self.feature_fuse(sequence_output)
            sequence_output = self.feature_output(sequence_output)

        return (sequence_output, pooled_output, content_output
                ) + encoder_outputs[1:]


class ErnieCtmMLMHead(torch.nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(ErnieCtmMLMHead, self).__init__()

        self.transform = nn.Linear(hidden_size, embedding_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[embedding_size, vocab_size],
            dtype=self.transform.weight.dtype,
            is_bias=True) if embedding_weights is None else embedding_weights
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


class ErnieCtmPretrainingHeads(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super().__init__()
        self.predictions = ErnieCtmMLMHead(hidden_size, vocab_size, activation,
                                           embedding_weights)
        #TODO: add dropout?
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class ErnieCtmForPreTraining(ErnieCtmPretrainedModel):
    def __init__(self, ernie_ctm):
        super(ErnieCtmForPreTraining, self).__init__()
        self.ernie_ctm = ernie_ctm
        self.cls = ErnieCtmPretrainingHeads(
            self.ernie_ctm.config["hidden_size"],
            self.ernie_ctm.config["vocab_size"],
            self.ernie_ctm.config["hidden_act"],
            embedding_weights=self.ernie_ctm.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None,
                sentence_order_label=None,
                use_content_summary=None,
                content_clone=False,
                masked_positions=None):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        sentence_order_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``. ``0`` indicates original order (sequence
            A, then sequence B), ``1`` indicates switched order (sequence B, then sequence A).
        Returns:
        Example::
            >>> from transformers import AlbertTokenizer, AlbertForPreTraining
            >>> import torch
            >>> tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            >>> model = AlbertForPreTraining.from_pretrained('albert-base-v2')
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, world, add_special_tokens=True)).unsqueeze(0)
            >>> outputs = model(input_ids)
            >>> prediction_logits = outputs.prediction_logits
            >>> sop_logits = outputs.sop_logits
        """
        with paddle.static.amp.fp16_guard():
            outputs = self.ernie_ctm(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                use_content_summary=use_content_summary,
                content_clone=content_clone, )

            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions)

            return prediction_scores, seq_relationship_score


class ErnieCtmPretrainingCriterion(Layer):
    def __init__(self, vocab_size):
        super().__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, sentence_order_label, masked_lm_scale):
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
                prediction_scores, masked_lm_labels, ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            sentence_order_loss = paddle.nn.functional.softmax_with_cross_entropy(
                seq_relationship_score, sentence_order_label)
        return paddle.sum(masked_lm_loss) + paddle.mean(sentence_order_loss)


class ErnieCtmForSequenceClassification(ErnieCtmPretrainedModel):
    def __init__(self, ernie_ctm, num_classes=2, dropout=None):
        super(ErnieCtmForSequenceClassification, self).__init__()
        self.num_labels = num_classes

        self.ernie_ctm = ernie_ctm
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie_ctm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            use_content_summary: typing.Optional[bool]=None,
            content_clone: typing.Optional[bool]=False, ):
        outputs = self.ernie_ctm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_content_summary=use_content_summary,
            content_clone=content_clone, )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class ErnieCtmForTokenClassification(ErnieCtmPretrainedModel):
    def __init__(self, ernie_ctm, num_classes=2, dropout=None):
        super(ErnieCtmForTokenClassification, self).__init__()
        self.num_labels = num_classes
        self.ernie_ctm = ernie_ctm
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie_ctm.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            use_content_summary: typing.Optional[bool]=None,
            content_clone: typing.Optional[bool]=False, ):
        outputs = self.ernie_ctm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_content_summary=use_content_summary,
            content_clone=content_clone, )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


class ErnieCtmForQuestionAnswering(ErnieCtmPretrainedModel):
    def __init__(self, ernie_ctm, num_labels, dropout=None):
        super(ErnieCtmForQuestionAnswering, self).__init__()
        self.num_labels = num_labels

        self.ernie_ctm = ernie_ctm
        self.classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                    num_labels)
        self.apply(self.init_weights)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                use_content_summary: typing.Optional[bool]=None,
                content_clone: typing.Optional[bool]=False):
        outputs = self.ernie_ctm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            use_content_summary=use_content_summary,
            content_clone=content_clone, )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class ErnieCtmWordtagModel(ErnieCtmPretrainedModel):
    """Wordtag task model.
    """

    def __init__(self,
                 ernie_ctm,
                 num_tag_labels,
                 num_sent_labels,
                 dropout=None,
                 crf_lr=0.1):
        super(ErnieCtmWordtagModel, self).__init__()
        self.num_tag_labels = num_tag_labels
        self.num_sent_labels = num_sent_labels
        self.ernie_ctm = ernie_ctm
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie_ctm.config["hidden_dropout_prob"])
        self.tag_classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                        self.num_tag_labels)
        self.crf = LinearChainCrf(
            self.num_tag_labels, crf_lr, with_start_stop_tag=True)
        self.sent_classifier = nn.Linear(self.ernie_ctm.config["hidden_size"],
                                         self.num_sent_labels)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(
            self.crf.transitions, with_start_stop_tag=True)

        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            tag_labels=None,
            sent_labels=None,
            use_content_summary=None,
            content_clone=False, ):
        outputs = self.ernie_ctm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids, )
        sequence_output, pooled_output = outputs[0], outputs[1]

        cls_logits = self.sent_classifier(pooled_output)
        seq_logits = self.tag_classifier(sequence_output)

        total_loss = None
        if tag_labels and sent_labels:
            loss_fct = paddle.CrossEntropyLoss(ignore_index=self._ignore_index)
            seq_loss = loss_fct(
                seq_logits.view(-1, self.num_sequence_label_tags),
                seq_logits.view(-1))
            cls_loss = loss_fct(cls_logits.view(-1), seq_logits.view(-1))
            seq_crf_loss = self.crf(seq_logits, lengths, None, tag_labels)
            total_loss = seq_loss + cls_loss + seq_crf_loss
            return total_loss, seq_logits, cls_logits, outputs[2:]
        else:
            return seq_logits, cls_logits, outputs[2:]
