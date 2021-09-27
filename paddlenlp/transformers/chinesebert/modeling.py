#encoding=utf8
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.transformers.bert.modeling import BertPooler, BertPretrainingHeads

from .fusion_embedding import FusionBertEmbeddings

__all__ = [
    "ChineseBertModel",
    "ChineseBertPretrainedModel",
    "ChineseBertForPretraining",
    "ChineseBertPretrainingCriterion",
    "ChineseBertForSequenceClassification",
    "ChineseBertForTokenClassification",
    "ChineseBertForQuestionAnswering",
]


class ChineseBertPretrainedModel(PretrainedModel):
    base_model_prefix = "chinesebert"
    model_config_file = "model_config.json"

    pretrained_init_configuration = {
        "ChineseBERT-base": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 23236,
            "glyph_embedding_dim": 1728,
            "pinyin_map_len": 32,
        },
        "ChineseBERT-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 23236,
            "glyph_embedding_dim": 1728,
            "pinyin_map_len": 32,
        },
    }

    resource_files_names = {"model_state": "model_state.pdparams"}

    pretrained_resource_files_map = {
        "model_state": {
            "ChineseBERT-base": "E:/ChineseBERT/ChineseBERT_paddle/ChineseBERT-base/chinesebert-base.pdparams",
            # "ChineseBERT-large": "/home/aistudio/data/data106015/ChineseBERT-large-paddle/chinesebert-large.pdparams",
            "ChineseBERT-large": "/home/aistudio/data/data109231/model_state.pdparams",
        }
    }

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
                        else self.chinesebert.config["initializer_range"],
                        shape=layer.weight.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = (
                self.layer_norm_eps
                if hasattr(self, "layer_norm_eps")
                else self.chinesebert.config["layer_norm_eps"]
            )


@register_base_model
class ChineseBertModel(ChineseBertPretrainedModel):
    def __init__(
        self,
        vocab_size=23236,
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
        pad_token_id=0,
        pool_act="tanh",
        layer_norm_eps=1e-12,
        glyph_embedding_dim=1728,
        pinyin_map_len=32,
    ):
        super(ChineseBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.embeddings = FusionBertEmbeddings(
            vocab_size,
            hidden_size,
            pad_token_id,
            type_vocab_size,
            max_position_embeddings,
            pinyin_map_len,
            glyph_embedding_dim,
            layer_norm_eps,
            hidden_dropout_prob,
        )
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
        self.pooler = BertPooler(hidden_size, pool_act)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        pinyin_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype)
                * -1e9,
                axis=[1, 2],
            )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            pinyin_ids=pinyin_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

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
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class ChineseBertForQuestionAnswering(ChineseBertPretrainedModel):
    def __init__(self, chinesebert):
        super(ChineseBertForQuestionAnswering, self).__init__()
        self.chinesebert = chinesebert  # allow chinesebert to be config
        self.classifier = nn.Linear(self.chinesebert.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, pinyin_ids=None, token_type_ids=None):
        sequence_output, _ = self.chinesebert(
            input_ids,
            pinyin_ids,
            token_type_ids=token_type_ids,
            position_ids=None
        )

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class ChineseBertForSequenceClassification(ChineseBertPretrainedModel):
    """
    Model for sentence (pair) classification task with BERT.
    Args:
        bert (BertModel): An instance of BertModel.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of BERT.
            If None, use the same value as `hidden_dropout_prob` of `BertModel`
            instance `bert`. Default None
    """

    def __init__(self, chinesebert, num_classes=2, dropout=None):
        super(ChineseBertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.chinesebert = chinesebert  # allow chinesebert to be config
        self.dropout = nn.Dropout(
            dropout
            if dropout is not None
            else self.chinesebert.config["hidden_dropout_prob"]
        )
        self.classifier = nn.Linear(
            self.chinesebert.config["hidden_size"], self.num_classes
        )
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        pinyin_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
    ):
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
    def __init__(self, chinesebert, num_classes=2, dropout=None):
        super(ChineseBertForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.chinesebert = chinesebert  # allow chinesebert to be config
        self.dropout = nn.Dropout(
            dropout
            if dropout is not None
            else self.chinesebert.config["hidden_dropout_prob"]
        )
        self.classifier = nn.Linear(
            self.chinesebert.config["hidden_size"], self.num_classes
        )
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        pinyin_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
    ):
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
    def __init__(self, chinesebert):
        super(ChineseBertForPretraining, self).__init__()
        self.chinesebert = chinesebert
        self.cls = BertPretrainingHeads(
            self.chinesebert.config["hidden_size"],
            self.chinesebert.config["vocab_size"],
            self.chinesebert.config["hidden_act"],
            embedding_weights=self.chinesebert.embeddings.word_embeddings.weight,
        )

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        pinyin_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        masked_positions=None,
    ):
        with paddle.static.amp.fp16_guard():
            outputs = self.chinesebert(
                input_ids,
                pinyin_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions
            )
            return prediction_scores, seq_relationship_score


class ChineseBertPretrainingCriterion(nn.Layer):
    def __init__(self, vocab_size):
        super(ChineseBertPretrainingCriterion, self).__init__()
        # CrossEntropyLoss is expensive since the inner reshape (copy)
        self.loss_fn = nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(
        self,
        prediction_scores,
        seq_relationship_score,
        masked_lm_labels,
        next_sentence_labels,
        masked_lm_scale,
    ):
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.cross_entropy(
                prediction_scores, masked_lm_labels, reduction="none", ignore_index=-1
            )
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score, next_sentence_labels, reduction="none"
            )
        return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)
