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

import paddle.nn as nn
from paddlenlp.transformers.ernie_gram.modeling import ErnieGramPretrainedModel, ErniePooler, ErnieGramEmbeddings


class ErnieGramForCSC(ErnieGramPretrainedModel):
    def __init__(self,
                 vocab_size,
                 pinyin_vocab_size,
                 emb_size=768,
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
                 rel_pos_size=None):
        super(ErnieGramModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = ErnieGramEmbeddings(
            vocab_size, emb_size, hidden_dropout_prob, max_position_embeddings,
            type_vocab_size, pad_token_id, rel_pos_size, num_attention_heads)

        self.pinyin_embeddings = nn.Embedding(
            self.pinyin_vocab_size, emb_size, padding_idx=pad_token_id)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.detection_layer = nn.Linear(hidden_size, 2)
        self.correction_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = paddle.nn.Softmax()
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        pinyin_embedding_output = self.pinyin_embeddings(pinyin_ids)

        # Detection module
        detection_outputs = self.encoder(embedding_output, attention_mask)
        # [B, T, 2]
        detection_error_probs = self.softmax(
            self.detection_layer(detection_outputs))
        # Correction module 
        word_pinyin_embedding_output = detection_error_probs[:, :, 0:1] * embedding_output \
                    + detection_error_probs[:,:, 1:2] * pinyin_embedding_output

        correction_outputs = self.encoder(word_pinyin_embedding_output,
                                          attention_mask)
        # [B, T, V]
        correction_logits = self.correction_layer(correction_output)
        return detection_error_probs, correction_logits
