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


class PretrainedModelForCSC(nn.Layer):
    def __init__(self, pretrained_model, pinyin_vocab_size, pad_pinyin_id=0):
        super(PretrainedModelForCSC, self).__init__()
        self.pretrained_model = pretrained_model
        emb_size = self.pretrained_model.config["hidden_size"]
        hidden_size = self.pretrained_model.config["hidden_size"]
        vocab_size = self.pretrained_model.config["vocab_size"]

        self.pad_token_id = self.pretrained_model.config["pad_token_id"]
        self.pinyin_vocab_size = pinyin_vocab_size
        self.pad_pinyin_id = pad_pinyin_id
        self.pinyin_embeddings = nn.Embedding(
            self.pinyin_vocab_size, emb_size, padding_idx=pad_pinyin_id)
        self.detection_layer = nn.Linear(hidden_size, 2)
        self.correction_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax()

    def forward(self,
                input_ids,
                pinyin_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.detection_layer.weight.dtype) * -1e9,
                axis=[1, 2])

        embedding_output = self.pretrained_model.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        pinyin_embedding_output = self.pinyin_embeddings(pinyin_ids)

        # Detection module
        detection_outputs = self.pretrained_model.encoder(embedding_output,
                                                          attention_mask)
        # [B, T, 2]
        detection_error_probs = self.softmax(
            self.detection_layer(detection_outputs))
        # Correction module 
        word_pinyin_embedding_output = detection_error_probs[:, :, 0:1] * embedding_output \
                    + detection_error_probs[:,:, 1:2] * pinyin_embedding_output

        correction_outputs = self.pretrained_model.encoder(
            word_pinyin_embedding_output, attention_mask)
        # [B, T, V]
        correction_logits = self.correction_layer(correction_outputs)
        return detection_error_probs, correction_logits
