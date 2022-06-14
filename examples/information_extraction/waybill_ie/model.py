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
from paddlenlp.transformers import ErniePretrainedModel
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss
from paddlenlp.utils.tools import compare_version

if compare_version(paddle.version.full_version, "2.2.0") >= 0:
    # paddle.text.ViterbiDecoder is supported by paddle after version 2.2.0
    from paddle.text import ViterbiDecoder
else:
    from paddlenlp.layers.crf import ViterbiDecoder


class BiGRUWithCRF(nn.Layer):

    def __init__(self,
                 emb_size,
                 hidden_size,
                 word_num,
                 label_num,
                 use_w2v_emb=False):
        super(BiGRUWithCRF, self).__init__()
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(
                extended_vocab_path='./data/word.dic', unknown_token='OOV')
        else:
            self.word_emb = nn.Embedding(word_num, emb_size)
        self.gru = nn.GRU(emb_size,
                          hidden_size,
                          num_layers=2,
                          direction='bidirect')
        # We need `label_num + 2` for appending BOS and EOS tag
        self.fc = nn.Linear(hidden_size * 2, label_num + 2)
        self.crf = LinearChainCrf(label_num)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, inputs, lengths, labels=None):
        embs = self.word_emb(inputs)
        output, _ = self.gru(embs)
        emission = self.fc(output)
        if labels is not None:
            loss = self.crf_loss(emission, lengths, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(emission, lengths)
            return prediction


class ErnieCrfForTokenClassification(nn.Layer):

    def __init__(self, ernie, crf_lr=100):
        super().__init__()
        self.num_classes = ernie.num_classes
        self.ernie = ernie  # allow ernie to be config
        self.crf = LinearChainCrf(self.num_classes,
                                  crf_lr=crf_lr,
                                  with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions, False)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                lengths=None,
                labels=None):
        logits = self.ernie(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids)

        if labels is not None:
            loss = self.crf_loss(logits, lengths, labels)
            return loss
        else:
            _, prediction = self.viterbi_decoder(logits, lengths)
            return prediction
