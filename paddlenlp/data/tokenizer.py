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

import jieba
from .vocab import Vocab


def get_idx_from_word(word, word_to_idx, unk_word):
    if word in word_to_idx:
        return word_to_idx[word]
    return word_to_idx[unk_word]


class BaseTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def get_tokenizer(self):
        return self.tokenizer

    def cut(self, sentence):
        pass

    def encode(self, sentence):
        pass


class JiebaTokenizer(BaseTokenizer):
    def __init__(self, vocab):
        super(JiebaTokenizer, self).__init__(vocab)
        self.tokenizer = jieba.Tokenizer()
        # initialize tokenizer
        self.tokenizer.FREQ = {key: 1 for key in self.vocab.token_to_idx.keys()}
        self.tokenizer.total = len(self.tokenizer.FREQ)
        self.tokenizer.initialized = True

    def cut(self, sentence, cut_all=False, use_hmm=True):
        return self.tokenizer.lcut(sentence, cut_all, use_hmm)

    def encode(self, sentence, cut_all=False, use_hmm=True):
        words = self.cut(sentence, cut_all, use_hmm)
        return [
            get_idx_from_word(word, self.vocab.token_to_idx,
                              self.vocab.unk_token) for word in words
        ]
