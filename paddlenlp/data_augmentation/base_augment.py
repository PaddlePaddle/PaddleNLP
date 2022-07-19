# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import math
import random
from typing import Iterable

import numpy as np
import paddle
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url

from ..utils.env import DATA_HOME
from ..data import Vocab, JiebaTokenizer


class BaseAugment(object):
    """
    A base class for data augmentation

    Args:
        create_n (int):
            Number of augmented sequences.
        aug_n (int):
            Number of augmented words in sequences.
        aug_percent (int):
            Percentage of augmented words in sequences.
        aug_min (int):
            Minimum number of augmented words in sequences.
        aug_max (int):
            Maximum number of augmented words in sequences.
    """

    def __init__(self,
                 create_n,
                 aug_n=None,
                 aug_percent=0.02,
                 aug_min=1,
                 aug_max=10):
        self._DATA = {
            'stop_words':
            ("stopwords.txt", "a4a76df756194777ca18cd788231b474",
             "https://bj.bcebos.com/paddlenlp/data/stopwords.txt"),
            'vocab':
            ("baidu_encyclopedia_w2v_vocab.json",
             "25c2d41aec5a6d328a65c1995d4e4c2e",
             "https://bj.bcebos.com/paddlenlp/data/baidu_encyclopedia_w2v_vocab.json"
             ),
            'word_synonym':
            ("word_synonym.json", "aaa9f864b4af4123bce4bf138a5bfa0d",
             "https://bj.bcebos.com/paddlenlp/data/word_synonym.json"),
            'word_homonym':
            ("word_homonym.json", "a578c04201a697e738f6a1ad555787d5",
             "https://bj.bcebos.com/paddlenlp/data/word_homonym.json")
        }
        self.stop_words = self._get_data('stop_words')
        self.aug_n = aug_n
        self.aug_percent = aug_percent
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.create_n = create_n
        self.vocab = Vocab.from_json(self._load_file('vocab'))
        self.tokenizer = JiebaTokenizer(self.vocab)
        self.loop = 3

    @classmethod
    def clean(cls, sequences):
        '''Clean input sequences'''
        if isinstance(sequences, str):
            return sequences.strip()
        if isinstance(sequences, Iterable):
            return [str(s).strip() if s else s for s in sequences]
        return str(sequences).strip()

    def _load_file(self, mode):
        '''Check and download data'''
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, url = self._DATA[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(url, default_root, data_hash)

        return fullname

    def _get_data(self, mode):
        '''Read data as list '''
        fullname = self._load_file(mode)
        data = []
        if os.path.exists(fullname):
            with open(fullname, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    data.append(line.strip())
            f.close()

        return data

    def _get_aug_n(self, size, size_a=None):
        '''Calculate number of words for data augmentation'''
        if size == 0:
            return 0
        aug_n = self.aug_n or int(math.ceil(self.aug_percent * size))
        if self.aug_min and aug_n < self.aug_min:
            aug_n = self.aug_min
        elif self.aug_max and aug_n > self.aug_max:
            aug_n = self.aug_max
        if size_a is not None:
            aug_n = min(aug_n, int(math.floor(size_a * 0.3)))
        return aug_n

    def _skip_stop_word_tokens(self, seq_tokens):
        '''Skip stopping word indexes'''
        indexes = []
        for i, seq_token in enumerate(seq_tokens):
            if seq_token not in self.stop_words:
                indexes.append(i)
        return indexes

    def augment(self, sequences, num_thread=1):
        '''
        Apply augmentation strategy on input sequences.

            Args:
            sequences (str or list(str)):
                Input sequence or list of input sequences.
            num_thread (int):
                Number of threads
        '''
        sequences = self.clean(sequences)
        # Single Thread
        if num_thread == 1:
            if isinstance(sequences, str):
                return self._augment(sequences)
            else:
                output = []
                for sequence in sequences:
                    output.append(self._augment(sequence))
                return output
        else:
            raise NotImplementedError

    def _augment(self, sequence):
        raise NotImplementedError
