# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
import re
from typing import Iterable

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url

from ..data import JiebaTokenizer, Vocab
from ..utils.env import DATA_HOME


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

    def __init__(self, create_n=1, aug_n=None, aug_percent=0.1, aug_min=1, aug_max=10, vocab="vocab"):
        self._DATA = {
            "stop_words": (
                "stopwords.txt",
                "a4a76df756194777ca18cd788231b474",
                "https://bj.bcebos.com/paddlenlp/data/stopwords.txt",
            ),
            "vocab": (
                "baidu_encyclopedia_w2v_vocab.json",
                "25c2d41aec5a6d328a65c1995d4e4c2e",
                "https://bj.bcebos.com/paddlenlp/data/baidu_encyclopedia_w2v_vocab.json",
            ),
            "test_vocab": (
                "test_vocab.json",
                "1d2fce1c80a4a0ec2e90a136f339ab88",
                "https://bj.bcebos.com/paddlenlp/data/test_vocab.json",
            ),
            "word_synonym": (
                "word_synonym.json",
                "aaa9f864b4af4123bce4bf138a5bfa0d",
                "https://bj.bcebos.com/paddlenlp/data/word_synonym.json",
            ),
            "word_embedding": (
                "word_embedding.json",
                "534aa4ad274def4deff585cefd8ead32",
                "https://bj.bcebos.com/paddlenlp/data/word_embedding.json",
            ),
            "word_homonym": (
                "word_homonym.json",
                "a578c04201a697e738f6a1ad555787d5",
                "https://bj.bcebos.com/paddlenlp/data/word_homonym.json",
            ),
            "char_homonym": (
                "char_homonym.json",
                "dd98d5d5d32a3d3dd45c8f7ca503c7df",
                "https://bj.bcebos.com/paddlenlp/data/char_homonym.json",
            ),
            "char_antonym": (
                "char_antonym.json",
                "f892f5dce06f17d19949ebcbe0ed52b7",
                "https://bj.bcebos.com/paddlenlp/data/char_antonym.json",
            ),
            "word_antonym": (
                "word_antonym.json",
                "cbea11fa99fbe9d07e8185750b37e84a",
                "https://bj.bcebos.com/paddlenlp/data/word_antonym.json",
            ),
        }
        self.stop_words = self._get_data("stop_words")
        self.aug_n = aug_n
        self.aug_percent = aug_percent
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.create_n = create_n
        self.vocab = Vocab.from_json(self._load_file(vocab))
        self.tokenizer = JiebaTokenizer(self.vocab)
        self.loop = 5

    @classmethod
    def clean(cls, sequences):
        """Clean input sequences"""
        if isinstance(sequences, str):
            return sequences.strip()
        if isinstance(sequences, Iterable):
            return [str(s).strip() if s else s for s in sequences]
        return str(sequences).strip()

    def _load_file(self, mode):
        """Check and download data"""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename, data_hash, url = self._DATA[mode]
        fullname = os.path.join(default_root, filename)
        if not os.path.exists(fullname) or (data_hash and not md5file(fullname) == data_hash):
            get_path_from_url(url, default_root, data_hash)

        return fullname

    def _get_data(self, mode):
        """Read data as list"""
        fullname = self._load_file(mode)
        data = []
        if os.path.exists(fullname):
            with open(fullname, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(line.strip())
            f.close()
        else:
            raise ValueError("The {} should exist.".format(fullname))

        return data

    def _get_aug_n(self, size, size_a=None):
        """Calculate number of words for data augmentation"""
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
        """Skip words. We can rewrite function to skip specify words."""
        indexes = []
        for i, seq_token in enumerate(seq_tokens):
            if (
                seq_token not in self.stop_words
                and not seq_token.isdigit()
                and not bool(re.search(r"\d", seq_token))
                and not seq_token.encode("UTF-8").isalpha()
            ):
                indexes.append(i)
        return indexes

    def augment(self, sequences, num_thread=1):
        """
        Apply augmentation strategy on input sequences.

            Args:
            sequences (str or list(str)):
                Input sequence or list of input sequences.
            num_thread (int):
                Number of threads
        """
        sequences = self.clean(sequences)
        # Single Thread
        if num_thread == 1:
            if isinstance(sequences, str):
                return [self._augment(sequences)]
            else:
                output = []
                for sequence in sequences:
                    output.append(self._augment(sequence))
                return output
        else:
            raise NotImplementedError

    def _augment(self, sequence):
        raise NotImplementedError


class FileAugment(object):
    """
    File data augmentation

    Args:
        strategies (List):
            List of augmentation strategies.
    """

    def __init__(self, strategies):
        self.strategies = strategies

    def augment(self, input_file, output_file="aug.txt", separator=None, separator_id=0):
        output_sequences = []
        sequences = []

        input_sequences = self.file_read(input_file)

        if separator:
            for input_sequence in input_sequences:
                sequences.append(input_sequence.split(separator)[separator_id])
        else:
            sequences = input_sequences

        for strategy in self.strategies:
            aug_sequences = strategy.augment(sequences)
            if separator:
                for aug_sequence, input_sequence in zip(aug_sequences, input_sequences):
                    input_items = input_sequence.split(separator)
                    for s in aug_sequence:
                        input_items[separator_id] = s
                        output_sequences.append(separator.join(input_items))
            else:
                for aug_sequence in aug_sequences:
                    output_sequences += aug_sequence

        if output_file:
            self.file_write(output_sequences, output_file)

        return output_sequences

    def file_read(self, input_file):
        input_sequences = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                input_sequences.append(line.strip())
        f.close()
        return input_sequences

    def file_write(self, output_sequences, output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            for output_sequence in output_sequences:
                f.write(output_sequence + "\n")
        f.close()
