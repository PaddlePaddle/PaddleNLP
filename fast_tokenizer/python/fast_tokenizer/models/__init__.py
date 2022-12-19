# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union, Tuple, List, Dict
from abc import ABC

from .. import C


class Model(ABC):
    def tokenize(self, tokens: List[str]):
        return self._model.tokenize(tokens)

    def token_to_id(self, token: str):
        return self._model.token_to_id(token)

    def id_to_token(self, id: int):
        return self._model.id_to_token(id)

    def get_vocab(self):
        return self._model.get_vocab()

    def get_vocab_size(self):
        return self._model.get_vocab_size()

    def save(self, folder: str, prefix: str = None):
        return self._model.save(folder, prefix)


class WordPiece(Model):
    def __init__(
        self,
        vocab: Dict[str, int],
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
        continuing_subword_prefix: str = "##",
        handle_chinese_chars: bool = True,
    ):
        self._model = None
        if vocab is not None:
            self._model = C.models.WordPiece(
                vocab, unk_token, max_input_chars_per_word, continuing_subword_prefix, handle_chinese_chars
            )

    @staticmethod
    def read_file(vocab: str):
        """Read a vocab.txt file

        :params vocab: (str) The path to a vocab.txt file
        :return: Dict[str, int], The vocabulary as a dict
        """
        return C.models.WordPiece.read_file(vocab)

    @staticmethod
    def from_file(
        vocab: str,
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
        continuing_subword_prefix: str = "continuing_subword_prefix",
    ):
        """Load a WordPiece instance from vocab file.

        :param vocab: (str) The path to a vocab.txt file
        :param unk_token: (str) The unknown token
        :param max_input_chars_per_word: (int) The max number of char when tokenize a word
        :param continuing_subword_prefix: (str) The latter subword prefix.
        :return: An instance of WordPiece.
        """
        wp = WordPiece(None)
        wp._model = C.models.WordPiece.from_file(vocab, unk_token, max_input_chars_per_word, continuing_subword_prefix)
        return wp


class FastWordPiece(Model):
    def __init__(
        self,
        vocab: Dict[str, int],
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
        continuing_subword_prefix: str = "##",
        with_pretokenization: bool = False,
    ):
        self._model = None
        if vocab is not None:
            self._model = C.models.FastWordPiece(
                vocab, unk_token, max_input_chars_per_word, continuing_subword_prefix, with_pretokenization
            )

    @staticmethod
    def read_file(vocab: str):
        """Read a vocab.txt file

        :params vocab: (str) The path to a vocab.txt file
        :return: Dict[str, int], The vocabulary as a dict
        """
        return C.models.FastWordPiece.read_file(vocab)

    @staticmethod
    def from_file(
        vocab: str,
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
        continuing_subword_prefix: str = "continuing_subword_prefix",
    ):
        """Load a FastWordPiece instance from vocab file.

        :param vocab: (str) The path to a vocab.txt file
        :param unk_token: (str) The unknown token
        :param max_input_chars_per_word: (int) The max number of char when tokenize a word
        :param continuing_subword_prefix: (str) The latter subword prefix.
        :return: An instance of FastWordPiece.
        """
        wp = FastWordPiece(None)
        wp._model = C.models.FastWordPiece.from_file(
            vocab, unk_token, max_input_chars_per_word, continuing_subword_prefix
        )
        return wp


class BPE(Model):
    def __init__(
        self,
        vocab: Dict[str, int] = None,
        merges: List[Tuple[str, str]] = None,
        cache_capacity: int = None,
        dropout: float = None,
        unk_token: str = None,
        continuing_subword_prefix: str = None,
        end_of_word_suffix: str = None,
        fuse_unk: bool = None,
    ):
        self._model = C.models.BPE(
            vocab, merges, cache_capacity, dropout, unk_token, continuing_subword_prefix, end_of_word_suffix, fuse_unk
        )

    @staticmethod
    def read_file(vocab: str, merges: str):
        return C.models.BPE.read_file(vocab, merges)

    @staticmethod
    def from_file(vocab: str, merges: str, **kwargs):
        bpe = BPE()
        bpe._model = C.models.BPE.from_file(vocab, merges, **kwargs)
        return bpe


class Unigram(Model):
    def __init__(self, vocab: List[Tuple[str, float]] = None, unk_id: int = None):
        self._model = C.models.Unigram(vocab, unk_id)

    def set_filter_token(self, filter_token: str = ""):
        return self._model.set_filter_token(filter_token)

    def set_split_rule(self, split_rule: str = ""):
        return self._model.set_split_rule(split_rule)
