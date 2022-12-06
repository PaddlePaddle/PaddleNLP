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
from ..normalizers import NormalizedString
from .. import Token, OffsetType


class StringSplit:
    def __init__(self, nomalized_text: NormalizedString, tokens: List[Token]):
        tokens = [token._token for token in tokens]
        self._string_split = C.pretokenizers.StringSplit(nomalized_text._normalized, tokens)

    @property
    def normalized(self):
        return NormalizedString(self._string_split.normalized)

    @normalized.setter
    def normalized(self, normalized: NormalizedString):
        self._string_split.normalized = normalized._normalized

    @property
    def tokens(self):
        return self._string_split.tokens

    @tokens.setter
    def tokens(self, tokens: List[Token]):
        self._string_split.tokens = [token._token for token in tokens]


class PreTokenizedString:
    def __init__(self, text: str):
        self._pretokenized = C.pretokenizers.PreTokenizedString(text)

    def get_string_split(self, idx: int):
        return self._pretokenized.get_string_split(idx)

    def get_string_splits_size(self):
        return self._pretokenized.get_string_splits_size()

    def get_original_text(self):
        return self._pretokenized.get_original_text()

    def get_splits(self, offset_referential: bool, offset_type: str):
        """
        param offset_referential: "original" or "normalized"
        param offset_type
        """
        return self._pretokenized.get_splits(offset_referential, offset_type)

    def to_encoding(self, word_idx: List[int], type_id: int, offset_type):
        return self._pretokenized.to_encoding(word_idx, type_id, offset_type)


class PreTokenizer(ABC):
    def __call__(self, pretokenized: PreTokenizedString):
        return self._pretokenizer(pretokenized._pretokenized)


class WhitespacePreTokenizer(PreTokenizer):
    def __init__(self):
        self._pretokenizer = C.pretokenizers.WhitespacePreTokenizer()


class BertPreTokenizer(PreTokenizer):
    def __init__(self):
        self._pretokenizer = C.pretokenizers.BertPreTokenizer()


class MetaSpacePreTokenizer(PreTokenizer):
    def __init__(self, replacement: str = "_", add_prefix_space: bool = True):
        self._pretokenizer = C.pretokenizers.MetaSpacePreTokenizer(replacement, add_prefix_space)


class SequencePreTokenizer(PreTokenizer):
    def __init__(self, pretokenizers: List):
        pretokenizers = [pretokenizer._pretokenizer for pretokenizer in pretokenizers]
        self._pretokenizer = C.pretokenizers.SequencePreTokenizer(pretokenizers)


class ByteLevelPreTokenizer(PreTokenizer):
    def __init__(self, add_prefix_space: bool = True, use_regex: bool = True):
        self._pretokenizer = C.pretokenizers.ByteLevelPreTokenizer(add_prefix_space, use_regex)


class SplitPreTokenizer(PreTokenizer):
    def __init__(self, pattern: str, split_mode: int, invert: bool = True):
        self._pretokenizer = C.pretokenizers.SplitPreTokenizer(pattern, C.SplitMode(split_mode), invert)
