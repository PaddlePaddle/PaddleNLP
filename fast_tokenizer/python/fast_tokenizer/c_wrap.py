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

from typing import Dict, List, Tuple, Union

from . import core_tokenizers as C

TextInputSequence = str
PreTokenizedInputSequence = Union[List[str], Tuple[str]]

TextEncodeInput = Union[
    TextInputSequence,
    Tuple[TextInputSequence, TextInputSequence],
    List[TextInputSequence],
]

PreTokenizedEncodeInput = Union[
    PreTokenizedInputSequence,
    Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence],
    List[PreTokenizedInputSequence],
]

InputSequence = Union[TextInputSequence, PreTokenizedInputSequence]

EncodeInput = Union[TextEncodeInput, PreTokenizedEncodeInput]


class OffsetType:
    CHAR = C.OffsetType.CHAR
    BYTE = C.OffsetType.BYTE


class Direction:
    LEFT = C.Direction.LEFT
    RIGHT = C.Direction.RIGHT


class TruncStrategy:
    LONGEST_FIRST = C.TruncStrategy.LONGEST_FIRST
    ONLY_FIRST = C.TruncStrategy.ONLY_FIRST
    ONLY_SECOND = C.TruncStrategy.ONLY_SECOND


class PadStrategy:
    BATCH_LONGEST = C.PadStrategy.BATCH_LONGEST
    FIXED_SIZE = C.PadStrategy.FIXED_SIZE


class SplitMode:
    REMOVED = C.SplitMode.REMOVED
    ISOLATED = C.SplitMode.ISOLATED
    MERGED_WITH_NEXT = C.SplitMode.MERGED_WITH_NEXT
    MERGED_WITH_PREVIOUS = C.SplitMode.MERGED_WITH_PREVIOUS
    CONTIGUOUS = C.SplitMode.CONTIGUOUS


class Token:
    def __init__(self):
        self._token = C.Token()

    @property
    def id(self):
        return self._token.id

    @id.setter
    def id(self, id: int):
        self._token.id = id

    @property
    def value(self):
        return self._token.value

    @value.setter
    def value(self, value: str):
        self._token.value = value

    @property
    def offset(self):
        return self._token.offset

    @offset.setter
    def offset(self, offset: Tuple[int, int]):
        self._token.offset = offset

    def __repr__(self):
        return self._token.__repr__()


class PadMethod:
    def __init__(self):
        self._pad_method = C.PadMethod()

    @property
    def strategy(self):
        return self._pad_method.strategy

    @strategy.setter
    def strategy(self, strategy: str):
        """Set the strategy of PadMethod.
        :param strategy: (str) The strategy of PadMethod, 'batch_longest' and 'fixed_size' are valid
        :return None
        """
        self._pad_method.strategy = getattr(PadStrategy, strategy.upper())

    @property
    def direction(self):
        return self._pad_method.direction

    @direction.setter
    def direction(self, direction: str):
        """Set the direction of PadMethod.
        :param strategy: (str) The direction of PadMethod, 'left' and 'right' are valid
        :return None
        """
        self._pad_method.direction = getattr(Direction, direction.upper())

    @property
    def pad_id(self):
        return self._pad_method.pad_id

    @pad_id.setter
    def pad_id(self, pad_id: int):
        self._pad_method.pad_id = pad_id

    @property
    def pad_token_type_id(self):
        return self._pad_method.pad_token_type_id

    @pad_token_type_id.setter
    def pad_token_type_id(self, pad_token_type_id: int):
        self._pad_method.pad_token_type_id = pad_token_type_id

    @property
    def pad_token(self):
        return self._pad_method.pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str):
        self._pad_method.pad_token = pad_token

    @property
    def pad_len(self):
        return self._pad_method.pad_len

    @pad_len.setter
    def pad_len(self, pad_len: int):
        self._pad_method.pad_len = pad_len

    @property
    def pad_to_multiple_of(self):
        return self._pad_method.pad_to_multiple_of

    @pad_to_multiple_of.setter
    def pad_to_multiple_of(self, pad_to_multiple_of):
        self._pad_method.pad_to_multiple_of = pad_to_multiple_of


class TruncMethod:
    def __init__(self):
        self._trunc_method = C.TruncMethod()

    @property
    def max_len(self):
        return self._trunc_method.max_len

    @max_len.setter
    def max_len(self, max_len: int):
        self._trunc_method.max_len = max_len

    @property
    def strategy(self):
        return self._trunc_method.strategy

    @strategy.setter
    def strategy(self, strategy: str):
        """Set the strategy of TruncMethod.
        :param strategy: (str) The strategy of PadMethod, 'longest_first', 'only_first' and 'only_second' are valid
        :return None
        """
        self._trunc_method.strategy = getattr(TruncStrategy, strategy.upper())

    @property
    def direction(self):
        return self._trunc_method.direction

    @direction.setter
    def direction(self, direction: str):
        """Set the direction of TruncMethod.
        :param strategy: (str) The direction of TruncMethod, 'left' and 'right' are valid
        :return None
        """
        self._trunc_method.direction = getattr(Direction, direction.upper())

    @property
    def stride(self):
        return self._trunc_method.stride

    @stride.setter
    def stride(self, stride: int):
        self._trunc_method.stride = stride


class AddedToken:
    def __init__(self, content="", single_word=False, lstrip=False, rstrip=False, normalized=True):
        self._added_token = C.AddedToken(content, single_word, lstrip, rstrip, normalized)

    @property
    def content(self):
        return self._added_token.content

    @property
    def get_is_special(self):
        return self._added_token.get_is_special

    @property
    def normalized(self):
        return self._added_token.normalized

    @property
    def lstrip(self):
        return self._added_token.lstrip

    @property
    def rstrip(self):
        return self._added_token.rstrip

    @property
    def single_word(self):
        return self._added_token.single_word

    def __eq__(self, other):
        return self._added_token == other._added_token


class Encoding:
    def __init__(
        self,
        ids: List[int],
        type_ids: List[int],
        tokens: List[str],
        words_idx: List[int],
        offsets: List[Tuple[int, int]],
        special_tokens_mask: List[int],
        attention_mask: List[int],
        overflowing: List,
        sequence_ranges: Dict[str, Tuple[int, int]],
    ):
        self._encoding = C.Encoding(
            ids,
            type_ids,
            tokens,
            words_idx,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
            sequence_ranges,
        )

    def __str__(self):
        return str(self._encoding)

    def __repr__(self):
        return self._encoding.__repr__()

    def __len__(self):
        return len(self._encoding)

    @property
    def n_sequences(self):
        return self._encoding.n_sequences

    @property
    def tokens(self):
        return self._encoding.tokens

    @property
    def word_ids(self):
        return self._encoding.word_ids

    @property
    def sequence_ids(self):
        return self._encoding.sequence_ids

    @property
    def ids(self):
        return self._encoding.ids

    @property
    def type_ids(self):
        return self._encoding.type_ids

    @property
    def offsets(self):
        return self._encoding.offsets

    @property
    def special_tokens_mask(self):
        return self._encoding.special_tokens_mask

    @property
    def attention_mask(self):
        return self._encoding.attention_mask

    @property
    def overflowing(self):
        return self._encoding.overflowing

    def set_sequence_ids(self, sequence_id: int):
        return self._encoding.set_sequence_ids(sequence_id)

    def char_to_token(self, char_pos, sequence_index: int = 0):
        return self._encoding.char_to_token(char_pos, sequence_index)

    @staticmethod
    def merge(encodings: List, growing_offsets: bool = True):
        return C.Encoding.merge(encodings, growing_offsets)

    def token_to_chars(self, token_index: int):
        return self._encoding.token_to_chars(token_index)

    def token_to_sequence(self, token_index: int):
        return self._encoding.token_to_sequence(token_index)

    def token_to_word(self, token_index: int):
        return self._encoding.token_to_word(token_index)

    def word_to_chars(self, word_index: int, sequence_index: int = 0):
        return self._encoding.word_to_chars(word_index, sequence_index)

    def word_to_tokens(self, word_index: int, sequence_index: int = 0):
        return self._encoding.word_to_tokens(word_index, sequence_index)

    def truncate(self, max_length: int, stride: int = 0, direction: str = "right"):
        return self._encoding.truncate(max_length, stride, direction)

    def pad(
        self, length: int, direction: str = "right", pad_id: int = 0, pad_type_id: int = 0, pad_token: str = "[PAD]"
    ):
        return self._encoding.pad(length, direction, pad_id, pad_type_id, pad_token)


class Tokenizer:
    def __init__(self, model):
        self._tokenizer = None
        if model is not None:
            self._tokenizer = C.Tokenizer(model._model)

    @property
    def normalizer(self):
        return self._tokenizer.normalizer

    @normalizer.setter
    def normalizer(self, normalizer):
        self._tokenizer.normalizer = normalizer._normalizer

    @property
    def pretokenizer(self):
        return self._tokenizer.pretokenizer

    @pretokenizer.setter
    def pretokenizer(self, pretokenizer):
        self._tokenizer.pretokenizer = pretokenizer._pretokenizer

    @property
    def model(self):
        return self._tokenizer.model

    @model.setter
    def model(self, model):
        self._tokenizer.model = model._model

    @property
    def postprocessor(self):
        return self._tokenizer.postprocessor

    @postprocessor.setter
    def postprocessor(self, postprocessor):
        self._tokenizer.postprocessor = postprocessor._postprocessor

    @property
    def decoder(self):
        return self._tokenizer.decoder

    @decoder.setter
    def decoder(self, decoder):
        self._tokenizer.decoder = decoder._decoder

    @property
    def padding(self):
        return self._tokenizer.padding

    @property
    def truncation(self):
        return self._tokenizer.truncation

    def add_special_tokens(self, tokens: List[str]):
        return self._tokenizer.add_special_tokens(tokens)

    def add_tokens(self, tokens: List[str]):
        return self._tokenizer.add_tokens(tokens)

    def enable_padding(
        self,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: int = None,
        pad_to_multiple_of: int = None,
    ):
        return self._tokenizer.enable_padding(direction, pad_id, pad_type_id, pad_token, length, pad_to_multiple_of)

    def disable_padding(self):
        return self._tokenizer.disable_padding()

    def enable_truncation(
        self, max_length: int, stride: int = 0, strategy: str = "longest_first", direction: str = "right"
    ):
        return self._tokenizer.enable_truncation(max_length, stride, strategy, direction)

    def disable_truncation(self):
        return self._tokenizer.disable_truncation()

    def get_vocab(self, with_added_vocabulary: bool = True):
        return self._tokenizer.get_vocab(with_added_vocabulary)

    def get_vocab_size(self, with_added_vocabulary: bool = True):
        return self._tokenizer.get_vocab_size(with_added_vocabulary)

    def encode(
        self,
        sequence: InputSequence,
        pair: InputSequence = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ):
        return self._tokenizer.encode(sequence, pair, is_pretokenized, add_special_tokens)

    def encode_batch(
        self,
        input: Union[List[EncodeInput], Tuple[EncodeInput]],
        add_special_tokens: bool = True,
        is_pretokenized: bool = False,
    ):
        return self._tokenizer.encode_batch(input, add_special_tokens, is_pretokenized)

    def decode(self, ids: List[int], skip_special_tokens: bool = True):
        return self._tokenizer.decode(ids, skip_special_tokens)

    def decode_batch(self, sequences: List[List[int]], skip_special_tokens: bool = True):
        return self._tokenizer.decode_batch(sequences, skip_special_tokens)

    def id_to_token(self, id: int):
        return self._tokenizer.id_to_token(id)

    def token_to_id(self, token: str):
        return self._tokenizer.token_to_id(token)

    def num_special_tokens_to_add(self, is_pair: bool = True):
        return self._tokenizer.num_special_tokens_to_add(is_pair)

    def save(self, path: str, pretty: bool = True):
        return self._tokenizer.save(path, pretty)

    def to_str(self, pretty: bool = True):
        return self._tokenizer.to_str(pretty)

    @staticmethod
    def from_str(json: str):
        tr = Tokenizer(None)
        tr._tokenizer = C.Tokenizer.from_str(json)
        return tr

    @staticmethod
    def from_file(json: str):
        tr = Tokenizer(None)
        tr._tokenizer = C.Tokenizer.from_file(json)
        return tr


def set_thread_num(thread_num):
    """Set the number of threads for accelerating batch tokenization
    :param thread_num: (int) The number of threads
    :return None
    """
    C.set_thread_num(thread_num)


def get_thread_num():
    """Get the number of tokenization threads
    :return int
    """
    return C.get_thread_num()
