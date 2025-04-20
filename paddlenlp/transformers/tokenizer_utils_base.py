# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Base classes common to both the slow and the fast tokenization classes: PretrainedTokenizerBase (host all the user
fronting encoding methods) Special token mixing (host the special tokens logic) and BatchEncoding (wrap the dictionary
of output with special method for the Fast tokenizers)
"""

import copy
import inspect
import json
import os
import re
import shutil
import tempfile
import warnings
from collections import UserDict
from collections.abc import Sized
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import isfunction
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import aistudio_sdk
import numpy as np
import paddle
from huggingface_hub import (
    create_repo,
    get_hf_file_metadata,
    hf_hub_url,
    repo_type_and_id_from_hf_id,
    upload_folder,
)
from huggingface_hub.utils import EntryNotFoundError

from ..utils.chat_template_utils import _compile_jinja_template, _render_with_assistant_indices
from ..utils.download import resolve_file_path
from ..utils.env import CHAT_TEMPLATE_CONFIG_NAME, CHAT_TEMPLATE_DIR, CHAT_TEMPLATE_FILE
from ..utils.generic import ExplicitEnum, PaddingStrategy, TensorType, _is_numpy, to_py_obj
from ..utils.import_utils import PROTOBUF_IMPORT_ERROR, is_protobuf_available, is_tokenizers_available
from ..utils.log import logger


def import_protobuf_decode_error(error_message=""):
    if is_protobuf_available():
        from google.protobuf.message import DecodeError

        return DecodeError
    else:
        raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))


if is_tokenizers_available():
    from tokenizers import AddedToken
    from tokenizers import Encoding as EncodingFast
else:

    @dataclass(frozen=False, eq=True)
    class AddedToken:
        """
        AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
        way it should behave.

        The `normalized` will default to `not special` if it is not specified, similarly to the definition in
        `tokenizers`.
        """

        def __init__(
            self, content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None
        ):
            self.content = content
            self.single_word = single_word
            self.lstrip = lstrip
            self.rstrip = rstrip
            self.special = special
            self.normalized = normalized if normalized is not None else not special

        def __getstate__(self):
            return self.__dict__

        def __str__(self):
            return self.content

        def __repr__(self) -> str:
            return f"AddedToken(content={self.content}, single_word={self.single_word}, lstrip={self.lstrip}, rstrip={self.rstrip}, special={self.special}, normalized={self.normalized})"

    @dataclass
    class EncodingFast:
        """This is dummy class because without the `tokenizers` library we don't have these objects anyway"""

        pass


VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

# Slow tokenizers used to be saved in three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PretrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class CharSpan(NamedTuple):
    """
    Character span in the original string.

    Args:
        start (`int`): Index of the first character in the original string.
        end (`int`): Index of the character following the last character in the original string.
    """

    start: int
    end: int


class TokenSpan(NamedTuple):
    """
    Token span in an encoded string (list of tokens).

    Args:
        start (`int`): Index of the first token in the span.
        end (`int`): Index of the token following the last token in the span.
    """

    start: int
    end: int


class BatchEncoding(UserDict):
    """
    Holds the output of the [`PretrainedTokenizerBase.__call__`],
    [`PretrainedTokenizerBase.encode_plus`] and
    [`PretrainedTokenizerBase.batch_encode_plus`] methods (tokens, attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode`/`batch_encode` methods
            ('input_ids', 'attention_mask', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in Paddle/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(data)

        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        self._encodings = encoding

        if n_sequences is None and encoding is not None and len(encoding):
            n_sequences = encoding[0].n_sequences

        self._n_sequences = n_sequences

        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    @property
    def n_sequences(self) -> Optional[int]:
        """
        `Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        [`BatchEncoding`]. Currently can be one of `None` (unknown), `1` (a single sentence) or `2` (a pair of
        sentences)
        """
        return self._n_sequences

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Indicate whether this [`BatchEncoding`] was generated from the result of a [`PretrainedTokenizerFast`]
        or not.
        """
        return self._encodings is not None

    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `tokenizers.Encoding` for batch item with index `key`.

        If the key is a slice, returns the value of the dict associated to `key` ('input_ids', 'attention_mask', etc.)
        with the constraint of slice.
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        elif isinstance(item, slice):
            return {key: self.data[key][item] for key in self.data.keys()}
        else:
            raise KeyError(
                "Invalid key. Only three types of key are available: "
                "(1) string, (2) integers for backend Encoding, and (3) slices for data subsetting."
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        """
        `Optional[List[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns `None` if
        the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        return self._encodings

    def tokens(self, batch_index: int = 0) -> List[str]:
        """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[str]`: The list of tokens at that index.
        """
        if not self._encodings:
            raise ValueError(
                "tokens() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].tokens

    def sequence_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to the id of their original sentences:

            - `None` for special tokens added around or between sequences,
            - `0` for tokens corresponding to words in the first sequence,
            - `1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly
              encoded.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the sequence id corresponding to each token. Special tokens added
            by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding
            sequence.
        """
        if not self._encodings:
            raise ValueError(
                "sequence_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].sequence_ids

    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        if not self._encodings:
            raise ValueError("words() is not available when using Python-based tokenizers")
        warnings.warn(
            "`BatchEncoding.words()` property is deprecated and should be replaced with the identical, "
            "but more self-explanatory `BatchEncoding.word_ids()` property.",
            FutureWarning,
        )
        return self.word_ids(batch_index)

    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        if not self._encodings:
            raise ValueError(
                "word_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].word_ids

    def token_to_sequence(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """
        Get the index of the sequence represented by the given token. In the general use case, this method returns `0`
        for a single sequence or the first sequence of a pair, and `1` for the second sequence of a pair

        Can be called as:

        - `self.token_to_sequence(token_index)` if batch size is 1
        - `self.token_to_sequence(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        if not self._encodings:
            raise ValueError("token_to_sequence() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_sequence(token_index)

    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """
        Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

        Can be called as:

        - `self.token_to_word(token_index)` if batch size is 1
        - `self.token_to_word(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        if not self._encodings:
            raise ValueError("token_to_word() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> Optional[TokenSpan]:
        """
        Get the encoded token span corresponding to a word in a sequence of the batch.

        Token spans are returned as a [`~tokenizer_utils_base.TokenSpan`] with:

        - **start** -- Index of the first token.
        - **end** -- Index of the token following the last token.

        Can be called as:

        - `self.word_to_tokens(word_index, sequence_index: int = 0)` if batch size is 1
        - `self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)` if batch size is greater or equal to
          1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the word in the sequence.
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            ([`~tokenizer_utils_base.TokenSpan`], *optional*): Span of tokens in the encoded sequence. Returns
            `None` if no tokens correspond to the word. This can happen especially when the token is a special token
            that has been used to format the tokenization. For example when we add a class token at the very beginning
            of the tokenization.
        """

        if not self._encodings:
            raise ValueError("word_to_tokens() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        span = self._encodings[batch_index].word_to_tokens(word_index, sequence_index)
        return TokenSpan(*span) if span is not None else None

    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> Optional[CharSpan]:
        """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a [`~tokenizer_utils_base.CharSpan`] with:

        - **start** -- Index of the first character in the original string associated to the token.
        - **end** -- Index of the character following the last character in the original string associated to the
          token.

        Can be called as:

        - `self.token_to_chars(token_index)` if batch size is 1
        - `self.token_to_chars(batch_index, token_index)` if batch size is greater or equal to 1

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token or tokens in
                the sequence.

        Returns:
            [`~tokenizer_utils_base.CharSpan`]: Span of characters in the original string, or None, if the token
            (e.g. <s>, </s>) doesn't correspond to any chars in the origin string.
        """

        if not self._encodings:
            raise ValueError("token_to_chars() is not available when using Python based tokenizers")
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        span_indices = self._encodings[batch_index].token_to_chars(token_index)

        return CharSpan(*span_indices) if span_indices is not None else None

    def char_to_token(
        self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0
    ) -> int:
        """
        Get the index of the token in the encoded output comprising a character in the original string for a sequence
        of the batch.

        Can be called as:

        - `self.char_to_token(char_index)` if batch size is 1
        - `self.char_to_token(batch_index, char_index)` if batch size is greater or equal to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int`: Index of the token, or None if the char index refers to a whitespace only token and whitespace is
                   trimmed with `trim_offsets=True`.
        """

        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(char_index, sequence_index)

    def word_to_chars(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> CharSpan:
        """
        Get the character span in the original string corresponding to given word in a sequence of the batch.

        Character spans are returned as a CharSpan NamedTuple with:

        - start: index of the first character in the original string
        - end: index of the character following the last character in the original string

        Can be called as:

        - `self.word_to_chars(word_index)` if batch size is 1
        - `self.word_to_chars(batch_index, word_index)` if batch size is greater or equal to 1

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            `CharSpan` or `List[CharSpan]`: Span(s) of the associated character or characters in the string. CharSpan
            are NamedTuple with:

                - start: index of the first character associated to the token in the original string
                - end: index of the character following the last character associated to the token in the original
                  string
        """

        if not self._encodings:
            raise ValueError("word_to_chars() is not available when using Python based tokenizers")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index, sequence_index)))

    def char_to_word(self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0) -> int:
        """
        Get the word in the original string corresponding to a character in the original string of a sequence of the
        batch.

        Can be called as:

        - `self.char_to_word(char_index)` if batch size is 1
        - `self.char_to_word(batch_index, char_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the character in the original string.
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the character in the
                original string.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int` or `List[int]`: Index or indices of the associated encoded token(s).
        """

        if not self._encodings:
            raise ValueError("char_to_word() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(char_index, sequence_index)

    def convert_to_tensors(
        self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)
        # Get a function reference for the correct framework
        if tensor_type == TensorType.PADDLE:
            as_tensor = paddle.to_tensor
            is_tensor = paddle.is_tensor
        else:
            as_tensor = np.asarray
            is_tensor = _is_numpy

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                    # # at-least2d
                    # if tensor.ndim > 2:
                    #     tensor = tensor.squeeze(0)
                    # elif tensor.ndim < 2:
                    #     tensor = tensor[None, :]

                    self[key] = tensor
            except Exception as e:
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    ) from e
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding with"
                    " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                    f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                    " expected)."
                ) from e

        return self


class SpecialTokensMixin:
    """
    A mixin derived by [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`] to handle specific behaviors related to
    special tokens. In particular, this class hold the attributes which can be used to directly access these special
    tokens in a model-independent manner and allow to set and update the special tokens.

    Args:
        bos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the beginning of a sentence.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the end of a sentence.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing an out-of-vocabulary token.
        sep_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the class of the input (used by BERT for instance).
        mask_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT).
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional tokens, which will be marked as `special`, meaning that they will be
            skipped when decoding if `skip_special_tokens` is set to `True`.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self, verbose=True, **kwargs):
        # note(guosheng): Since `__init__` might be called multiple times which
        # is hooked before `PretrainedTokenizer` init, we do not set to None as
        # HF to avoid unintentional overriding.
        self._bos_token = getattr(self, "_bos_token", None)
        self._eos_token = getattr(self, "_eos_token", None)
        self._unk_token = getattr(self, "_unk_token", None)
        self._sep_token = getattr(self, "_sep_token", None)
        self._pad_token = getattr(self, "_pad_token", None)
        self._cls_token = getattr(self, "_cls_token", None)
        self._mask_token = getattr(self, "_mask_token", None)
        self._additional_special_tokens = getattr(self, "_additional_special_tokens", [])
        self._pad_token_type_id = getattr(self, "_pad_token_type_id", 0)
        self.verbose = verbose
        self._special_tokens_map = dict.fromkeys(self.SPECIAL_TOKENS_ATTRIBUTES)
        self._special_tokens_map["additional_special_tokens"] = []  # for BC where it defaults to empty list

        # We directly set the hidden value to allow initialization with special tokens
        # which are not yet in the vocabulary. Necessary for serialization/de-serialization
        # TODO clean this up at some point (probably by switching to fast tokenizers)

        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(isinstance(t, (str, AddedToken)) for t in value), (
                        "One of the tokens is not a string or an AddedToken"
                    )
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")

    def sanitize_special_tokens(self) -> int:
        """
        Make sure that all the special tokens attributes of the tokenizer (`tokenizer.mask_token`,
        `tokenizer.cls_token`, etc.) are in the vocabulary.

        Add the missing ones to the vocabulary if needed.

        Return:
            `int`: The number of tokens added in the vocabulary during the operation.
        """
        return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

    def add_special_tokens(
        self, special_tokens_dict: Dict[str, Union[str, AddedToken]], replace_additional_special_tokens=True
    ) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).

        When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
        model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - Special tokens can be skipped when decoding using `skip_special_tokens = True`.
        - Special tokens are carefully handled by the tokenizer (they are never split), similar to `AddedTokens`.
        - You can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This
          makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained models (for instance
        [`BertTokenizer`] `cls_token` is already registered to be :obj*'[CLS]'* and XLM's one is also registered to be
        `'</s>'`).

        Args:
            special_tokens_dict (dictionary *str* to *str* or `tokenizers.AddedToken`):
                Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
                `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
                assign the index of the `unk_token` to them).
            replace_additional_special_tokens (`bool`, *optional*,, defaults to `True`):
                If `True`, the existing list of additional special tokens will be replaced by the list provided in
                `special_tokens_dict`. Otherwise, `self._special_tokens_map["additional_special_tokens"]` is just extended. In the former
                case, the tokens will NOT be removed from the tokenizer's full vocabulary - they are only being flagged
                as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
                `added_tokens_encoder` and `added_tokens_decoder`. This means that the previous
                `additional_special_tokens` are still added tokens, and will not be split by the model.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to add a new classification token to GPT-2
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2Model.from_pretrained("gpt2")

        special_tokens_dict = {"cls_token": "<CLS>"}

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))

        assert tokenizer.cls_token == "<CLS>"
        ```"""
        if not special_tokens_dict:
            return 0

        added_tokens = []
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

            if self.verbose:
                logger.info(f"Assigning {value} to the {key} key of the tokenizer")

            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(isinstance(t, (str, AddedToken)) for t in value), (
                    f"Tokens {value} for key {key} should all be str or AddedToken instances"
                )

                to_add = []
                for token in value:
                    if isinstance(token, str):
                        # for legacy purpose we default to stripping. `test_add_tokens_tokenizer` depends on this
                        token = AddedToken(token, rstrip=False, lstrip=False, normalized=False, special=True)
                    if not replace_additional_special_tokens and str(token) in self.additional_special_tokens:
                        continue
                    to_add.append(token)
                if replace_additional_special_tokens and len(to_add) > 0:
                    setattr(self, key, list(to_add))
                else:
                    self._special_tokens_map["additional_special_tokens"].extend(to_add)
                added_tokens += to_add

            else:
                if not isinstance(value, (str, AddedToken)):
                    raise ValueError(f"Token {value} for key {key} should be a str or an AddedToken instance")
                if isinstance(value, (str)):
                    # for legacy purpose we default to stripping. `False` depends on this
                    value = AddedToken(value, rstrip=False, lstrip=False, normalized=False, special=True)
                if isinstance(value, AddedToken):
                    setattr(self, key, value)
                if value not in added_tokens:
                    added_tokens.append(value)

        # if we are adding tokens that were not part of the vocab, we ought to add them
        added_tokens = self.add_tokens(added_tokens, special_tokens=True)
        return added_tokens

    def add_tokens(
        self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
    ) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary and will be isolated before the tokenization
        algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
        not treated in the same way.

        Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
        of the model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Args:
            new_tokens (`str`, `tokenizers.AddedToken` or a list of *str* or `tokenizers.AddedToken`):
                Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string
                token to let you personalize its behavior: whether this token should only match against a single word,
                whether this token should strip all potential whitespaces on the left side, whether this token should
                strip all potential whitespaces on the right side, etc.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Can be used to specify if the token is a special token. This mostly change the normalization behavior
                (special tokens like CLS or [MASK] are usually not lower-cased for instance).

                See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        return self._pad_token_type_id

    def __setattr__(self, key, value):
        key_without_id = key
        key_is_special_id = key.endswith("_id") or key.endswith("_ids")
        if key_is_special_id:
            key_without_id = key[:-3] if not key.endswith("_ids") else key[:-4]

        if self.__dict__.get("_special_tokens_map", None) is not None and any(
            name in self.__dict__["_special_tokens_map"] for name in [key, key_without_id]
        ):
            if key_is_special_id:
                if value is not None:
                    value = (
                        self.convert_ids_to_tokens(value)
                        if key != "additional_special_tokens"
                        else [self.convert_ids_to_tokens(val) for val in value]
                    )
                key = key_without_id

            if key != "additional_special_tokens" and not isinstance(value, (str, AddedToken)) and value is not None:
                raise ValueError(f"Cannot set a non-string value as the {key}")
            self._special_tokens_map[key] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        key_without_id = key
        key_is_special_id = key.endswith("_id") or key.endswith("_ids")
        if key_is_special_id:
            key_without_id = key[:-3] if not key.endswith("_ids") else key[:-4]

        if self.__dict__.get("_special_tokens_map", None) is not None and any(
            name in self.__dict__["_special_tokens_map"] for name in [key, key_without_id]
        ):
            _special_tokens_map = self.__dict__["_special_tokens_map"]
            if not key_is_special_id:
                if _special_tokens_map[key] is None:
                    if self.verbose:
                        logger.error(f"Using {key}, but it is not set yet.")
                    return None
                value = _special_tokens_map[key]
                return str(value) if key != "additional_special_tokens" else [str(tok) for tok in value]
            else:
                attr_as_tokens = getattr(self, key_without_id)
                return self.convert_tokens_to_ids(attr_as_tokens) if attr_as_tokens is not None else None

        if key not in self.__dict__:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
        else:
            return super().__getattr__(key)

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
        `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Convert potential tokens of `tokenizers.AddedToken` type to string.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        """
        `Dict[str, Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]]]`: A dictionary mapping
        special token class attributes (`cls_token`, `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = self._special_tokens_map[attr]
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        `List[Union[str, tokenizers.AddedToken]]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.), the order has
        nothing to do with the index of each tokens. If you want to know the correct indices, check
        `self.added_tokens_encoder`. We can't create an order anymore as the keys are `AddedTokens` and not `Strings`.

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        all_tokens = []
        seen = set()
        for value in self.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if str(value) not in seen else []
            seen.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    def _set_model_specific_special_tokens(self, special_tokens: List[str]):
        """
        Adds new special tokens to the "SPECIAL_TOKENS_ATTRIBUTES" list which will be part
        of "self.special_tokens" and saved as a special token in tokenizer's config.
        This allows us to dynamically add new model-type specific tokens after initializing the tokenizer.
        For example: if the model tokenizers is multimodal, we can support special image or audio tokens.
        """
        self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())
        for key, value in special_tokens.items():
            if isinstance(value, (str, AddedToken)):
                self._special_tokens_map[key] = value
            else:
                raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")


class PretrainedTokenizerBase(SpecialTokensMixin):
    """
    Base class for [`PretrainedTokenizer`].

    Class attributes (overridden by derived classes)

         - **vocab_files_names** (`Dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
            vocabulary file required by the model, and as associated values, the filename for saving the associated file
            (string).
        - **pretrained_vocab_files_map** (`Dict[str, Dict[str, str]]`) -- A dictionary of dictionaries, with the
            high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
            low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
            associated pretrained vocabulary file.
        - **max_model_input_sizes** (`Dict[str, Optional[int]]`) -- A dictionary with, as keys, the `short-cut-names`
            of the pretrained models, and as associated values, the maximum length of the sequence inputs of this model,
            or `None` if the model has no maximum input size.
        - **pretrained_init_configuration** (`Dict[str, Dict[str, Any]]`) -- A dictionary with, as keys, the
            `short-cut-names` of the pretrained models, and as associated values, a dictionary of specific arguments to
            pass to the `__init__` method of the tokenizer class for this pretrained model when loading the tokenizer
            with the [`~tokenizer_utils_base.PretrainedTokenizerBase.from_pretrained`] method.
        - **model_input_names** (`List[str]`) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
            Should be `'right'` or `'left'`.
        - **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
            applied. Should be `'right'` or `'left'`.

    Args:
        model_max_length (`int`, *optional*):
            The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
            loaded with [`~tokenizer_utils_base.PretrainedTokenizerBase.from_pretrained`], this will be set to the
            value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will
            default to VERY_LARGE_INTEGER (`int(1e30)`).
        padding_side (`str`, *optional*):
            The side on which the model should have padding applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        truncation_side (`str`, *optional*):
            The side on which the model should have truncation applied. Should be selected between ['right', 'left'].
            Default value is picked from the class attribute of the same name.
        chat_template (`str`, *optional*):
            A Jinja template string that will be used to format lists of chat messages. See
            https://huggingface.co/docs/transformers/chat_templating for a full description.
        model_input_names (`List[string]`, *optional*):
            The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
            `"attention_mask"`). Default value is picked from the class attribute of the same name.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the beginning of a sentence. Will be associated to `self.bos_token` and
            `self.bos_token_id`.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the end of a sentence. Will be associated to `self.eos_token` and
            `self.eos_token_id`.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing an out-of-vocabulary token. Will be associated to `self.unk_token` and
            `self.unk_token_id`.
        sep_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token separating two different sentences in the same input (used by BERT for instance). Will be
            associated to `self.sep_token` and `self.sep_token_id`.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation. Will be associated to `self.pad_token` and `self.pad_token_id`.
        cls_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the class of the input (used by BERT for instance). Will be associated to
            `self.cls_token` and `self.cls_token_id`.
        mask_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional special tokens. Add them here to ensure they are skipped when decoding with
            `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end
            of the vocabulary.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cleanup the spaces that were added when splitting the input text during the
            tokenization process.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. Passing will affect the
            internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
            `<s>` is the `bos_token`, then `tokenizer.tokenize("<s>") = ['<s>`]. Otherwise, if
            `split_special_tokens=True`, then `tokenizer.tokenize("<s>")` will be give `['<','s', '>']`.
    """

    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
    resource_files_names: Dict[str, str] = {}
    pretrained_resource_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, Optional[int]] = {}
    _auto_class: Optional[str] = None

    # first name has to correspond to main model input name
    # to make sure `tokenizer.pad(...)` works correctly
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side: str = "right"
    truncation_side: str = "right"
    slow_tokenizer_class = None

    def __init__(self, **kwargs):
        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        for key in kwargs:
            if hasattr(self, key) and callable(getattr(self, key)):
                raise AttributeError(f"{key} conflicts with the method {key} in {self.__class__.__name__}")

        self.init_kwargs = getattr(self, "init_kwargs", None) or copy.deepcopy(kwargs)
        self.name_or_path = kwargs.pop("name_or_path", "")
        self._processor_class = kwargs.pop("processor_class", None)

        # For backward compatibility we fallback to set model_max_length from max_len if provided
        model_max_length = kwargs.pop("model_max_length", kwargs.pop("max_len", None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER

        # Padding and truncation side are right by default and overridden in subclasses. If specified in the kwargs, it
        # is changed.
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        if self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        self.truncation_side = kwargs.pop("truncation_side", self.truncation_side)
        if self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Truncation side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )

        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)

        # By default, cleaning tokenization spaces for both fast and slow tokenizers
        self.clean_up_tokenization_spaces = kwargs.pop("clean_up_tokenization_spaces", False)

        # By default, do not split special tokens for both fast and slow tokenizers
        self.split_special_tokens = kwargs.pop("split_special_tokens", False)

        self.deprecation_warnings = {}  # Use to store when we have already noticed a deprecation warning (avoid overlogging).
        self._in_target_context_manager = False

        # Stores a Jinja template that formats chat histories into tokenizable strings
        self.chat_template = kwargs.pop("chat_template", None)
        if isinstance(self.chat_template, (list, tuple)):
            # Chat templates are stored as lists of dicts with fixed key names,
            # we reconstruct that into a single dict while loading them.
            self.chat_template = {template["name"]: template["template"] for template in self.chat_template}

        super().__init__(**kwargs)

        self.extra_special_tokens = kwargs.pop("extra_special_tokens", {})
        self._set_model_specific_special_tokens(special_tokens=self.extra_special_tokens)

    @property
    def max_len_single_sentence(self) -> int:
        """
        `int`: The maximum length of a sentence that can be fed to the model.
        """
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)

    @property
    def max_len_sentences_pair(self) -> int:
        """
        `int`: The maximum combined length of a pair of sentences that can be fed to the model.
        """
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> int:
        # For backward compatibility, allow to try to setup 'max_len_single_sentence'.
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=False) and self.verbose:
            if not self.deprecation_warnings.get("max_len_single_sentence", False):
                logger.warning(
                    "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
                )
            self.deprecation_warnings["max_len_single_sentence"] = True
        else:
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
            )

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int:
        # For backward compatibility, allow to try to setup 'max_len_sentences_pair'.
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=True) and self.verbose:
            if not self.deprecation_warnings.get("max_len_sentences_pair", False):
                logger.warning(
                    "Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up."
                )
            self.deprecation_warnings["max_len_sentences_pair"] = True
        else:
            raise ValueError("Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up.")

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        added_tokens_decoder_rep = "\n\t".join([f"{k}: {v.__repr__()}," for k, v in self.added_tokens_decoder.items()])
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, model_max_length={self.model_max_length}, is_fast={self.is_fast},"
            f" padding_side='{self.padding_side}', truncation_side='{self.truncation_side}',"
            f" special_tokens={self.special_tokens_map}, clean_up_tokenization_spaces={self.clean_up_tokenization_spaces},"
            " added_tokens_decoder={\n\t" + added_tokens_decoder_rep + "\n}\n)"
        )

    def __len__(self) -> int:
        raise NotImplementedError()

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        vocab.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        raise NotImplementedError()

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        tools: Optional[List[Union[Dict, Callable]]] = None,
        documents: Optional[List[Dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[str, List[int], List[str], List[List[int]], BatchEncoding]:
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to
        determine the format and control tokens to use when converting.

        Args:
            conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            tools (`List[Dict]`, *optional*):
                A list of tools (callable functions) that will be accessible to the model. If the template does not
                support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
                giving the name, description and argument types for the tool. See our
                [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
                for more information.
            documents (`List[Dict[str, str]]`, *optional*):
                A list of dicts representing documents that will be accessible to the model if it is performing RAG
                (retrieval-augmented generation). If the template does not support RAG, this argument will have no
                effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
                see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
                for examples of passing documents with chat templates.
            chat_template (`str`, *optional*):
                A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
                argument, as the model's template will be used by default.
            add_generation_prompt (bool, *optional*):
                If this is set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            continue_final_message (bool, *optional*):
                If this is set, the chat will be formatted so that the final
                message in the chat is open-ended, without any EOS tokens. The model will continue this message
                rather than starting a new one. This allows you to "prefill" part of
                the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
            return_assistant_tokens_mask (`bool`, defaults to `False`):
                Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
                the mask will contain 1. For user and system tokens, the mask will contain 0.
                This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
            **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

        Returns:
            `Union[List[int], Dict]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
            set, will return a dict of tokenizer outputs instead.
        """

        if return_dict and not tokenize:
            raise ValueError(
                "`return_dict=True` is incompatible with `tokenize=False`, because there is no dict "
                "of tokenizer outputs to return."
            )

        if return_assistant_tokens_mask and not return_dict:
            raise ValueError("`return_assistant_tokens_mask=True` is incompatible with `return_dict=False`")

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        chat_template = self.get_chat_template(chat_template, tools)

        if return_assistant_tokens_mask and not re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):
            logger.warning_once(
                "return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword."
            )

        # Compilation function uses a cache to avoid recompiling the same template
        compiled_template = _compile_jinja_template(chat_template)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if return_assistant_tokens_mask:
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        # We accept either JSON schemas or functions for tools. If we get functions, we convert them to schemas
        if tools is not None:
            tool_schemas = []
            for tool in tools:
                if isinstance(tool, dict):
                    tool_schemas.append(tool)
                elif isfunction(tool):
                    tool_schemas.append(get_json_schema(tool))
                else:
                    raise ValueError(
                        "Tools should either be a JSON schema, or a callable function with type hints "
                        "and a docstring suitable for auto-conversion to a schema."
                    )
        else:
            tool_schemas = None

        if documents is not None:
            for document in documents:
                if not isinstance(document, dict):
                    raise TypeError("Documents should be a list of dicts with 'title' and 'text' keys!")

        rendered = []
        all_generation_indices = []
        template_kwargs = {**self.special_tokens_map, **kwargs}  # kwargs overwrite special tokens if both are present
        for chat in conversations:
            if hasattr(chat, "messages"):
                # Indicates it's a Conversation object
                chat = chat.messages
            if return_assistant_tokens_mask:
                rendered_chat, generation_indices = _render_with_assistant_indices(
                    compiled_template=compiled_template,
                    messages=chat,
                    tools=tool_schemas,
                    documents=documents,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
                all_generation_indices.append(generation_indices)
            else:
                rendered_chat = compiled_template.render(
                    messages=chat,
                    tools=tool_schemas,
                    documents=documents,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
            if continue_final_message:
                final_message = chat[-1]["content"]
                if isinstance(final_message, (list, tuple)):
                    for content_block in reversed(final_message):
                        if "text" in content_block:
                            # Pick the last text block in the message (the first one we hit while iterating in reverse)
                            final_message = content_block["text"]
                            break
                    else:
                        raise ValueError(
                            "continue_final_message is set but we could not find any text to continue"
                            "in the final message!"
                        )
                if final_message.strip() not in rendered_chat:
                    raise ValueError(
                        "continue_final_message is set but the final message does not appear in the chat after "
                        "applying the chat template! This can happen if the chat template deletes portions of "
                        "the final message. Please verify the chat template and final message in your chat to "
                        "ensure they are compatible."
                    )
                final_msg_loc = rendered_chat.rindex(final_message.strip())
                if rendered_chat[final_msg_loc : final_msg_loc + len(final_message.lstrip())] == final_message:
                    # The template preserves spacing or the message doesn't have trailing spacing, so things are simple
                    rendered_chat = rendered_chat[: final_msg_loc + len(final_message.lstrip())]
                else:
                    # The message has trailing spacing that was trimmed, so we must be more cautious
                    rendered_chat = rendered_chat[: final_msg_loc + len(final_message.strip())]
            rendered.append(rendered_chat)

        if not is_batched:
            rendered = rendered[0]

        if tokenize:
            out = self(
                rendered,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors=return_tensors,
                **tokenizer_kwargs,
            )
            if return_dict:
                if return_assistant_tokens_mask:
                    assistant_masks = []
                    if is_batched or return_tensors:
                        input_ids = out["input_ids"]
                    else:
                        input_ids = [out["input_ids"]]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        for assistant_start_char, assistant_end_char in all_generation_indices[i]:
                            start_token = out.char_to_token(i, assistant_start_char)
                            end_token = out.char_to_token(i, assistant_end_char - 1)
                            if start_token is None:
                                # start_token is out of bounds maybe due to truncation.
                                break
                            for token_id in range(start_token, end_token + 1 if end_token else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)

                    if not is_batched and not return_tensors:
                        assistant_masks = assistant_masks[0]

                    out["assistant_masks"] = assistant_masks

                    if return_tensors:
                        out.convert_to_tensors(tensor_type=return_tensors)

                return out
            else:
                return out["input_ids"]
        else:
            return rendered

    def get_chat_template(self, chat_template: Optional[str] = None, tools: Optional[List[Dict]] = None) -> str:
        """
        Retrieve the chat template string used for tokenizing chat messages. This template is used
        internally by the `apply_chat_template` method and can also be used externally to retrieve the model's chat
        template for better generation tracking.

        Args:
            chat_template (`str`, *optional*):
                A Jinja template or the name of a template to use for this conversion.
                It is usually not necessary to pass anything to this argument,
                as the model's template will be used by default.
            tools (`List[Dict]`, *optional*):
                A list of tools (callable functions) that will be accessible to the model. If the template does not
                support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
                giving the name, description and argument types for the tool. See our
                [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
                for more information.

        Returns:
            `str`: The chat template string.
        """
        # First, handle the cases when the model has a dict of multiple templates
        if isinstance(self.chat_template, dict):
            template_dict = self.chat_template
            if chat_template is not None and chat_template in template_dict:
                # The user can pass the name of a template to the chat template argument instead of an entire template
                chat_template = template_dict[chat_template]
            elif chat_template is None:
                if tools is not None and "tool_use" in template_dict:
                    chat_template = template_dict["tool_use"]
                elif "default" in template_dict:
                    chat_template = template_dict["default"]
                else:
                    raise ValueError(
                        "This model has multiple chat templates with no default specified! Please either pass a chat "
                        "template or the name of the template you wish to use to the `chat_template` argument. Available "
                        f"template names are {sorted(template_dict.keys())}."
                    )

        elif chat_template is None:
            # These are the cases when the model has a single template
            # priority: `chat_template` argument > `tokenizer.chat_template`
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use chat template functions because tokenizer.chat_template is not set and no template "
                    "argument was passed! For information about writing templates and setting the "
                    "tokenizer.chat_template attribute, please see the documentation at "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating"
                )

        return chat_template

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        **kwargs,
    ):
        """
        Creates an instance of `PretrainedTokenizer`. Related resources are loaded
        by specifying name of a built-in pretrained model, or a community-contributed
        pretrained model, or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains tokenizer related resources
                  and tokenizer config file ("tokenizer_config.json").
            from_hf_hub (bool, optional): whether to load from Huggingface Hub
            subfolder (str, optional) An optional value corresponding to a folder inside the repo.
                Only works when loading from Huggingface Hub.
            *args (tuple): position arguments for model `__init__`. If provided,
                use these as position argument values for tokenizer initialization.
            **kwargs (dict): keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for tokenizer
                initialization.

        Returns:
            PretrainedTokenizer: An instance of `PretrainedTokenizer`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertTokenizer

                # Name of built-in pretrained model
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                tokenizer = BertTokenizer.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned')

                # Load from local directory path
                tokenizer = BertTokenizer.from_pretrained('./my_bert/')
        """
        cache_dir = kwargs.pop("cache_dir", None)
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        from_aistudio = kwargs.pop("from_aistudio", False)
        subfolder = kwargs.pop("subfolder", "")
        return_tokenizer_file_dir = kwargs.pop("return_tokenizer_file_dir", False)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        init_configuration = {}

        is_local = os.path.isdir(pretrained_model_name_or_path)

        additional_files_names = {
            "added_tokens_file": ADDED_TOKENS_FILE,  # kept only for legacy
            "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,  # kept only for legacy
            "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
            # tokenizer_file used to initialize a slow from a fast. Properly copy the `addedTokens` instead of adding in random orders
            "tokenizer_file": FULL_TOKENIZER_FILE,
            "chat_template_file": CHAT_TEMPLATE_CONFIG_NAME,
        }

        if hasattr(cls, "vocab_files_names") and len(cls.resource_files_names) == 0:
            cls.resource_files_names = copy.deepcopy(cls.vocab_files_names)
        vocab_files = {**cls.resource_files_names, **additional_files_names}
        # From HF Hub or AI Studio
        if from_hf_hub or from_aistudio:
            # Only include the necessary resource files specified by the tokenizer cls
            # Deep copy to avoid modifiying the class attributes
            vocab_files = copy.deepcopy(cls.resource_files_names)
            vocab_files["tokenizer_config_file"] = cls.tokenizer_config_file

        # From built-in pretrained models
        elif pretrained_model_name_or_path in cls.pretrained_init_configuration:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(cls.pretrained_init_configuration[pretrained_model_name_or_path])
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            vocab_files["tokenizer_config_file"] = cls.tokenizer_config_file
            for file_id, file_name in vocab_files.items():
                full_file_name = os.path.join(pretrained_model_name_or_path, subfolder, file_name)
                if os.path.isfile(full_file_name):
                    vocab_files[file_id] = full_file_name
        else:
            # Assuming from community-contributed pretrained models
            for file_id, file_name in vocab_files.items():
                vocab_files[file_id] = file_name
        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            if file_path is None or os.path.isfile(file_path):
                resolved_vocab_files[file_id] = file_path
                continue
            resolved_vocab_files[file_id] = resolve_file_path(
                pretrained_model_name_or_path,
                [file_path],
                subfolder,
                cache_dir=cache_dir,
                from_aistudio=from_aistudio,
                from_hf_hub=from_hf_hub,
            )
        for file_id, file_path in resolved_vocab_files.items():
            if resolved_vocab_files[file_id] is not None:
                cache_dir = os.path.dirname(resolved_vocab_files[file_id])
                break
        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            cache_dir=cache_dir,
            return_tokenizer_file_dir=return_tokenizer_file_dir,
            from_hf_hub=from_hf_hub,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        cache_dir=None,
        return_tokenizer_file_dir=False,
        from_hf_hub=False,
        **kwargs,
    ):
        if cls.__name__.endswith("Fast"):
            from_slow = kwargs.get("from_slow", False)
        else:
            from_slow = kwargs.get("from_slow", True)
        has_tokenizer_file = resolved_vocab_files.get("tokenizer_file", None) is not None
        if (from_slow or not has_tokenizer_file) and cls.slow_tokenizer_class is not None:
            slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
                copy.deepcopy(resolved_vocab_files),
                pretrained_model_name_or_path,
                copy.deepcopy(init_configuration),
                *init_inputs,
                cache_dir=cache_dir,
                **(copy.deepcopy(kwargs)),
            )
        else:
            slow_tokenizer = None

        # Prepare tokenizer initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            # First attempt. We get tokenizer_class from tokenizer_config to check mismatch between tokenizers.
            config_tokenizer_class = init_kwargs.get("tokenizer_class")
            init_kwargs.pop("tokenizer_class", None)
            if not has_tokenizer_file:
                init_kwargs.pop("tokenizer_file", None)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            config_tokenizer_class = None
            init_kwargs = init_configuration

        # If independent chat template file(s) exist, they take priority over template entries in the tokenizer config
        chat_templates = {}
        chat_template_file = resolved_vocab_files.pop("chat_template_file", None)
        extra_chat_templates = [key for key in resolved_vocab_files if key.startswith("chat_template_")]
        if chat_template_file is not None:
            with open(chat_template_file) as chat_template_handle:
                chat_templates["default"] = chat_template_handle.read()
        for extra_chat_template in extra_chat_templates:
            template_file = resolved_vocab_files.pop(extra_chat_template, None)
            if template_file is None:
                continue  # I think this should never happen, but just in case
            template_name = extra_chat_template.removeprefix("chat_template_")
            with open(template_file) as chat_template_handle:
                chat_templates[template_name] = chat_template_handle.read()
        if len(chat_templates) == 1 and "default" in chat_templates:
            init_kwargs["chat_template"] = chat_templates["default"]
        elif chat_templates:
            init_kwargs["chat_template"] = chat_templates
        if config_tokenizer_class is not None:
            if cls.__name__.replace("Fast", "") != config_tokenizer_class.replace("Fast", ""):
                logger.warning(
                    "The tokenizer class you load from this checkpoint is not the same type as the class this"
                    " function is called from. It may result in unexpected tokenization. \nThe tokenizer class you"
                    f" load from this checkpoint is '{config_tokenizer_class}'. \nThe class this function is called"
                    f" from is '{cls.__name__}'."
                )

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        # Merge resolved_vocab_files arguments in init_kwargs.
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
        tokenizer_file = resolved_vocab_files.pop("tokenizer_file", None)

        if slow_tokenizer is not None:
            init_kwargs["__slow_tokenizer"] = slow_tokenizer
        init_kwargs["name_or_path"] = pretrained_model_name_or_path
        init_kwargs["from_slow"] = from_slow

        pass_added_tokens_file = False
        # Handle tokenizer serialization of added and special tokens
        added_tokens_decoder: Dict[int, AddedToken] = {}
        added_tokens_map: Dict[str, AddedToken] = {}
        # if we have info on the slow added tokens
        if "added_tokens_decoder" in init_kwargs:
            for idx, token in init_kwargs["added_tokens_decoder"].items():
                if isinstance(token, dict):
                    token = AddedToken(**token)
                if isinstance(token, AddedToken):
                    added_tokens_decoder[int(idx)] = token
                    added_tokens_map[str(token)] = token
                else:
                    raise ValueError(
                        f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
                    )
            init_kwargs["added_tokens_decoder"] = added_tokens_decoder

            pass_added_tokens_file = True

        # position args are stored in kwargs, maybe better not include
        init_kwargs.pop("init_class", None)

        # Update with newly provided args and kwargs
        init_kwargs.update(kwargs)

        def convert_added_tokens(obj):
            if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
                obj.pop("__type")
                return AddedToken(**obj)
            elif isinstance(obj, (list, tuple)):
                return list(convert_added_tokens(o) for o in obj)
            elif isinstance(obj, dict):
                return {k: convert_added_tokens(v) for k, v in obj.items()}
            return obj

        init_kwargs = convert_added_tokens(init_kwargs)
        # Set max length if needed
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            model_max_length = cls.max_model_input_sizes[pretrained_model_name_or_path]
            if model_max_length is not None and isinstance(model_max_length, (int, float)):
                init_kwargs["model_max_length"] = min(init_kwargs.get("model_max_length", int(1e30)), model_max_length)

        # Merge resolved_vocab_files arguments in init_kwargs if not including.
        # Maybe need more ways to load resources.
        for args_name, file_path in resolved_vocab_files.items():
            # when `pretrained_model_name_or_path` is a pretrained model name,
            # use pretrained_init_configuration as `init_kwargs` to init which
            # does not include the vocab file in it, thus add vocab file into
            # args.
            if args_name not in init_kwargs or init_kwargs[args_name] is None:
                init_kwargs[args_name] = file_path
            # when `pretrained_model_name_or_path` is a pretrained model dir,
            # use tokenizer_config_file.json as `init_kwargs` to init which
            # does include a vocab file path in it. However, if the vocab file
            # path included in json does not exist, such as was deleted, to make
            # it still work, use the vocab file under this dir.
            elif not os.path.isfile(init_kwargs[args_name] or "") and os.path.isfile(file_path):
                init_kwargs[args_name] = file_path

        # TODO(zhoushunjie): It's not supportted to load tokenizer.json of hf so far.
        if from_hf_hub and "tokenizer_file" in init_kwargs:
            init_kwargs.pop("tokenizer_file")

        # TODO(guosheng): avoid reduplication of position args and key word args
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except import_protobuf_decode_error():
            logger.info(
                "Unable to load tokenizer model from SPM, loading from TikToken will be attempted instead."
                "(Google protobuf error: Tried to load SPM model with non-SPM vocab file).",
            )
            return False
        except RuntimeError as e:
            if "sentencepiece_processor.cc" in str(e):
                logger.info(
                    "Unable to load tokenizer model from SPM, loading from TikToken will be attempted instead."
                    "(SentencePiece RuntimeError: Tried to load SPM model with non-SPM vocab file).",
                )
            return False

        chat_template = init_kwargs.pop("chat_template", None)
        if chat_template is not None:
            tokenizer.init_chat_template(chat_template)

        special_tokens_map_file = resolved_vocab_files.pop("special_tokens_map_file", None)
        if special_tokens_map_file is not None:
            with open(special_tokens_map_file, encoding="utf-8") as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if key in kwargs and kwargs[key]:
                    # This value has already been redefined by the kwargs
                    # We keep this new value and ignore the one stored in the special_tokens_map_file
                    continue
                if isinstance(value, dict):
                    value = AddedToken(**value)
                elif isinstance(value, list):
                    value = [AddedToken(**token) if isinstance(token, dict) else token for token in value]
                setattr(tokenizer, key, value)
                cls._add_extra_special_tokens(key)

        # Add supplementary tokens.
        special_tokens = tokenizer.all_special_tokens
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        added_tokens_file = None if pass_added_tokens_file else added_tokens_file
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)

            # Sort added tokens by index
            added_tok_encoder_sorted = list(sorted(added_tok_encoder.items(), key=lambda x: x[1]))
            for token, index in added_tok_encoder_sorted:
                if has_tokenizer_file and index != len(tokenizer) and tokenizer.convert_tokens_to_ids(token) != index:
                    # index is the current length of the tokenizer (not in vocabulary)
                    raise ValueError(
                        f"Wrong index found for {token}: should be {tokenizer.convert_tokens_to_ids(token)} but found "
                        f"{index}."
                    )
                elif not has_tokenizer_file and index != len(tokenizer):
                    # Tokenizer slow: added token cannot already be in the vocabulary so its index needs to be the
                    # current length of the tokenizer.
                    raise ValueError(
                        f"Non-consecutive added token '{token}' found. "
                        f"Should have index {len(tokenizer)} but has index {index} in saved vocabulary."
                    )

                tokenizer.add_tokens(token, special_tokens=bool(token in special_tokens))
        # Check all our special tokens are registered as "no split" token (we don't cut them) and are in the vocab
        added_tokens = tokenizer.sanitize_special_tokens()
        if added_tokens:
            logger.info(
                "Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained."
            )
        # save all of related things into default root dir
        if pretrained_model_name_or_path in cls.pretrained_init_configuration:
            # tokenizer.save_pretrained(os.path.join(cache_dir, pretrained_model_name_or_path, subfolder))
            tokenizer.save_pretrained(cache_dir)

        if return_tokenizer_file_dir:
            return tokenizer, list(tokenizer_config_file_dir_list)[0]
        return tokenizer

    @classmethod
    def convert_added_tokens(cls, obj: Union[AddedToken, Any], save=False, add_type_field=True):
        if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
            obj.pop("__type")
            return AddedToken(**obj)
        if isinstance(obj, AddedToken) and save:
            obj = obj.__getstate__()
            if add_type_field:
                obj["__type"] = "AddedToken"
            else:
                # Don't save "special" for previous tokenizers
                obj.pop("special")
            return obj
        elif isinstance(obj, (list, tuple)):
            return [cls.convert_added_tokens(o, save=save, add_type_field=add_type_field) for o in obj]
        elif isinstance(obj, dict):
            return {k: cls.convert_added_tokens(v, save=save, add_type_field=add_type_field) for k, v in obj.items()}
        return obj

    def save_chat_templates(
        self,
        save_directory: Union[str, os.PathLike],
        tokenizer_config: dict,
        filename_prefix: Optional[str],
        save_jinja_files: bool,
    ):
        """
        Writes chat templates out to the save directory if we're using the new format, and removes them from
        the tokenizer config if present. If we're using the legacy format, it doesn't write any files, and instead
        writes the templates to the tokenizer config in the correct format.
        """
        chat_template_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + CHAT_TEMPLATE_FILE
        )
        chat_template_dir = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + CHAT_TEMPLATE_DIR
        )

        saved_raw_chat_template_files = []
        if save_jinja_files and isinstance(self.chat_template, str):
            # New format for single templates is to save them as chat_template.jinja
            with open(chat_template_file, "w", encoding="utf-8") as f:
                f.write(self.chat_template)
            logger.info(f"chat template saved in {chat_template_file}")
            saved_raw_chat_template_files.append(chat_template_file)
            if "chat_template" in tokenizer_config:
                tokenizer_config.pop("chat_template")  # To ensure it doesn't somehow end up in the config too
        elif save_jinja_files and isinstance(self.chat_template, dict):
            # New format for multiple templates is to save the default as chat_template.jinja
            # and the other templates in the chat_templates/ directory
            for template_name, template in self.chat_template.items():
                if template_name == "default":
                    with open(chat_template_file, "w", encoding="utf-8") as f:
                        f.write(self.chat_template["default"])
                    logger.info(f"chat template saved in {chat_template_file}")
                    saved_raw_chat_template_files.append(chat_template_file)
                else:
                    Path(chat_template_dir).mkdir(exist_ok=True)
                    template_filepath = os.path.join(chat_template_dir, f"{template_name}.jinja")
                    with open(template_filepath, "w", encoding="utf-8") as f:
                        f.write(template)
                    logger.info(f"chat template saved in {template_filepath}")
                    saved_raw_chat_template_files.append(template_filepath)
            if "chat_template" in tokenizer_config:
                tokenizer_config.pop("chat_template")  # To ensure it doesn't somehow end up in the config too
        elif isinstance(self.chat_template, dict):
            # Legacy format for multiple templates:
            # chat template dicts are saved to the config as lists of dicts with fixed key names.
            tokenizer_config["chat_template"] = [{"name": k, "template": v} for k, v in self.chat_template.items()]
        elif self.chat_template is not None:
            # Legacy format for single templates: Just make them a key in tokenizer_config.json
            tokenizer_config["chat_template"] = self.chat_template
        return tokenizer_config, saved_raw_chat_template_files

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str]:
        """
        Save tokenizer configuration and related resources to files under
        `save_directory`. The tokenizer configuration would be saved into
        `tokenizer_config_file` indicating file (thus `tokenizer_config.json`),
        and resources would be saved into `resource_files_names` indicating files
        by using `self.save_resources(save_directory)`.

        The `save_directory` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the tokenizer.

        Args:
            save_directory (str): Directory to save files into.
            filename_prefix: (str, optional):
                A prefix to add to the names of the files saved by the tokenizer.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                tokenizer.save_pretrained('trained_model')
                # reload from save_directory
                tokenizer = BertTokenizer.from_pretrained('trained_model')
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        special_tokens_map_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + SPECIAL_TOKENS_MAP_FILE
        )
        tokenizer_config_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE
        )

        tokenizer_config = copy.deepcopy(self.init_kwargs)

        # Let's save the init kwargs
        target_keys = set(self.init_kwargs.keys())
        # Let's save the special tokens map (only the strings)
        target_keys.update(["model_max_length", "clean_up_tokenization_spaces"])

        for k in target_keys:
            if hasattr(self, k):
                tokenizer_config[k] = getattr(self, k)

        # Let's make sure we properly save the special tokens
        tokenizer_config.update(self.special_tokens_map)
        if "extra_special_tokens" not in tokenizer_config:
            tokenizer_config["extra_special_tokens"] = self.extra_special_tokens
            tokenizer_config.update(self.extra_special_tokens)

        save_jinja_files = kwargs.get("save_jinja_files", True)
        tokenizer_config, saved_raw_chat_template_files = self.save_chat_templates(
            save_directory, tokenizer_config, filename_prefix, save_jinja_files
        )

        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.resource_files_names.keys():
            tokenizer_config.pop(file_id, None)

        # no typefields, this way old fast and slow can load it
        tokenizer_config = self.convert_added_tokens(tokenizer_config, add_type_field=True, save=True)

        # Process added tokens separately: allows previous versions to ignore it!
        added_tokens = {}
        for key, value in self.added_tokens_decoder.items():
            added_tokens[key] = value.__getstate__()
        tokenizer_config["added_tokens_decoder"] = added_tokens

        # Add tokenizer class to the tokenizer config to be able to reload it with from_pretrained
        tokenizer_class = self.__class__.__name__
        # Remove the Fast at the end if we can save the slow tokenizer
        if tokenizer_class.endswith("Fast") and getattr(self, "can_save_slow_tokenizer", False):
            tokenizer_class = tokenizer_class[:-4]
        tokenizer_config["tokenizer_class"] = tokenizer_class
        if getattr(self, "_auto_map", None) is not None:
            tokenizer_config["auto_map"] = self._auto_map
        if getattr(self, "_processor_class", None) is not None:
            tokenizer_config["processor_class"] = self._processor_class

        # remove private information
        if "name_or_path" in tokenizer_config:
            tokenizer_config.pop("name_or_path")
            tokenizer_config.pop("special_tokens_map_file", None)
            tokenizer_config.pop("tokenizer_file", None)
        if "device_map" in tokenizer_config:
            tokenizer_config.pop("device_map")

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)
        logger.info(f"tokenizer config file saved in {tokenizer_config_file}")

        # Sanitize AddedTokens in special_tokens_map

        # kept for forward compatibility, will be removed in transoformers 5. Typefields are not saved for FC, special should not be save either
        write_dict = self.convert_added_tokens(self.special_tokens_map_extended, save=True, add_type_field=False)
        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(write_dict, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)
        logger.info(f"Special tokens file saved in {special_tokens_map_file}")

        file_names = (tokenizer_config_file, special_tokens_map_file, *saved_raw_chat_template_files)

        save_files = self._save_pretrained(
            save_directory=save_directory,
            file_names=file_names,
            filename_prefix=filename_prefix,
        )

        return save_files

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.

        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens} using the
        specific [`~tokenization_utils_fast.PreTrainedTokenizerFast._save_pretrained`]
        """
        if legacy_format is False:
            raise ValueError(
                "Only fast tokenizers (instances of PreTrainedTokenizerFast) can be saved in non legacy format."
            )

        save_directory = str(save_directory)

        added_tokens_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
        )
        # the new get_added_vocab() also returns special tokens and tokens that have an index < vocab_size
        added_vocab = {tok: index for tok, index in self.added_tokens_encoder.items() if index >= self.vocab_size}
        if added_vocab:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                f.write(out_str)
                logger.info(f"added tokens file saved in {added_tokens_file}")

        vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)

        return file_names + vocab_files + (added_tokens_file,)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        [`~PreTrainedTokenizerFast._save_pretrained`] to save the whole state of the tokenizer.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        try:
            return self.save_resources(save_directory=save_directory)
        except Exception:
            raise NotImplementedError

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to `resource_files_names` indicating
        files under `save_directory` by copying directly. Override it if necessary.

        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            src_path = self.init_kwargs[name]
            dst_path = os.path.join(save_directory, file_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copyfile(src_path, dst_path)

    def save_to_hf_hub(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        subfolder: Optional[str] = None,
        commit_message: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all elements of this tokenizer to a new HuggingFace Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
            commit_message (str, optional) — The summary / title / first line of the generated commit. Defaults to: f"Upload {path_in_repo} with huggingface_hub"
            revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
            create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False.
                If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch.
                If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.

        Returns: The url of the commit of your model in the given repository.
        """
        repo_url = create_repo(repo_id, private=private, exist_ok=True)

        # Infer complete repo_id from repo_url
        # Can be different from the input `repo_id` if repo_owner was implicit
        _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
        repo_id = f"{repo_owner}/{repo_name}"

        # Check if README file already exist in repo
        try:
            get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
            has_readme = True
        except EntryNotFoundError:
            has_readme = False

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir
            # save model
            self.save_pretrained(save_dir)
            # Add readme if does not exist
            logger.info("README.md not found, adding the default README.md")
            if not has_readme:
                with open(os.path.join(root_dir, "README.md"), "w") as f:
                    f.write(f"---\nlibrary_name: paddlenlp\n---\n# {repo_id}")
            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=root_dir,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
            )

    def save_to_aistudio(
        self, repo_id, private=True, license="Apache License 2.0", exist_ok=True, subfolder=None, **kwargs
    ):
        """
        Uploads all elements of this model to a new AiStudio Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            token (str): Your token for the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private. Defaults to True.
            license (str): The license of your model/tokenizer. Defaults to: "Apache License 2.0".
            exist_ok (bool, optional): Whether to override existing repository. Defaults to: True.
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
        """

        res = aistudio_sdk.hub.create_repo(repo_id=repo_id, private=private, license=license, **kwargs)
        if "error_code" in res:
            if res["error_code"] == 10003 and exist_ok:
                logger.info(
                    f"Repo {repo_id} already exists, it will override files with the same name. To avoid this, please set exist_ok=False"
                )
            else:
                logger.error(
                    f"Failed to create repo {repo_id}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                )
        else:
            logger.info(f"Successfully created repo {repo_id}")

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir
            # save model
            self.save_pretrained(save_dir)

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            for filename in os.listdir(save_dir):
                res = aistudio_sdk.hub.upload(
                    repo_id=repo_id, path_or_fileobj=os.path.join(save_dir, filename), path_in_repo=filename, **kwargs
                )
                if "error_code" in res:
                    logger.error(
                        f"Failed to upload {filename}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                    )
                else:
                    logger.info(f"{filename}: {res['message']}")

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        """
        Converts a string into a sequence of tokens, replacing unknown tokens with the `unk_token`.

        Args:
            text (`str`):
                The sequence to be encoded.
            pair (`str`, *optional*):
                A second sequence to be encoded with the first.
            add_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to add the special tokens associated with the corresponding model.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method. See details in
                [`~PretrainedTokenizerBase.__call__`]

        Returns:
            `List[str]`: The list of tokens.
        """
        raise NotImplementedError

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            padding_side=padding_side,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        raise NotImplementedError

    def _get_padding_truncation_strategies(
        self, padding=False, truncation=None, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy", "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is None:
            if verbose:
                if not self.deprecation_warnings.get("Truncation-not-explicitly-activated", False):
                    logger.warning(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, please"
                        " use `truncation=True` to explicitly truncate examples to max length. Defaulting to"
                        " 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the"
                        " tokenizer you can select this strategy more precisely by providing a specific strategy to"
                        " `truncation`."
                    )
                self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning,
                )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (
                        truncation is None or truncation is False or truncation == "do_not_truncate"
                    ):
                        warnings.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`."
                        )
                    if old_pad_to_max_length is not False:
                        warnings.warn("Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`.")
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is None and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, use"
                    " `truncation=True` to truncate examples to a max length. You can give a specific length with"
                    " `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input"
                    " size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific"
                    " truncation strategy selected among `truncation='only_first'` (will only truncate the first"
                    " sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the"
                    " pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence"
                    " in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False and truncation is not None:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-pad-to-max_length", False):
                            logger.warning(
                                "Asking to pad to max_length but no maximum length is provided and the model has no"
                                " predefined maximum length. Default to no padding."
                            )
                        self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get("Asking-to-truncate-to-max_length", False):
                            logger.warning(
                                "Asking to truncate to max_length but no maximum length is provided and the model has"
                                " no predefined maximum length. Default to no truncation."
                            )
                        self.deprecation_warnings["Asking-to-truncate-to-max_length"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.pad_token is None or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                "Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is allowed. `self.encode()` or `self.batch_encode()` would be called
        separately for single or batch input depending on input format and
        `is_split_into_words` argument.

        Args:
            text (str, List[str] or List[List[str]]):
                The sequence or batch of sequences to be processed. One sequence
                is a string or a list of strings depending on whether it has been
                pretokenized. If each sequence is provided as a list of strings
                (pretokenized), you must set `is_split_into_words` as `True` to
                disambiguate with a batch of sequences.
            text_pair (str, List[str] or List[List[str]], optional):
                Same as `text` argument, while it represents for the latter
                sequence of the sequence pair.
            max_length (int, optional):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length. If there are overflowing tokens,
                those overflowing tokens will be added to the returned dictionary
                when `return_overflowing_tokens` is `True`. Defaults to `None`.
            stride (int, optional):
                Only available for batch input of sequence pair and mainly for
                question answering usage. When for QA, `text` represents questions
                and `text_pair` represents contexts. If `stride` is set to a
                positive number, the context will be split into multiple spans
                where `stride` defines the number of (tokenized) tokens to skip
                from the start of one span to get the next span, thus will produce
                a bigger batch than inputs to include all spans. Moreover, 'overflow_to_sample'
                and 'offset_mapping' preserving the original example and position
                information will be added to the returned dictionary. Defaults to 0.
            is_split_into_words (Union[bool, str], optional):
                when the text is words or tokens, `is_split_into_words` should be True or `token`.
                `True`: means that the text should be words which should be tokenized.
                `token`: means that the text should be tokens which already be tokenized, so it should not be tokenized again.
            padding (bool, str or [PaddingStrategy], optional):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
                Defaults to `False`.
            truncation (bool, str or [TruncationStrategy], optional):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
                Defaults to `False`.
            return_position_ids (bool, optional):
                Whether to include tokens position ids in the returned dictionary.
                Defaults to `False`.
            return_token_type_ids (bool, optional):
                Whether to include token type ids in the returned dictionary.
                Defaults to `True`.
            return_attention_mask (bool, optional):
                Whether to include the attention mask in the returned dictionary.
                Defaults to `False`.
            return_length (bool, optional):
                Whether to include the length of each encoded inputs in the
                returned dictionary. Defaults to `False`.
            return_overflowing_tokens (bool, optional):
                Whether to include overflowing token information in the returned
                dictionary. Defaults to `False`.
            return_special_tokens_mask (bool, optional):
                Whether to include special tokens mask information in the returned
                dictionary. Defaults to `False`.
            return_dict (bool, optional):
                Decide the format for returned encoded batch inputs. Only works when
                input is a batch of data.
                ::
                    - If True, encoded inputs would be a dictionary like:
                        {'input_ids': [[1, 4444, 4385, 1545, 6712],[1, 4444, 4385]],
                        'token_type_ids': [[0, 0, 0, 0, 0], [0, 0, 0]]}
                    - If False, encoded inputs would be a list like:
                        [{'input_ids': [1, 4444, 4385, 1545, 6712],
                          'token_type_ids': [0, 0, 0, 0, 0]},
                         {'input_ids': [1, 4444, 4385], 'token_type_ids': [0, 0, 0]}]

                Defaults to `True`.
            return_offsets_mapping (bool, optional):
                Whether to include the list of pair preserving the index of start
                and end char in original input for each token in the returned
                dictionary. Would be automatically set to `True` when `stride` > 0.
                Defaults to `False`.
            add_special_tokens (bool, optional):
                Whether to add the special tokens associated with the corresponding model
                to the encoded inputs. Defaults to `True`
            pad_to_multiple_of (int, optional):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                Defaults to `None`.
            padding_side (`str`, *optional*):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_tensors (str or [TensorType], optional):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pd'`: Return Paddle `paddle.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
                Defaults to `None`.
            verbose (bool, optional):
                Whether or not to print more information and warnings. Defaults to True.

        Returns:
            dict or list[dict] (for batch input):
                The dict has the following optional items:

                - **input_ids** (list[int] or list[list[int]]): List of token ids to be fed to a model.
                - **position_ids** (list[int] or list[list[int]], optional): List of token position ids to be
                  fed to a model. Included when `return_position_ids` is `True`
                - **token_type_ids** (list[int] or list[list[int]], optional): List of token type ids to be
                  fed to a model. Included when `return_token_type_ids` is `True`.
                - **attention_mask** (list[int] or list[list[int]], optional): List of integers valued 0 or 1,
                  where 0 specifies paddings and should not be attended to by the
                  model. Included when `return_attention_mask` is `True`.
                - **seq_len** (int or list[int], optional): The input_ids length. Included when `return_length`
                  is `True`.
                - **overflowing_tokens** (list[int] or list[list[int]], optional): List of overflowing tokens.
                  Included when if `max_length` is specified and `return_overflowing_tokens`
                  is True.
                - **num_truncated_tokens** (int or list[int], optional): The number of overflowing tokens.
                  Included when if `max_length` is specified and `return_overflowing_tokens`
                  is True.
                - **special_tokens_mask** (list[int] or list[list[int]], optional): List of integers valued 0 or 1,
                  with 0 specifying special added tokens and 1 specifying sequence tokens.
                  Included when `return_special_tokens_mask` is `True`.
                - **offset_mapping** (list[int], optional): list of pair preserving the
                  index of start and end char in original input for each token.
                  For a sqecial token, the index pair is `(0, 0)`. Included when
                  `return_overflowing_tokens` is True or `stride` > 0.
                - **overflow_to_sample** (int or list[int], optional): Index of example from which this
                  feature is generated. Included when `stride` works.
        """
        # To avoid duplicating
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "padding_side": padding_side,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "split_special_tokens": kwargs.pop("split_special_tokens", self.split_special_tokens),
            "verbose": verbose,
        }
        all_kwargs.update(kwargs)
        if text is None and text_target is None:
            raise ValueError("You need to specify either `text` or `text_target`.")
        if text is not None:
            # The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
            # input mode in this case.
            if not self._in_target_context_manager:
                self._switch_to_input_mode()
            encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
        if text_target is not None:
            self._switch_to_target_mode()
            target_encodings = self._call_one(text=text_target, text_pair=text_pair_target, **all_kwargs)
        # Leave back tokenizer in input mode
        self._switch_to_input_mode()

        if text_target is None:
            return encodings
        elif text is None:
            return target_encodings
        else:
            encodings["labels"] = target_encodings["input_ids"]
            return encodings

    def _call_one(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        return_position_ids: bool = False,
        return_dict: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                "text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        # check `split_into_words` value
        if isinstance(is_split_into_words, str) and is_split_into_words != "token":
            raise ValueError(
                "the value of `is_split_into_words` should be one of: {True, False, 'token'} but receive: <%s>",
                is_split_into_words,
            )

        if is_split_into_words:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple))

        if is_batched:
            if isinstance(text_pair, str):
                raise TypeError(
                    "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as"
                    " `text`."
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`:"
                    f" {len(text_pair)}."
                )
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                return_position_ids=return_position_ids,
                return_dict=return_dict,
                verbose=verbose,
                split_special_tokens=split_special_tokens,
                **kwargs,
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                return_position_ids=return_position_ids,
                return_dict=return_dict,
                verbose=verbose,
                split_special_tokens=split_special_tokens,
                **kwargs,
            )

    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            text (`str`, `List[str]` or (for non-fast tokenizers) `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            split_special_tokens=kwargs.pop("split_special_tokens", self.split_special_tokens),
            **kwargs,
        )

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        raise NotImplementedError

    def batch_encode(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        return_position_ids=None,
        return_dict=True,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in `encode_plus`).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            return_position_ids=return_position_ids,
            return_dict=return_dict,
            verbose=verbose,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        return_position_ids: Optional[bool] = None,
        return_dict: bool = True,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        raise NotImplementedError

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`).

        Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
        text followed by a call to the `pad` method to get a padded encoding.

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, Paddle tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`.
        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a Paddle Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, Paddle tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            padding_side (`str`, *optional*):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pd'`: Return Paddle `paddle.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has been passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have Paddle/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if isinstance(first_element, paddle.Tensor):
                return_tensors = "pd" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be either python or paddle object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            # some tokenizers might not have the padding_side attribute
            if "padding_side" in set(inspect.signature(self._pad).parameters.keys()):
                encoded_inputs = self._pad(
                    encoded_inputs,
                    max_length=max_length,
                    padding_strategy=padding_strategy,
                    pad_to_multiple_of=pad_to_multiple_of,
                    padding_side=padding_side,
                    return_attention_mask=return_attention_mask,
                )
            else:
                padding_side = padding_side if padding_side is not None else self.padding_side

                original_padding_side = self.padding_side
                self.padding_side = padding_side
                encoded_inputs = self._pad(
                    encoded_inputs,
                    max_length=max_length,
                    padding_strategy=padding_strategy,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_attention_mask=return_attention_mask,
                )
                self.padding_side = original_padding_side

            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(len(v) == batch_size for v in encoded_inputs.values()), (
            "Some items in the output dictionary have a different batch size than others."
        )

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            if "padding_side" in set(inspect.signature(self._pad).parameters.keys()):
                outputs = self._pad(
                    inputs,
                    max_length=max_length,
                    padding_strategy=padding_strategy,
                    padding_side=padding_side,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_attention_mask=return_attention_mask,
                )
            else:
                padding_side = padding_side if padding_side is not None else self.padding_side

                original_padding_side = self.padding_side
                self.padding_side = padding_side
                outputs = self._pad(
                    inputs,
                    max_length=max_length,
                    padding_strategy=padding_strategy,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_attention_mask=return_attention_mask,
                )
                self.padding_side = original_padding_side

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        return_position_ids=None,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_position_ids is None:
            return_position_ids = "position_ids" in self.model_input_names
        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        if return_position_ids:
            encoded_inputs["position_ids"] = list(range(len(encoded_inputs["input_ids"])))

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])
            # for compatibility
            encoded_inputs["seq_len"] = encoded_inputs["length"]

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (`int`, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `'longest_first'`):
                The strategy to follow for truncation. Can be:

                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will truncate
                  token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
                  batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
                  than the model maximum admissible input size).
            stride (`int`, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of
            overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair
            of sequences (or a batch of pairs) is provided.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
            truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None
        ):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == "left":
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                elif self.truncation_side == "right":
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                else:
                    raise ValueError(f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'.")

            else:
                error_msg = (
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the first sequence has a length {len(ids)}. "
                )
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                        error_msg + "Please select another truncation strategy than "
                        f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            logger.warning(
                "Be aware, overflowing tokens are not returned for the setting you have chosen,"
                f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
                "truncation strategy. So the returned list will always be empty even if some "
                "tokens have been removed."
            )
            len_pair_ids = len(pair_ids) if pair_ids is not None else 0
            len_ids = len(ids)
            first_remove = min(abs(len_pair_ids - len_ids), num_tokens_to_remove)
            second_remove = num_tokens_to_remove - first_remove
            if len_ids > len_pair_ids:
                ids_to_move = first_remove + second_remove // 2
                pair_ids_to_move = second_remove - second_remove // 2
            else:
                ids_to_move = second_remove // 2
                pair_ids_to_move = first_remove + second_remove - (second_remove // 2)

            if self.truncation_side == "right":
                ids = ids[:-ids_to_move] if ids_to_move > 0 else ids
                pair_ids = pair_ids[:-pair_ids_to_move] if pair_ids is not None and pair_ids_to_move > 0 else pair_ids
            elif self.truncation_side == "left":
                ids = ids[ids_to_move:]
                pair_ids = pair_ids[pair_ids_to_move:] if pair_ids is not None else None
            else:
                raise ValueError(f"invalid truncation strategy:{self.truncation_side}")

        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == "right":
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                elif self.truncation_side == "left":
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                else:
                    raise ValueError(f"invalid truncation strategy:{self.truncation_side}")
            else:
                logger.error(
                    f"We need to remove {num_tokens_to_remove} to truncate the input "
                    f"but the second sequence has a length {len(pair_ids)}. "
                    f"Please select another truncation strategy than {truncation_strategy}, "
                    "for instance 'longest_first' or 'only_first'."
                )

        return (ids, pair_ids, overflowing_tokens)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in `padding_side` argument:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_side = padding_side if padding_side is not None else self.padding_side

            if padding_side == "right":
                if return_attention_mask:
                    if len(np.shape(encoded_inputs["attention_mask"])) > 2:
                        # attention_mask shape [1,seq_len,seq_len]
                        encoded_inputs["attention_mask"] = np.pad(
                            encoded_inputs["attention_mask"],
                            pad_width=[(0, 0), (0, difference), (0, difference)],
                            mode="constant",
                            constant_values=0,
                        ).tolist()
                    else:
                        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "attn_mask_startend_row_indices" in encoded_inputs:
                    # TODO @DrownFish19 encoded_inputs["attn_mask_startend_row_indices"] is generated in the shape [seq_len]
                    # and convert the shape to [1,seq_len] here. However, it is supported in the generation phase.
                    encoded_inputs["attn_mask_startend_row_indices"] = np.concatenate(
                        [
                            np.array([encoded_inputs["attn_mask_startend_row_indices"]], dtype=np.int32),
                            np.zeros([1, difference], dtype=np.int32),
                        ],
                        axis=-1,
                    )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                if "offset_mapping" in encoded_inputs:
                    encoded_inputs["offset_mapping"] = encoded_inputs["offset_mapping"] + [(0, 0)] * difference
                if "position_ids" in encoded_inputs:
                    encoded_inputs["position_ids"] = encoded_inputs["position_ids"] + [0] * difference
                # NOTE: In ernie3.0-qa, the type of `*_positions` is int.
                if "start_positions" in encoded_inputs and isinstance(encoded_inputs["start_positions"], list):
                    encoded_inputs["start_positions"] = encoded_inputs["start_positions"] + [0] * difference
                if "end_positions" in encoded_inputs and isinstance(encoded_inputs["end_positions"], list):
                    encoded_inputs["end_positions"] = encoded_inputs["end_positions"] + [0] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    if len(np.shape(encoded_inputs["attention_mask"])) > 2:
                        # attention_mask shape [1,seq_len,seq_len]
                        encoded_inputs["attention_mask"] = np.pad(
                            encoded_inputs["attention_mask"],
                            pad_width=[(0, 0), (difference, 0), (difference, 0)],
                            mode="constant",
                            constant_values=0,
                        ).tolist()
                    else:
                        encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "attn_mask_startend_row_indices" in encoded_inputs:
                    # TODO @DrownFish19 encoded_inputs["attn_mask_startend_row_indices"] is generated in the shape [seq_len]
                    # and convert the shape to [1,seq_len] here. However, it is supported in the generation phase.
                    encoded_inputs["attn_mask_startend_row_indices"] = np.concatenate(
                        [
                            np.zeros([1, difference], dtype=np.int32),
                            np.array([encoded_inputs["attn_mask_startend_row_indices"]], dtype=np.int32) + difference,
                        ],
                        axis=-1,
                    )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                if "offset_mapping" in encoded_inputs:
                    encoded_inputs["offset_mapping"] = [(0, 0)] * difference + encoded_inputs["offset_mapping"]
                if "position_ids" in encoded_inputs:
                    encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
                if "start_positions" in encoded_inputs and isinstance(encoded_inputs["start_positions"], list):
                    encoded_inputs["start_positions"] = [0] * difference + encoded_inputs["start_positions"]
                if "end_positions" in encoded_inputs and isinstance(encoded_inputs["end_positions"], list):
                    encoded_inputs["end_positions"] = [0] * difference + encoded_inputs["end_positions"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        else:
            if "attn_mask_startend_row_indices" in encoded_inputs:
                if len(np.shape(encoded_inputs["attn_mask_startend_row_indices"])) == 1:
                    # TODO @DrownFish19 encoded_inputs["attn_mask_startend_row_indices"] is generated in the shape [seq_len]
                    # and convert the shape to [1,seq_len] here. However, it is supported in the generation phase.
                    encoded_inputs["attn_mask_startend_row_indices"] = np.array([encoded_inputs["attn_mask_startend_row_indices"]], dtype=np.int32)  # fmt:skip

        if "attn_mask_startend_row_indices" in encoded_inputs:
            assert len(np.shape(encoded_inputs["attn_mask_startend_row_indices"])) == 2  # [num_head, seq_len]

        return encoded_inputs

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        raise NotImplementedError

    def decode_token(
        self,
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
    ) -> Tuple[str, int, int]:
        """tokenizer decoding for the streaming generation use case. This method can be overrided for tokenizer that doesn't follow this API"""
        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.decode(
            all_input_ids[prefix_offset:read_offset], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        new_text = self.decode(
            all_input_ids[prefix_offset:], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        if len(new_text) > len(prefix_text) and not prefix_text.endswith("�") and not new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            prefix_index = new_text.index(prefix_text)
            new_text = new_text[prefix_index + len(prefix_text) :]
            return new_text, read_offset, len(all_input_ids)
        else:
            return "", prefix_offset, read_offset

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "paddle.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, paddle.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for seq in sequences
        ]

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "paddle.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, paddle.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument. "
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = self.all_special_ids  # cache the property

        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

        return special_tokens_mask

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def _eventual_warn_about_too_long_sequence(self, ids: List[int], max_length: Optional[int], verbose: bool):
        """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (`List[str]`): The ids produced by the tokenization
            max_length (`int`, *optional*): The max_length desired (does not trigger a warning if it is set)
            verbose (`bool`): Whether or not to print more information and warnings.

        """
        if max_length is None and len(ids) > self.model_max_length and verbose and self.model_max_length != 0:
            if not self.deprecation_warnings.get("sequence-length-is-longer-than-the-specified-maximum", False):
                logger.warning(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                    "will result in indexing errors"
                )
            self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

    def _switch_to_input_mode(self):
        """
        Private method to put the tokenizer in input mode (when it has different modes for input/outputs)
        """
        pass

    def _switch_to_target_mode(self):
        """
        Private method to put the tokenizer in target mode (when it has different modes for input/outputs)
        """
        pass

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        warnings.warn(
            "`as_target_tokenizer` is deprecated and will be removed in v5 of Transformers. You can tokenize your "
            "labels by using the argument `text_target` of the regular `__call__` method (either in the same call as "
            "your input texts if you use the same keyword arguments, or in a separate call."
        )
        self._switch_to_target_mode()
        self._in_target_context_manager = True
        yield
        self._in_target_context_manager = False
        self._switch_to_input_mode()

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTokenizer"):
        """
        Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the
        library are already mapped with `AutoTokenizer`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoTokenizer"`):
                The auto class to register this new tokenizer with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: Optional[str] = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepare model inputs for translation. For best performance, translate one sentence at a time.

        Arguments:
            src_texts (`List[str]`):
                List of documents to summarize or source language texts.
            tgt_texts (`list`, *optional*):
                List of summaries or target language texts.
            max_length (`int`, *optional*):
                Controls the maximum length for encoder inputs (documents to summarize or source language texts) If
                left unset or set to `None`, this will use the predefined model maximum length if a maximum length is
                required by one of the truncation/padding parameters. If the model has no specific maximum input length
                (like XLNet) truncation/padding to a maximum length will be deactivated.
            max_target_length (`int`, *optional*):
                Controls the maximum length of decoder inputs (target language texts or summaries) If left unset or set
                to `None`, this will use the max_length value.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `True`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            **kwargs:
                Additional keyword arguments passed along to `self.__call__`.

        Return:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to the encoder.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **labels** -- List of token ids for tgt_texts.

            The full set of keys `[input_ids, attention_mask, labels]`, will only be returned if tgt_texts is passed.
            Otherwise, input_ids, attention_mask will be the only keys.
        """
        # docstyle-ignore
        formatted_warning = """
`prepare_seq2seq_batch` is deprecated and will be removed in version 5 of HuggingFace Transformers. Use the regular
`__call__` method to prepare your inputs and targets.

Here is a short example:

model_inputs = tokenizer(src_texts, text_target=tgt_texts, ...)

If you either need to use different keyword arguments for the source and target texts, you should do two calls like
this:

model_inputs = tokenizer(src_texts, ...)
labels = tokenizer(text_target=tgt_texts, ...)
model_inputs["labels"] = labels["input_ids"]

See the documentation of your specific tokenizer for more details on the specific arguments to the tokenizer of choice.
For a more complete example, see the implementation of `prepare_seq2seq_batch`.
"""
        warnings.warn(formatted_warning, FutureWarning)
        # mBART-specific kwargs that should be ignored by other models.
        kwargs.pop("src_lang", None)
        kwargs.pop("tgt_lang", None)
        if max_length is None:
            max_length = self.model_max_length
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        with self.as_target_tokenizer():
            labels = self(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def get_fast_tokenizer_file(tokenization_files: List[str]) -> str:
    """
    Get the tokenization file to use for this version of transformers.

    Args:
        tokenization_files (`List[str]`): The list of available configuration files.

    Returns:
        `str`: The tokenization file to use.
    """
    tokenizer_files_map = {}
    for file_name in tokenization_files:
        search = _re_tokenizer_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            tokenizer_files_map[v] = file_name

    # Defaults to FULL_TOKENIZER_FILE and then try to look at some newer versions.
    tokenizer_file = FULL_TOKENIZER_FILE
    tokenizer_file = tokenizer_files_map[v]
    return tokenizer_file
