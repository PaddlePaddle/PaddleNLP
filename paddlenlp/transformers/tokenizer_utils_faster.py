# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import six

from faster_tokenizers import Encoding as FasterEncoding
from faster_tokenizers import Tokenizer as FasterTokenizer

from .utils import InitTrackerMeta, fn_args_to_dict
from .convert_slow_tokenizer import convert_slow_tokenizer
from .tokenizer_utils_base import (
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PretrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
    PaddingStrategy, )
from .tokenizer_utils import PretrainedTokenizer
from paddlenlp.utils.log import logger

TOKENIZER_FILE = "tokenizer.json"
VOCAB_FILES_NAMES = {"tokenizer_file": TOKENIZER_FILE}
ADDED_TOKENS_FILE = "added_tokens.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"


class PretrainedFasterTokenizer(PretrainedTokenizerBase):
    resource_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class: PretrainedTokenizer = None
    can_save_slow_tokenizer: bool = True

    def __init__(self, *args, **kwargs):
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        faster_tokenizer_file = kwargs.pop("tokenizer_file", None)
        from_slow = kwargs.pop("from_slow", False)
        if tokenizer_object is not None:
            faster_tokenizer = tokenizer_object
        elif faster_tokenizer_file is not None and not from_slow:
            # We have a serialization from tokenizers which let us directly build the backend
            # From json file
            faster_tokenizer = FasterTokenizer.from_file(faster_tokenizer_file)
        elif slow_tokenizer is not None:
            # We need to convert a slow tokenizer to build the backend
            faster_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        elif self.slow_tokenizer_class is not None:
            # We need to create and convert a slow tokenizer to build the backend
            slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
            faster_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        else:
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: \n"
                "(1) a `faster_tokenizers` library serialization file, \n"
                "(2) a slow tokenizer instance to convert or \n"
                "(3) an equivalent slow tokenizer class to instantiate and convert. \n"
                "You need to have sentencepiece installed to convert a slow tokenizer to a fast one."
            )
        self._tokenizer = faster_tokenizer

        if slow_tokenizer is not None:
            kwargs.update(slow_tokenizer.init_kwargs)

        self._decode_use_source_tokenizer = False
        # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self._tokenizer.get_vocab_size(with_added_vocabulary=False)

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_vocabulary=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        base_vocab = self._tokenizer.get_vocab(with_added_vocabulary=False)
        full_vocab = self._tokenizer.get_vocab(with_added_vocabulary=True)
        added_vocab = dict((tok, index) for tok, index in full_vocab.items()
                           if tok not in base_vocab)
        return added_vocab

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> FasterTokenizer:
        return self._tokenizer

    def _convert_encoding(
            self,
            encoding: FasterEncoding,
            return_token_type_ids: Optional[bool]=None,
            return_attention_mask: Optional[bool]=None,
            return_overflowing_tokens: bool=False,
            return_special_tokens_mask: bool=False,
            return_offsets_mapping: bool=False,
            return_length: bool=False,
            verbose: bool=True, ) -> Tuple[Dict[str, Any], List[
                FasterEncoding]]:
        """
        Convert the encoding representation (from low-level PaddleNLP FasterTokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(
                    e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        return encoding_dict, encodings

    def convert_tokens_to_ids(
            self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self,
                    new_tokens: List[Union[str, AddedToken]],
                    special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool=False) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(
            self, ids: Union[int, List[int]],
            skip_special_tokens: bool=False) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def tokenize(self,
                 text: str,
                 pair: Optional[str]=None,
                 add_special_tokens: bool=False,
                 **kwargs) -> List[str]:
        return self.encode(
            text=text,
            text_pair=pair,
            add_special_tokens=add_special_tokens,
            **kwargs).tokens()

    def set_truncation_and_padding(
            self,
            padding_strategy: PaddingStrategy,
            truncation_strategy: TruncationStrategy,
            max_length: int,
            stride: int,
            pad_to_multiple_of: Optional[int], ):
        """
        Define the truncation and the padding strategies for fast tokenizers (provided by PaddleNLP's faster_tokenizers
        library) and restore the tokenizer settings afterwards.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy ([`~utils.PaddingStrategy`]):
                The kind of padding that will be applied to the input
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`]):
                The kind of truncation that will be applied to the input
            max_length (`int`):
                The maximum size of a sequence.
            stride (`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        """
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            if _truncation is not None:
                self._tokenizer.disable_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy.value,
                "direction": self.truncation_side,
            }

            # _truncation might contain more keys that the target `transformers`
            # supports. Use only the target keys to trigger `enable_truncation`.
            # This should enable this code to works on various `tokenizers`
            # targets.
            if _truncation != target:
                self._tokenizer.enable_truncation(**target)

        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            if _padding is not None:
                self._tokenizer.disable_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_token_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            if _padding != target:
                self._tokenizer.enable_padding(**target)

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[List[TextInput], List[
                TextInputPair], List[PreTokenizedInput], List[
                    PreTokenizedInputPair], List[EncodedInput], List[
                        EncodedInputPair], ],
            add_special_tokens: bool=True,
            padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy=TruncationStrategy.
            DO_NOT_TRUNCATE,
            max_length: Optional[int]=None,
            stride: int=0,
            is_split_into_words: bool=False,
            pad_to_multiple_of: Optional[int]=None,
            return_position_ids: Optional[bool]=None,
            return_tensors: Optional[str]=None,
            return_token_type_ids: Optional[bool]=None,
            return_attention_mask: Optional[bool]=None,
            return_overflowing_tokens: bool=False,
            return_special_tokens_mask: bool=False,
            return_dict: bool=True,
            return_offsets_mapping: bool=False,
            return_length: bool=False,
            verbose: bool=True, ) -> BatchEncoding:

        if not isinstance(batch_text_or_text_pairs, list):
            raise TypeError(
                f"batch_text_or_text_pairs has to be a list (got {type(batch_text_or_text_pairs)})"
            )

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of, )
        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words)

        # Convert encoding to dict
        # `Tokens` has type: Tuple[
        #                       List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]],
        #                       List[FasterEncoding]
        #                    ]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose, ) for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
        # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
        # (we say ~ because the number of overflow varies with the example in the batch)
        #
        # To match each overflowing sample with the original sample in the batch
        # we add an overflow_to_sample_mapping array (see below)
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [
            e for _, item in tokens_and_encodings for e in item
        ]

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, (toks, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks["input_ids"])
            sanitized_tokens[
                "overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids in sanitized_tokens["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length,
                                                        verbose)
        return BatchEncoding(
            sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput,
                                      EncodedInput]]=None,
            add_special_tokens: bool=True,
            padding_strategy: PaddingStrategy=PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy=TruncationStrategy.
            DO_NOT_TRUNCATE,
            max_length: Optional[int]=None,
            stride: int=0,
            is_split_into_words: bool=False,
            pad_to_multiple_of: Optional[int]=None,
            return_position_ids: Optional[bool]=None,
            return_tensors: Optional[bool]=None,
            return_token_type_ids: Optional[bool]=None,
            return_attention_mask: Optional[bool]=None,
            return_overflowing_tokens: bool=False,
            return_special_tokens_mask: bool=False,
            return_offsets_mapping: bool=False,
            return_length: bool=False,
            verbose: bool=True,
            **kwargs) -> BatchEncoding:

        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_output = self._batch_encode_plus(
            batched_input,
            is_split_into_words=is_split_into_words,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_position_ids=return_position_ids,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs, )

        # Return tensor is None, then we can remove the leading batch axis
        # Overflowing tokens are returned as a batch of output so we keep them in this case
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0]
                    if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings, )

        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"],
                                                    max_length, verbose)

        return batched_output

    def _save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            file_names: Tuple[str],
            legacy_format: Optional[bool]=None,
            filename_prefix: Optional[str]=None, ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens as well as in a unique JSON
        file containing {config + vocab + added-tokens}.
        """
        save_directory = str(save_directory)

        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError(
                "Your tokenizer does not have a legacy version defined and therefore cannot register this version. You "
                "might consider leaving the legacy_format at `None` or setting it to `False`."
            )

        save_slow = ((legacy_format is None or legacy_format is True) and
                     self.slow_tokenizer_class is not None and
                     self.can_save_slow_tokenizer)
        save_fast = legacy_format is None or legacy_format is False

        if save_slow:
            added_tokens_file = os.path.join(
                save_directory,
                (filename_prefix + "-"
                 if filename_prefix else "") + ADDED_TOKENS_FILE)
            added_vocab = self.get_added_vocab()
            if added_vocab:
                with open(added_tokens_file, "w", encoding="utf-8") as f:
                    out_str = json.dumps(added_vocab, ensure_ascii=False)
                    f.write(out_str)

            vocab_files = self.save_vocabulary(
                save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file, )

        if save_fast:
            tokenizer_file = os.path.join(
                save_directory, (filename_prefix + "-"
                                 if filename_prefix else "") + TOKENIZER_FILE)
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file, )
        return file_names

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        return self.backend_tokenizer.decoder.decode(tokens)

    def _decode(self,
                token_ids: Union[int, List[int]],
                skip_special_tokens: bool=False,
                clean_up_tokenization_spaces: bool=True,
                **kwargs) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer",
                                                       False)

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text
