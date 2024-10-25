# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Inc. team.
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
# flake8: noqa
"""
Base classes common to both the slow and the fast tokenization classes: PreTrainedTokenizerBase (host all the user
fronting encoding methods) Special token mixing (host the special tokens logic) and BatchEncoding (wrap the dictionary
of output with special method for the Fast tokenizers)
"""

import copy
import importlib
import json
import os
import re
import warnings
from collections import OrderedDict, UserDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


def is_sentencepiece_available():
    return importlib.util.find_spec("sentencepiece") is not None


def is_tokenizers_available():
    return importlib.util.find_spec("tokenizers") is not None


if is_tokenizers_available():
    from tokenizers import AddedToken
else:

    @dataclass(frozen=True, eq=True)
    class AddedToken:
        """
        AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
        way it should behave.
        """

        content: str = field(default_factory=str)
        single_word: bool = False
        lstrip: bool = False
        rstrip: bool = False
        normalized: bool = True

        def __getstate__(self):
            return self.__dict__


TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        (
            "albert",
            (
                "AlbertTokenizer" if is_sentencepiece_available() else None,
                "AlbertTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("bart", ("BartTokenizer", "BartTokenizerFast")),
        (
            "barthez",
            (
                "BarthezTokenizer" if is_sentencepiece_available() else None,
                "BarthezTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("bartpho", ("BartphoTokenizer", None)),
        ("bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
        ("bert-generation", ("BertGenerationTokenizer" if is_sentencepiece_available() else None, None)),
        ("bert-japanese", ("BertJapaneseTokenizer", None)),
        ("bertweet", ("BertweetTokenizer", None)),
        (
            "big_bird",
            (
                "BigBirdTokenizer" if is_sentencepiece_available() else None,
                "BigBirdTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("bigbird_pegasus", ("PegasusTokenizer", "PegasusTokenizerFast" if is_tokenizers_available() else None)),
        ("blenderbot", ("BlenderbotTokenizer", "BlenderbotTokenizerFast")),
        ("blenderbot-small", ("BlenderbotSmallTokenizer", None)),
        ("bloom", (None, "BloomTokenizerFast" if is_tokenizers_available() else None)),
        ("byt5", ("ByT5Tokenizer", None)),
        (
            "camembert",
            (
                "CamembertTokenizer" if is_sentencepiece_available() else None,
                "CamembertTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("canine", ("CanineTokenizer", None)),
        (
            "clip",
            (
                "CLIPTokenizer",
                "CLIPTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("convbert", ("ConvBertTokenizer", "ConvBertTokenizerFast" if is_tokenizers_available() else None)),
        (
            "cpm",
            (
                "CpmTokenizer" if is_sentencepiece_available() else None,
                "CpmTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("ctrl", ("CTRLTokenizer", None)),
        ("data2vec-text", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
        ("deberta", ("DebertaTokenizer", "DebertaTokenizerFast" if is_tokenizers_available() else None)),
        (
            "deberta-v2",
            (
                "DebertaV2Tokenizer" if is_sentencepiece_available() else None,
                "DebertaV2TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("distilbert", ("DistilBertTokenizer", "DistilBertTokenizerFast" if is_tokenizers_available() else None)),
        (
            "dpr",
            (
                "DPRQuestionEncoderTokenizer",
                "DPRQuestionEncoderTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("electra", ("ElectraTokenizer", "ElectraTokenizerFast" if is_tokenizers_available() else None)),
        ("flaubert", ("FlaubertTokenizer", None)),
        ("fnet", ("FNetTokenizer", "FNetTokenizerFast" if is_tokenizers_available() else None)),
        ("fsmt", ("FSMTTokenizer", None)),
        ("funnel", ("FunnelTokenizer", "FunnelTokenizerFast" if is_tokenizers_available() else None)),
        ("gpt2", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
        ("gpt_neo", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
        ("gpt_neox", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
        ("gptj", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
        ("herbert", ("HerbertTokenizer", "HerbertTokenizerFast" if is_tokenizers_available() else None)),
        ("hubert", ("Wav2Vec2CTCTokenizer", None)),
        ("ibert", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
        ("layoutlm", ("LayoutLMTokenizer", "LayoutLMTokenizerFast" if is_tokenizers_available() else None)),
        ("layoutlmv2", ("LayoutLMv2Tokenizer", "LayoutLMv2TokenizerFast" if is_tokenizers_available() else None)),
        ("layoutlmv3", ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast" if is_tokenizers_available() else None)),
        ("layoutxlm", ("LayoutXLMTokenizer", "LayoutXLMTokenizerFast" if is_tokenizers_available() else None)),
        ("led", ("LEDTokenizer", "LEDTokenizerFast" if is_tokenizers_available() else None)),
        ("longformer", ("LongformerTokenizer", "LongformerTokenizerFast" if is_tokenizers_available() else None)),
        (
            "longt5",
            (
                "T5Tokenizer" if is_sentencepiece_available() else None,
                "T5TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("luke", ("LukeTokenizer", None)),
        ("lxmert", ("LxmertTokenizer", "LxmertTokenizerFast" if is_tokenizers_available() else None)),
        ("m2m_100", ("M2M100Tokenizer" if is_sentencepiece_available() else None, None)),
        ("marian", ("MarianTokenizer" if is_sentencepiece_available() else None, None)),
        (
            "mbart",
            (
                "MBartTokenizer" if is_sentencepiece_available() else None,
                "MBartTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        (
            "mbart50",
            (
                "MBart50Tokenizer" if is_sentencepiece_available() else None,
                "MBart50TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("megatron-bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
        ("mluke", ("MLukeTokenizer" if is_sentencepiece_available() else None, None)),
        ("mobilebert", ("MobileBertTokenizer", "MobileBertTokenizerFast" if is_tokenizers_available() else None)),
        ("mpnet", ("MPNetTokenizer", "MPNetTokenizerFast" if is_tokenizers_available() else None)),
        (
            "mt5",
            (
                "MT5Tokenizer" if is_sentencepiece_available() else None,
                "MT5TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        (
            "nystromformer",
            (
                "AlbertTokenizer" if is_sentencepiece_available() else None,
                "AlbertTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("openai-gpt", ("OpenAIGPTTokenizer", "OpenAIGPTTokenizerFast" if is_tokenizers_available() else None)),
        ("opt", ("GPT2Tokenizer", None)),
        (
            "pegasus",
            (
                "PegasusTokenizer" if is_sentencepiece_available() else None,
                "PegasusTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        (
            "perceiver",
            (
                "PerceiverTokenizer",
                None,
            ),
        ),
        ("phobert", ("PhobertTokenizer", None)),
        ("plbart", ("PLBartTokenizer" if is_sentencepiece_available() else None, None)),
        ("prophetnet", ("ProphetNetTokenizer", None)),
        ("qdqbert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
        ("rag", ("RagTokenizer", None)),
        ("realm", ("RealmTokenizer", "RealmTokenizerFast" if is_tokenizers_available() else None)),
        (
            "reformer",
            (
                "ReformerTokenizer" if is_sentencepiece_available() else None,
                "ReformerTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        (
            "rembert",
            (
                "RemBertTokenizer" if is_sentencepiece_available() else None,
                "RemBertTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("retribert", ("RetriBertTokenizer", "RetriBertTokenizerFast" if is_tokenizers_available() else None)),
        ("roberta", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
        ("roformer", ("RoFormerTokenizer", "RoFormerTokenizerFast" if is_tokenizers_available() else None)),
        ("speech_to_text", ("Speech2TextTokenizer" if is_sentencepiece_available() else None, None)),
        ("speech_to_text_2", ("Speech2Text2Tokenizer", None)),
        ("splinter", ("SplinterTokenizer", "SplinterTokenizerFast")),
        (
            "squeezebert",
            ("SqueezeBertTokenizer", "SqueezeBertTokenizerFast" if is_tokenizers_available() else None),
        ),
        (
            "t5",
            (
                "T5Tokenizer" if is_sentencepiece_available() else None,
                "T5TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("tapas", ("TapasTokenizer", None)),
        ("tapex", ("TapexTokenizer", None)),
        ("transfo-xl", ("TransfoXLTokenizer", None)),
        ("vilt", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
        ("visual_bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
        ("wav2vec2", ("Wav2Vec2CTCTokenizer", None)),
        ("wav2vec2-conformer", ("Wav2Vec2CTCTokenizer", None)),
        ("wav2vec2_phoneme", ("Wav2Vec2PhonemeCTCTokenizer", None)),
        (
            "xglm",
            (
                "XGLMTokenizer" if is_sentencepiece_available() else None,
                "XGLMTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("xlm", ("XLMTokenizer", None)),
        ("xlm-prophetnet", ("XLMProphetNetTokenizer" if is_sentencepiece_available() else None, None)),
        (
            "xlm-roberta",
            (
                "XLMRobertaTokenizer" if is_sentencepiece_available() else None,
                "XLMRobertaTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("xlm-roberta-xl", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
        (
            "xlnet",
            (
                "XLNetTokenizer" if is_sentencepiece_available() else None,
                "XLNetTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        (
            "yoso",
            (
                "AlbertTokenizer" if is_sentencepiece_available() else None,
                "AlbertTokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
    ]
)

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
    ]
)


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

    return key.replace("-", "_")


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:

            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys():
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.add("Hello 友達")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

        >>> trie.add("Hello")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        ```
        """
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str) -> List[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string splitted along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS] This is a extra_id_100"]

        >>> trie.add("[CLS]")
        >>> trie.add("extra_id_1")
        >>> trie.add("extra_id_100")
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS]", " This is a ", "extra_id_100"]
        ```
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = 0
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                        # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
                    " anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


from enum import Enum


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PADDLE = "paddle"
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class BatchEncoding(UserDict):
    """
    Holds the output of the [`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`],
    [`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`] and
    [`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus`] methods (tokens, attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods
            ('input_ids', 'attention_mask', etc.).
        encoding (`tokenizers.Encoding` or `Sequence[tokenizers.Encoding]`, *optional*):
            If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
            space to token space the `tokenizers.Encoding` instance or list of instance (for batches) hold this
            information.
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
        n_sequences (`Optional[int]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    def __init__(
        self,
        data=None,
        encoding=None,
        tensor_type=None,
        prepend_batch_axis: bool = False,
        n_sequences=None,
    ):
        super().__init__(data)

        # if isinstance(encoding, EncodingFast):
        #    encoding = [encoding]

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
        `bool`: Indicate whether this [`BatchEncoding`] was generated from the result of a [`PreTrainedTokenizerFast`]
        or not.
        """
        return self._encodings is not None

    # def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:

    def __getitem__(self, item):
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `tokenizers.Encoding` for batch item with index `key`.
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError(
                "Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers"
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
    def encodings(self):
        """
        `Optional[List[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns `None` if
        the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        return self._encodings

    def tokens(self, batch_index=0):
        """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[str]`: The list of tokens at that index.
        """
        if not self._encodings:
            raise ValueError("tokens() is not available when using Python-based tokenizers")
        return self._encodings[batch_index].tokens

    def sequence_ids(self, batch_index=0):
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
            raise ValueError("sequence_ids() is not available when using Python-based tokenizers")
        return self._encodings[batch_index].sequence_ids

    def words(self, batch_index=0):
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
            raise ValueError("word_ids() is not available when using Python-based tokenizers")
        return self._encodings[batch_index].word_ids

    def token_to_sequence(self, batch_or_token_index, token_index):
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

    def token_to_word(self, batch_or_token_index, token_index=None):
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

    def word_to_tokens(self, batch_or_word_index, word_index=None, sequence_index=0):
        """
        Get the encoded token span corresponding to a word in a sequence of the batch.

        Token spans are returned as a [`~tokenization_utils_base.TokenSpan`] with:

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
            Optional [`~tokenization_utils_base.TokenSpan`] Span of tokens in the encoded sequence. Returns `None` if
            no tokens correspond to the word.
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

    def token_to_chars(self, batch_or_token_index: int, token_index=None):
        """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a [`~tokenization_utils_base.CharSpan`] with:

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
            [`~tokenization_utils_base.CharSpan`]: Span of characters in the original string, or None, if the token
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
            `int`: Index of the token.
        """

        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(char_index, sequence_index)

    def word_to_chars(self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0):
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

    def convert_to_tensors(self, tensor_type=None, prepend_batch_axis: bool = False):
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

        # Get a function reference for the correct framework
        if tensor_type == "paddle":
            import paddle

            as_tensor = paddle.to_tensor
            is_tensor = paddle.is_tensor
        else:
            as_tensor = np.asarray
            is_tensor = _is_numpy
        # (mfuntowicz: This code is unreachable)
        # else:
        #     raise ImportError(
        #         f"Unable to convert output to tensors format {tensor_type}"
        #     )

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
            except:  # noqa E722
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    )
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )

        return self


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


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
            A tuple or a list of additional special tokens.
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
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        self.verbose = verbose
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []
        self.tokens_trie = Trie()

        self._decode_use_source_tokenizer = False

        # We directly set the hidden value to allow initialization with special tokens
        # which are not yet in the vocabulary. Necessary for serialization/de-serialization
        # TODO clean this up at some point (probably by switching to fast tokenizers)
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"special token {key} has to be either str or AddedToken but got: {type(value)}")

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
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

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def sanitize_special_tokens(self) -> int:
        """
        Make sure that all the special tokens attributes of the tokenizer (`tokenizer.mask_token`,
        `tokenizer.cls_token`, etc.) are in the vocabulary.

        Add the missing ones to the vocabulary if needed.

        Return:
            `int`: The number of tokens added in the vocabulary during the operation.
        """
        return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, AddedToken]]) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).

        Note,None When adding new tokens to the vocabulary, you should make sure to also resize the token embedding
        matrix of the model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - Special tokens are carefully handled by the tokenizer (they are never split).
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

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

            if self.verbose:
                # logger.info(f"Assigning {value} to the {key} key of the tokenizer")
                print(f"Assigning {value} to the {key} key of the tokenizer")
            setattr(self, key, value)

            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, (str, AddedToken)) for t in value
                ), f"Tokens {value} for key {key} should all be str or AddedToken instances"
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(
                    value, (str, AddedToken)
                ), f"Token {value} for key {key} should be a str or an AddedToken instance"
                added_tokens += self.add_tokens([value], special_tokens=True)

        return added_tokens

    def add_tokens(
        self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
    ) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        Note,None When adding new tokens to the vocabulary, you should make sure to also resize the token embedding
        matrix of the model so that its embedding matrix matches the tokenizer.

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
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
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
        new_tokens = [str(tok) for tok in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            if not isinstance(token, str):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if not special_tokens and hasattr(self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (
                token != self.unk_token
                and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                # if self.verbose:
            # logger.info(f"Adding {token} to the vocabulary")
            # print(f"Adding {token} to the vocabulary")

        added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        # Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
        if special_tokens:
            if len(new_tokens) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, new_tokens[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:
            # Or on the newly added tokens
            if len(tokens_to_add) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens, tokens_to_add[0])
            else:
                self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(tokens_to_add)))
        self._create_trie(self.unique_no_split_tokens)

        return len(tokens_to_add)

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(self, "do_lower_case") and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    @property
    def bos_token(self) -> str:
        """
        `str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        if self._bos_token is None and self.verbose:
            print("Using bos_token, but it is not set yet.")
            # logger.error("Using bos_token, but it is not set yet.")
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._eos_token is None and self.verbose:
            # logger.error("Using eos_token, but it is not set yet.")
            print("Using eos_token, but it is not set yet.")
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None and self.verbose:
            print("Using unk_token, but it is not set yet.")
            # logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
        having been set.
        """
        if self._sep_token is None and self.verbose:
            print("Using sep_token, but it is not set yet.")
            # logger.error("Using sep_token, but it is not set yet.")
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        if self._pad_token is None and self.verbose:
            # logger.error("Using pad_token, but it is not set yet.")
            print("Using pad_token, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
        depth of the model. Log an error if used while not having been set.
        """
        if self._cls_token is None and self.verbose:
            # logger.error("Using cls_token, but it is not set yet.")
            print("Using cls_token, but it is not set yet.")
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
        """
        if self._mask_token is None and self.verbose:
            # logger.error("Using mask_token, but it is not set yet.")
            print("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the additional special tokens you may want to use. Log an error if used while not having been
        set.
        """
        if self._additional_special_tokens is None and self.verbose:
            # logger.error("Using additional_special_tokens, but it is not set yet.")
            print("Using additional_special_tokens, but it is not set yet.")
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns `None` if the token has not been set.
        """
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        return self._pad_token_type_id

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
        leveraging self-attention along the full depth of the model.

        Returns `None` if the token has not been set.
        """
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns `None` if the token has not been set.
        """
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        `List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not having
        been set.
        """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @bos_token_id.setter
    def bos_token_id(self, value):
        self._bos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @eos_token_id.setter
    def eos_token_id(self, value):
        self._eos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_ids_to_tokens(value) if value is not None else None

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_ids_to_tokens(value) if value is not None else None

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_ids_to_tokens(value) if value is not None else None

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_ids_to_tokens(value) if value is not None else None

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_ids_to_tokens(value) if value is not None else None

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [self.convert_ids_to_tokens(value) for value in values]

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
        `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Convert potential tokens of `tokenizers.AddedToken` type to string.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = (
                    type(attr_value)(str(attr_value_sub) for attr_value_sub in attr_value)
                    if isinstance(attr_value, (list, tuple))
                    else str(attr_value)
                )
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
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        `List[Union[str, tokenizers.AddedToken]]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class
        attributes.

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids
