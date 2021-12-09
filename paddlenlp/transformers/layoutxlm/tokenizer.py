# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Tokenization classes for LayoutXLM model."""

import itertools
from dataclasses import dataclass, field
from collections import OrderedDict
from paddle.utils import try_import
from typing import List, Optional

from .. import PretrainedTokenizer
from ..tokenizer_utils import _is_punctuation, _is_control, _is_whitespace

SPIECE_UNDERLINE = "▁"


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(
        _is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(
            last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(
        _is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(
            first_char))


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


class LayoutXLMTokenizer(PretrainedTokenizer):
    resource_files_names = {"vocab_file": "sentencepiece.bpe.model"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "layoutxlm-base-uncased":
            "https://bj.bcebos.com/paddlenlp/models/transformers/layoutxlm_base/sentencepiece.bpe.model",
        }
    }
    pretrained_init_configuration = {
        "layoutxlm-base-uncased": {
            "do_lower_case": False
        },
    }
    pretrained_positional_embedding_sizes = {"layoutxlm-base-uncased": 512, }
    max_model_input_sizes = pretrained_positional_embedding_sizes
    model_input_names = ["input_ids", "attention_mask"]

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

    def __init__(self,
                 vocab_file,
                 bos_token="<s>",
                 eos_token="</s>",
                 sep_token="</s>",
                 cls_token="<s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 mask_token="<mask>",
                 **kwargs):
        mask_token = AddedToken(
            mask_token, lstrip=True,
            rstrip=False) if isinstance(mask_token, str) else mask_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._sep_token = sep_token
        self._cls_token = cls_token
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        spm = try_import("sentencepiece")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self.vocab_file = vocab_file

        self.tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.offset = 1

        self.tokens_to_ids["<mask>"] = len(self.sp_model) + self.offset
        self.ids_to_tokens = {v: k for k, v in self.tokens_to_ids.items()}

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int],
            token_ids_1: Optional[List[int]]=None) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]]=None,
            already_has_special_tokens: bool=False) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)
                                                          ) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int],
            token_ids_1: Optional[List[int]]=None) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.offset + 1  # Add the <mask> token

    def get_vocab(self):
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def special_tokens_map_extended(self):
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            if hasattr(self, "_" + attr):
                attr_value = getattr(self, "_" + attr)
                if attr_value:
                    set_attr[attr] = attr_value
        return set_attr

    def all_special_tokens_extended(self):
        all_toks = []
        set_attr = self.special_tokens_map_extended()
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (
                list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    def tokenize(self, text, **kwargs) -> List[str]:
        all_special_tokens_extended = dict(
            (t.content, t) for t in self.all_special_tokens_extended()
            if isinstance(t, AddedToken))

        def split_on_token(tok, text):
            result = []
            if isinstance(tok, AddedToken):
                tok_extended = all_special_tokens_extended.get(tok, None)
                split_text = text.split(tok.content)
            else:
                tok_extended = None
                split_text = text.split(tok)
            full_word = ""
            for i, sub_text in enumerate(split_text):
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.single_word:
                        # Try to avoid splitting on token
                        if (i < len(split_text) - 1 and
                                not _is_end_of_word(sub_text) and
                                not _is_start_of_word(split_text[i + 1])):
                            # Don't extract the special token
                            full_word += sub_text + tok
                        elif full_word:
                            full_word += sub_text
                            result.append(full_word)
                            full_word = ""
                            continue
                    # Strip white spaces on the right
                    if tok_extended.rstrip and i > 0:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        sub_text = sub_text.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()
                    if i > 0:
                        sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.all_special_tokens_extended():
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable((self._tokenize(
                    token) if token not in self.all_special_tokens_extended(
                    ) else [token] for token in tokenized_text)))

        no_split_token = self.all_special_tokens_extended()
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.tokens_to_ids:
            return self.tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.ids_to_tokens:
            return self.ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def convert_tokens_to_ids(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._convert_token_to_id(tokens)
        else:
            return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if not isinstance(ids, (list, tuple)):
            return self._convert_id_to_token(ids)
        tokens = [self._convert_id_to_token(_id) for _id in ids]
        if skip_special_tokens:
            return [
                token for token in tokens
                if token not in self.all_special_tokens
            ]
        return tokens

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))
