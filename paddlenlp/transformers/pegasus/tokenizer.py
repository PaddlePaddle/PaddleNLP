# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The IDEA-CCNL Authors and The HuggingFace Inc. team.
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

import collections
import os
import re
import jieba
import shutil
from paddle.utils import try_import
from .. import PretrainedTokenizer, AddedToken
from .. import BasicTokenizer, WordpieceTokenizer
from ..tokenizer_utils import _is_punctuation

__all__ = ["PegasusChineseTokenizer"]

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese": 1024,
    "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese": 1024,
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True

    return False


class PegasusChineseTokenizer(PretrainedTokenizer):
    r"""
    Construct a Pegasus tokenizer. Based on WordPiece.
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).

    Examples:
        .. code-block::

            from paddlenlp.transformers import PegasusChineseTokenizer

            tokenizer = PegasusChineseTokenizer.from_pretrained('IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese')
            print(tokenizer('欢迎使用PaddleNLP'))

            '''
            {'input_ids': [22355, 8994, 35941, 48563, 49375, 48877, 1],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
            '''

    """
    resource_files_names = {"vocab_file": "vocab.txt"}
    pretrained_resource_files_map = {}
    pretrained_init_configuration = {}
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        sep_token="[SEP]",
        cls_token="[CLS]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        offset=100,
        **kwargs
    ):

        self.offset = offset

        if additional_special_tokens is not None:
            if not isinstance(additional_special_tokens, list):
                raise TypeError(
                    f"additional_special_tokens should be of type {type(list)}, \
                     but is {type(additional_special_tokens)}"
                )

            additional_special_tokens_extended = (
                ([mask_token_sent] + additional_special_tokens)
                if mask_token_sent not in additional_special_tokens and mask_token_sent is not None
                else additional_special_tokens
            )

            # fill additional tokens with ..., <unk_token_102> in case not all additional tokens are already taken
            additional_special_tokens_extended += [
                f"<unk_{i}>" for i in range(len(additional_special_tokens_extended), self.offset - 1)
            ]

            if len(set(additional_special_tokens_extended)) != len(additional_special_tokens_extended):
                raise ValueError(
                    f"Please make sure that the provided additional_special_tokens \
                        do not contain an incorrectly shifted list of <unk_x> tokens. \
                        Found {additional_special_tokens_extended}."
                )
            additional_special_tokens = additional_special_tokens_extended
        else:
            additional_special_tokens = [mask_token_sent] if mask_token_sent is not None else []

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. \
                To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            eos_token=eos_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            additional_special_tokens=additional_special_tokens,
            strip_accents=strip_accents,
            **kwargs,
        )

        # Function object isn't serializable
        self.pre_tokenizer = lambda x: jieba.cut(x, HMM=False)
        self.mask_token_sent = mask_token_sent
        self.vocab = load_vocab(vocab_file)

        self.vocab[self.eos_token] = self.vocab.pop("[unused1]")
        self.vocab[self.pad_token] = self.vocab.pop("[PAD]")
        self.vocab[self.unk_token] = self.vocab.pop("[UNK]")

        if self.mask_token_sent is not None:
            self.vocab[self.mask_token] = self.vocab.pop("[unused3]")
            self.vocab[self.mask_token_sent] = self.vocab.pop("[unused2]")

        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                if self.do_basic_tokenize:
                    for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                        # If the token is part of the never_split set
                        if token in self.basic_tokenizer.never_split:
                            split_tokens.append(token)
                        else:
                            split_tokens += self.wordpiece_tokenizer.tokenize(token)
                else:
                    split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    @staticmethod
    def _cjk_punctuation():
        return "\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\
            \uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\
            \uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\
            \u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\
            \u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002"

    def convert_ids_to_tokens(self, ids, skip_special_tokens):
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
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids and index != 2:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # for token in
        # tokens = tokens or self.ids_to_tokens(ids)
        # tokens = [token for token in tokens if not self._is_special(token)]

        text = ""
        for i, token in enumerate(tokens):
            if token[:2] == "##":
                text += token[2:]
            elif len(token) == 1 and _is_chinese_char(ord(token)):
                text += token
            elif len(token) == 1 and _is_punctuation(token):
                text += token
                text += " "
            elif i > 0 and _is_chinese_char(ord(text[-1])):
                text += token
            elif tokens == "</s>":
                continue
            else:
                text += " "
                text += token

        text = re.sub(" +", " ", text)
        text = re.sub("' (re|m|s|t|ve|d|ll) ", "'\\1 ", text)
        punctuation = re.sub(" +", "", self._cjk_punctuation()).strip() + "+-/={(<["
        punctuation_regex = "|".join([re.escape(p) for p in punctuation])
        punctuation_regex = "(%s) " % punctuation_regex
        text = re.sub(punctuation_regex, "\\1", text)
        text = re.sub(r"(\d\.) (\d)", "\\1\\2", text)

        return text.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification
        tasks by concatenating and adding special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        # all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special

        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is
        called when adding special tokens using the tokenizer ``encode`` methods.
        """
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [self.eos_token_id]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [self.eos_token_id]

    def num_special_tokens_to_add(self, pair=False):
        """Just EOS"""
        return 1
