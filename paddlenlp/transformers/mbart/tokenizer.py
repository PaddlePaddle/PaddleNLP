# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
from contextlib import contextmanager
import sentencepiece as spm
from .. import PretrainedTokenizer

__all__ = ['MBartTokenizer']
FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR",
    "cs_CZ",
    "de_DE",
    "en_XX",
    "es_XX",
    "et_EE",
    "fi_FI",
    "fr_XX",
    "gu_IN",
    "hi_IN",
    "it_IT",
    "ja_XX",
    "kk_KZ",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "my_MM",
    "ne_NP",
    "nl_XX",
    "ro_RO",
    "ru_RU",
    "si_LK",
    "tr_TR",
    "vi_VN",
    "zh_CN",
]


class MBartTokenizer(PretrainedTokenizer):
    resource_files_names = {"vocab_file": "sentencepiece.bpe.model", }
    pretrained_resource_files_map = {
        "vocab_file": {
            "mbart-large-en-ro":
            "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model",
            "mbart-large-cc25":
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model",
        }
    }
    pretrained_init_configuration = {
        "mbart-large-cc25": {},
        "mbart-large-en-ro": {}
    }

    def __init__(self,
                 vocab_file,
                 src_lang=None,
                 tgt_lang=None,
                 bos_token="<s>",
                 eos_token="</s>",
                 sep_token="</s>",
                 cls_token="<s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 mask_token="<mask>",
                 **kwargs):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.fairseq_offset = 1
        self.fairseq_tokens_to_ids = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3
        }
        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset
            for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(
            self.lang_code_to_id) + self.fairseq_offset
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {
            v: k
            for k, v in self.fairseq_tokens_to_ids.items()
        }
        self.src_lang = src_lang if src_lang is not None else "en_XX"
        self.tgt_lang = tgt_lang
        # Get `special_tokens_map` after `_wrap_init()`
        self.eos_token_id = self.fairseq_tokens_to_ids[eos_token]
        self.unk_token_id = self.fairseq_tokens_to_ids[unk_token]
        self.set_src_lang_special_tokens(self.src_lang)

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len=None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=False,
                 return_attention_mask=True,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return super(MBartTokenizer, self).__call__(
            text, text_pair, max_seq_len, stride, is_split_into_words,
            pad_to_max_seq_len, truncation_strategy, return_position_ids,
            return_token_type_ids, return_attention_mask, return_length,
            return_overflowing_tokens, return_special_tokens_mask)

    def _tokenize(self, text):
        """ Tokenize a string. """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if tok == self.mask_token and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
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
            """
            Process special tokens: don't split special tokens.
            """
            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.fairseq_tokens_to_ids.keys():
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable((self.sp_model.encode(
                    token, out_type=str
                ) if token not in self.fairseq_tokens_to_ids.keys(
                ) else [token] for token in tokenized_text)))

        tokenized_text = split_on_tokens(self.fairseq_tokens_to_ids.keys(),
                                         text)
        return tokenized_text

    def _convert_token_to_id(self, token):
        """
        Converts a token (str) in an id using the vocab.
        """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab.
        """
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a single token or a sequence of tokens to an index or a
        sequence of indices.
        """
        if not isinstance(tokens, (list, tuple)):
            return self._convert_token_to_id(tokens)
        else:
            return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        """
        Converts a single index or a sequence of indices to a token or a
        sequence of tokens.
        """
        if not isinstance(ids, (list, tuple)):
            return self._convert_id_to_token(ids)
        tokens = [self._convert_id_to_token(_id) for _id in ids]
        return tokens

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (strings for sub-words) in a single string.
        """
        out_string = "".join(tokens).replace("▁", " ").strip()
        return out_string

    def convert_ids_to_string(self, ids):
        """
        Converts a sequence of tokens (strings for sub-words) in a single string.
        """
        tokens = self.convert_ids_to_tokens(ids)
        out_string = "".join(tokens).replace("▁", " ").strip()
        return out_string

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieve sequence ids from a token list that has no special tokens added.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)
                                                         ) + suffix_ones

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART sequence has the following format, where ``X`` represents the sequence:

        - ``input_ids`` (for encoder) ``X [eos, src_lang_code]``
        - ``decoder_input_ids``: (for decoder) ``X [eos, tgt_lang_code]``

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)

    def set_src_lang_special_tokens(self, src_lang):
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        self.cur_lang_code_id = self.lang_code_to_id[src_lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code_id]

    def set_tgt_lang_special_tokens(self, tgt_lang: str):
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        self.cur_lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code_id]
