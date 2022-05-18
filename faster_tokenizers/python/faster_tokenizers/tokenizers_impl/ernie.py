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

from .base_tokenizer import BaseFasterTokenizer

from faster_tokenizers.normalizers import BertNormalizer
from faster_tokenizers.pretokenizers import BertPreTokenizer
from faster_tokenizers.models import WordPiece
from faster_tokenizers.postprocessors import BertPostProcessor
from faster_tokenizers import decoders
from faster_tokenizers import Tokenizer

__all__ = ['ErnieFasterTokenizer']


class ErnieFasterTokenizer(BaseFasterTokenizer):
    def __init__(self,
                 vocab=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 cls_token="[CLS]",
                 pad_token="[PAD]",
                 mask_token="[MASK]",
                 clean_text=True,
                 handle_chinese_chars=True,
                 strip_accents=None,
                 lowercase=True,
                 wordpieces_prefix="##",
                 max_sequence_len=None):
        if vocab is not None:
            tokenizer = Tokenizer(
                WordPiece(
                    vocab,
                    unk_token=str(unk_token),
                    continuing_subword_prefix=wordpieces_prefix))
        else:
            tokenizer = Tokenizer(
                WordPiece(
                    unk_token=str(unk_token),
                    continuing_subword_prefix=wordpieces_prefix))

        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        tokenizer.normalizer = BertNormalizer(
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase)
        tokenizer.pretokenizer = BertPreTokenizer()

        if vocab is not None:
            sep_token_id = tokenizer.token_to_id(str(sep_token))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = tokenizer.token_to_id(str(cls_token))
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            tokenizer.postprocessor = BertPostProcessor(
                (str(sep_token), sep_token_id), (str(cls_token), cls_token_id))

        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)
        if max_sequence_len == None:
            tokenizer.disable_truncation()
        else:
            tokenizer.enable_truncation(max_sequence_len)

        parameters = {
            "model": "BertWordPiece",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "clean_text": clean_text,
            "handle_chinese_chars": handle_chinese_chars,
            "strip_accents": strip_accents,
            "lowercase": lowercase,
            "wordpieces_prefix": wordpieces_prefix,
        }

        super().__init__(tokenizer, parameters)
