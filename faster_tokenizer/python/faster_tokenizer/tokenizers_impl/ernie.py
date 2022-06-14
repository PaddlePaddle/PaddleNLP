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

from faster_tokenizer.normalizers import BertNormalizer
from faster_tokenizer.pretokenizers import BertPreTokenizer
from faster_tokenizer.models import WordPiece, FasterWordPiece
from faster_tokenizer.postprocessors import BertPostProcessor
from faster_tokenizer import decoders
from faster_tokenizer import Tokenizer

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
                 strip_accents=True,
                 lowercase=True,
                 wordpieces_prefix="##",
                 max_sequence_len=None,
                 use_faster_wordpiece=False,
                 use_faster_wordpiece_with_pretokenization=False):
        tokenizer_model = WordPiece if not use_faster_wordpiece else FasterWordPiece
        model_kwargs = {
            "unk_token": str(unk_token),
            "continuing_subword_prefix": wordpieces_prefix
        }
        if use_faster_wordpiece:
            model_kwargs[
                "with_pretokenization"] = use_faster_wordpiece_with_pretokenization
        if vocab is not None:
            tokenizer = Tokenizer(tokenizer_model(vocab, **model_kwargs))
        else:
            tokenizer = Tokenizer(tokenizer_model(**model_kwargs))

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
        if not use_faster_wordpiece or not use_faster_wordpiece_with_pretokenization:
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
