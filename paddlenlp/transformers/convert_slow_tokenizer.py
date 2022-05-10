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

from typing import Dict, List, Tuple

from faster_tokenizers import Tokenizer, normalizers, pretokenizers, postprocessors
from faster_tokenizers.models import WordPiece


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class BertConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(
            WordPiece(
                vocab._token_to_idx,
                unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = True
        strip_accents = True
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case, )
        tokenizer.pretokenizer = pretokenizers.BertPreTokenizer()

        cls_token = str(self.original_tokenizer.cls_token)
        sep_token = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.postprocessor = postprocessors.BertPostProcessor(
            (str(sep_token), sep_token_id), (str(cls_token), cls_token_id))
        return tokenizer


class ErnieConverter(BertConverter):
    pass


SLOW_TO_FAST_CONVERTERS = {
    "BertTokenizer": BertConverter,
    "ErnieTokenizer": ErnieConverter,
    # TODO(zhoushunjie): Need to implement more TokenizerConverter
}


def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenizer_utils_base.PretrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenizer_utils_base.PretrainedFasterTokenizer`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenizer_utils_base.PretrainedFasterTokenizer`]
    """

    tokenizer_class_name = transformer_tokenizer.__class__.__name__

    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        raise ValueError(
            f"An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance. "
            f"No converter was found. Currently available slow->fast convertors: {list(SLOW_TO_FAST_CONVERTERS.keys())}"
        )

    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]

    return converter_class(transformer_tokenizer).converted()
