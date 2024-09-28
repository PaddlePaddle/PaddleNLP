# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""Tokenization class for ALBERT model."""

from .. import BertTokenizer

__all__ = ["AlbertChineseTokenizer"]

SPIECE_UNDERLINE = "▁"


class AlbertChineseTokenizer(BertTokenizer):
    resource_files_names = {"vocab_file": "vocab.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "albert-chinese-tiny": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-tiny.vocab.txt",
            "albert-chinese-small": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-small.vocab.txt",
            "albert-chinese-base": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-base.vocab.txt",
            "albert-chinese-large": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-large.vocab.txt",
            "albert-chinese-xlarge": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-xlarge.vocab.txt",
            "albert-chinese-xxlarge": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-chinese-xxlarge.vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "albert-chinese-tiny": {
            "do_lower_case": False,
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
        },
        "albert-chinese-small": {
            "do_lower_case": False,
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
        },
        "albert-chinese-base": {
            "do_lower_case": False,
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
        },
        "albert-chinese-large": {
            "do_lower_case": False,
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
        },
        "albert-chinese-xlarge": {
            "do_lower_case": False,
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
        },
        "albert-chinese-xxlarge": {
            "do_lower_case": False,
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
        },
    }
    max_model_input_sizes = {
        "albert-chinese-tiny": 512,
        "albert-chinese-small": 512,
        "albert-chinese-base": 512,
        "albert-chinese-large": 512,
        "albert-chinese-xlarge": 512,
        "albert-chinese-xxlarge": 512,
    }

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super(AlbertChineseTokenizer, self).__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
