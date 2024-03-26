# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
from typing import Optional, Tuple

from tokenizers import normalizers

from ..tokenizer_utils_fast import PretrainedFastTokenizer
from .tokenizer import BertTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class BertFastTokenizer(PretrainedFastTokenizer):
    resource_files_names = VOCAB_FILES_NAMES  # for save_pretrained
    slow_tokenizer_class = BertTokenizer
    pretrained_resource_files_map = slow_tokenizer_class.pretrained_resource_files_map
    pretrained_resource_files_map.update(
        {
            "tokenizer_file": {
                "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json",
                "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/tokenizer.json",
                "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json",
                "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/tokenizer.json",
                "bert-base-multilingual-uncased": (
                    "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/tokenizer.json"
                ),
                "bert-base-multilingual-cased": (
                    "https://huggingface.co/bert-base-multilingual-cased/resolve/main/tokenizer.json"
                ),
                "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/tokenizer.json",
                "bert-wwm-chinese": "fake/tokenizer.json",
                "bert-wwm-ext-chinese": "fake/tokenizer.json",
                "macbert-large-chinese": "fake/tokenizer.json",
                "macbert-base-chinese": "fake/tokenizer.json",
                "simbert-base-chinese": "fake/tokenizer.json",
                "uer/chinese-roberta-base": "fake/tokenizer.json",
                "uer/chinese-roberta-medium": "fake/tokenizer.json",
                "uer/chinese-roberta-6l-768h": "fake/tokenizer.json",
                "uer/chinese-roberta-small": "fake/tokenizer.json",
                "uer/chinese-roberta-mini": "fake/tokenizer.json",
                "uer/chinese-roberta-tiny": "fake/tokenizer.json",
            }
        }
    )
    pretrained_init_configuration = slow_tokenizer_class.pretrained_init_configuration

    padding_side = "right"

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if "normalizers" in normalizer_state:
            normalizer_state = normalizer_state["normalizers"][0]
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        self.do_lower_case = do_lower_case

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, prefix=filename_prefix)
        return tuple(files)
