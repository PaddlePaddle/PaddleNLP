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
from typing import List, Optional, Tuple

from faster_tokenizers import normalizers
from ..tokenizer_utils_faster import PretrainedFasterTokenizer
from .tokenizer import BertTokenizer

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "tokenizer_file": "tokenizer.json"
}


class BertFasterTokenizer(PretrainedFasterTokenizer):
    resource_files_names = VOCAB_FILES_NAMES  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "bert-base-uncased":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-uncased-vocab.txt",
            "bert-large-uncased":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-large-uncased-vocab.txt",
            "bert-base-cased":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-cased-vocab.txt",
            "bert-large-cased":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-large-cased-vocab.txt",
            "bert-base-multilingual-uncased":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-multilingual-uncased-vocab.txt",
            "bert-base-multilingual-cased":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-multilingual-cased-vocab.txt",
            "bert-base-chinese":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-chinese-vocab.txt",
            "bert-wwm-chinese":
            "http://bj.bcebos.com/paddlenlp/models/transformers/bert/bert-wwm-chinese-vocab.txt",
            "bert-wwm-ext-chinese":
            "http://bj.bcebos.com/paddlenlp/models/transformers/bert/bert-wwm-ext-chinese-vocab.txt",
            "macbert-large-chinese":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-chinese-vocab.txt",
            "macbert-base-chinese":
            "https://bj.bcebos.com/paddle-hapi/models/bert/bert-base-chinese-vocab.txt",
            "simbert-base-chinese":
            "https://bj.bcebos.com/paddlenlp/models/transformers/simbert/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "bert-base-uncased": {
            "do_lower_case": True
        },
        "bert-large-uncased": {
            "do_lower_case": True
        },
        "bert-base-cased": {
            "do_lower_case": False
        },
        "bert-large-cased": {
            "do_lower_case": False
        },
        "bert-base-multilingual-uncased": {
            "do_lower_case": True
        },
        "bert-base-multilingual-cased": {
            "do_lower_case": False
        },
        "bert-base-chinese": {
            "do_lower_case": False
        },
        "bert-wwm-chinese": {
            "do_lower_case": False
        },
        "bert-wwm-ext-chinese": {
            "do_lower_case": False
        },
        "macbert-large-chinese": {
            "do_lower_case": False
        },
        "macbert-base-chinese": {
            "do_lower_case": False
        },
        "simbert-base-chinese": {
            "do_lower_case": True
        },
    }
    slow_tokenizer_class = BertTokenizer

    padding_side = 'right'

    def __init__(self,
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
                 **kwargs):
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
            **kwargs, )

        normalizer_state = json.loads(
            self.backend_tokenizer.normalizer.__getstate__())
        if (normalizer_state.get("lowercase", do_lower_case) != do_lower_case or
                normalizer_state.get("strip_accents", strip_accents) !=
                strip_accents or normalizer_state.get(
                    "handle_chinese_chars",
                    tokenize_chinese_chars) != tokenize_chinese_chars):
            normalizer_class = getattr(normalizers,
                                       normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(
                **normalizer_state)

        self.do_lower_case = do_lower_case

    def save_vocabulary(self,
                        save_directory: str,
                        filename_prefix: Optional[str]=None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
