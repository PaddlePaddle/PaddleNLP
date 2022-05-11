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
from .tokenizer import ErnieTokenizer

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "tokenizer_file": "tokenizer.json"
}


class ErnieFasterTokenizer(PretrainedFasterTokenizer):
    resource_files_names = VOCAB_FILES_NAMES  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-1.0":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/vocab.txt",
            "ernie-tiny":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_tiny/vocab.txt",
            "ernie-2.0-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_base/vocab.txt",
            "ernie-2.0-en-finetuned-squad":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_base/vocab.txt",
            "ernie-2.0-large-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_v2_large/vocab.txt",
            "ernie-gen-base-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-gen-base-en/vocab.txt",
            "ernie-gen-large-en":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-gen-large/vocab.txt",
            "ernie-gen-large-en-430g":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie-gen-large-430g/vocab.txt",
            "rocketqa-zh-dureader-query-encoder":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa-zh-dureader-vocab.txt",
            "rocketqa-zh-dureader-para-encoder":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa-zh-dureader-vocab.txt",
            "rocketqa-v1-marco-query-encoder":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa-v1-marco-vocab.txt",
            "rocketqa-v1-marco-para-encoder":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa-v1-marco-vocab.txt",
            "rocketqa-zh-dureader-cross-encoder":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa-zh-dureader-vocab.txt",
            "rocketqa-v1-marco-cross-encoder":
            "https://bj.bcebos.com/paddlenlp/models/transformers/rocketqa/rocketqa-v1-marco-vocab.txt",
            "ernie-3.0-base-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh_vocab.txt",
            "ernie-3.0-medium-zh":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh_vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie-1.0": {
            "do_lower_case": True
        },
        "ernie-tiny": {
            "do_lower_case": True
        },
        "ernie-2.0-en": {
            "do_lower_case": True
        },
        "ernie-2.0-en-finetuned-squad": {
            "do_lower_case": True
        },
        "ernie-2.0-large-en": {
            "do_lower_case": True
        },
        "ernie-gen-base-en": {
            "do_lower_case": True
        },
        "ernie-gen-large-en": {
            "do_lower_case": True
        },
        "ernie-gen-large-en-430g": {
            "do_lower_case": True
        },
        "ppminilm-6l-768h": {
            "do_lower_case": True
        },
        "rocketqa-zh-dureader-query-encoder": {
            "do_lower_case": True
        },
        "rocketqa-zh-dureader-para-encoder": {
            "do_lower_case": True
        },
        "rocketqa-v1-marco-query-encoder": {
            "do_lower_case": True
        },
        "rocketqa-v1-marco-para-encoder": {
            "do_lower_case": True
        },
        "rocketqa-zh-dureader-cross-encoder": {
            "do_lower_case": True
        },
        "rocketqa-v1-marco-cross-encoder": {
            "do_lower_case": True
        },
        "ernie-3.0-base-zh": {
            "do_lower_case": True
        },
        "ernie-3.0-medium-zh": {
            "do_lower_case": True
        },
    }
    slow_tokenizer_class = ErnieTokenizer
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
        files = self._tokenizer.model.save(save_directory, filename_prefix)
        return tuple(files)
