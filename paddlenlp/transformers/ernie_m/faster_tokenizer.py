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

import os
import json
from typing import List, Optional, Tuple
from shutil import copyfile

from faster_tokenizer import normalizers
from ..tokenizer_utils_faster import PretrainedFasterTokenizer
from .tokenizer import ErnieMTokenizer
from ...utils.log import logger

VOCAB_FILES_NAMES = {
    "sentencepiece_model_file": "sentencepiece.bpe.model",
    "vocab_file": "vocab.txt",
    "tokenizer_file": "tokenizer.json"
}

SPIECE_UNDERLINE = "▁"


class ErnieMFasterTokenizer(PretrainedFasterTokenizer):
    resource_files_names = VOCAB_FILES_NAMES  # for save_pretrained
    slow_tokenizer_class = ErnieMTokenizer
    pretrained_resource_files_map = slow_tokenizer_class.pretrained_resource_files_map
    pretrained_init_configuration = slow_tokenizer_class.pretrained_init_configuration

    def __init__(self,
                 vocab_file,
                 sentencepiece_model_file,
                 tokenizer_file=None,
                 do_lower_case=True,
                 encoding="utf8",
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):
        super().__init__(
            vocab_file,
            sentencepiece_model_file=sentencepiece_model_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.do_lower_case = do_lower_case
        self.encoding = encoding
        self.vocab_file = vocab_file
        self.sentencepiece_model_file = sentencepiece_model_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def save_vocabulary(self,
                        save_directory: str,
                        filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your faster tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer.")
        if not os.path.isdir(save_directory):
            logger.error(
                f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_sentencepiece_model_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") +
            VOCAB_FILES_NAMES["sentencepiece_model_file"])
        if os.path.abspath(self.sentencepiece_model_file) != os.path.abspath(
                out_sentencepiece_model_file):
            copyfile(self.sentencepiece_model_file,
                     out_sentencepiece_model_file)
        return (out_sentencepiece_model_file, )
