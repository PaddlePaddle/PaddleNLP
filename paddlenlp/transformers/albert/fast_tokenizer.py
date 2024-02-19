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

import logging
import os
from shutil import copyfile
from typing import Optional, Tuple

from ..tokenizer_utils_fast import PretrainedFastTokenizer
from .tokenizer import AlbertEnglishTokenizer

VOCAB_FILES_NAMES = {"sentencepiece_model_file": "spiece.model", "tokenizer_file": "tokenizer.json"}
logger = logging.getLogger(__name__)


class AlbertEnglishFastTokenizer(PretrainedFastTokenizer):
    resource_files_names = VOCAB_FILES_NAMES  # for save_pretrained
    slow_tokenizer_class = AlbertEnglishTokenizer
    pretrained_resource_files_map = slow_tokenizer_class.pretrained_resource_files_map
    pretrained_resource_files_map.update(
        {
            "tokenizer_file": {
                "albert-base-v1": "fake/tokenizer.json",
                "albert-large-v1": "fake/tokenizer.json",
                "albert-xlarge-v1": "fake/tokenizer.json",
                "albert-xxlarge-v1": "fake/tokenizer.json",
                "albert-base-v2": "fake/tokenizer.json",
                "albert-large-v2": "fake/tokenizer.json",
                "albert-xlarge-v2": "fake/tokenizer.json",
                "albert-xxlarge-v2": "fake/tokenizer.json",
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
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
