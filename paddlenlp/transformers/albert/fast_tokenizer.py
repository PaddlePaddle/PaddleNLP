# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
from shutil import copyfile
from typing import Optional, Tuple

from fast_tokenizer import normalizers

from ...utils.log import logger
from ..tokenizer_utils import AddedToken
from ..tokenizer_utils_fast import PretrainedFastTokenizer
from .tokenizer import AlbertChineseTokenizer, AlbertEnglishTokenizer, AlbertTokenizer

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "sentencepiece_model_file": "spiece.model",
    "tokenizer_file": "tokenizer.json",
}


class AlbertFastTokenizer(PretrainedFastTokenizer):
    resource_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = AlbertTokenizer
    pretrained_resource_files_map = slow_tokenizer_class.pretrained_resource_files_map
    pretrained_init_configuration = slow_tokenizer_class.pretrained_init_configuration

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        sentencepiece_model_file=None,
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
        **kwargs
    ):
        # Mask token behave like a normal word, i.e. include the space before it and
        # is included in the raw text, there should be a match in a non-normalized sentence.
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            sentencepiece_model_file=sentencepiece_model_file,
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
        self.sentencepiece_model_file = sentencepiece_model_file

        if vocab_file is None and sentencepiece_model_file is None:
            raise ValueError(
                "You should only specify either one(not both) of 'vocal_file'"
                "and 'sentencepiece_model_file' to construct an albert tokenizer."
                "Specify 'vocal_file' for Chinese tokenizer and "
                "'sentencepiece_model_file' for English tokenizer"
            )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        if self.vocab_file is None:
            out_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["sentencepiece_model_file"],
            )

            if os.path.abspath(self.sentencepiece_model_file) != os.path.abspath(out_vocab_file):
                copyfile(self.sentencepiece_model_file, out_vocab_file)

            return (out_vocab_file,)
        else:
            files = self._tokenizer.model.save(save_directory, prefix=filename_prefix)
            return tuple(files)


class AlbertChineseFastTokenizer(PretrainedFastTokenizer):
    resource_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = AlbertChineseTokenizer
    pretrained_resource_files_map = slow_tokenizer_class.pretrained_resource_files_map
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


class AlbertEnglishFastTokenizer(PretrainedFastTokenizer):
    resource_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = AlbertEnglishTokenizer
    pretrained_resource_files_map = slow_tokenizer_class.pretrained_resource_files_map
    pretrained_init_configuration = slow_tokenizer_class.pretrained_init_configuration

    def __init__(
        self,
        sentencepiece_model_file,
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
        **kwargs
    ):
        super().__init__(
            sentencepiece_model_file,
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
        self.sentencepiece_model_file = sentencepiece_model_file

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["sentencepiece_model_file"],
        )

        if os.path.abspath(self.sentencepiece_model_file) != os.path.abspath(out_vocab_file):
            copyfile(self.sentencepiece_model_file, out_vocab_file)

        return (out_vocab_file,)
